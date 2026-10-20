"""Frame selection over the raw OpenFace CSVs.

With real frames in hand the curation problem is selection, not synthesis: every
candidate is an observed human face, so realism is free and the only question is
which frames to keep. The pipeline is

    load -> drop tracking failures (implausible pose, or a frame index repeated
            within a single-face track) -> temporal dedup
         -> hybrid farthest-point / random selection -> lateralize -> frames.json

Pure farthest-point sampling maximizes spread but drifts to the boundary of the
distribution; uniform random keeps the density but reproduces the original
complaint (almost no high-intensity expressions). The default 70/30 hybrid keeps
the support without abandoning the dense core.

OpenFace's AU*_r columns are bilateral: one intensity per AU, no side. Every
frame we select is therefore a perfectly symmetric expression, while the ARKit
rig it drives has separate left/right targets for six of the AUs we track. That
half of the label space is unreachable from the data, so asymmetry is
synthesized rather than recovered -- see lateralize(). It runs *after* selection
so the selection itself is untouched: still K frames, still the 70/30 hybrid,
still scored in the 17-D symmetric space the CSVs actually live in.

Usage:
    python src/TemporalDedup.py                  # K=2000, tau=1.0, 70% FPS, 35% asymmetric
    python src/TemporalDedup.py --k 5000 --report
    python src/TemporalDedup.py --asym-frac 0.0  # symmetric output, as before
"""

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(BASE_DIR, 'data', 'csv')
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'Selection')

# Tracking-failure gates, in degrees. OpenFace writes pose_R* in radians:
# Rx is pitch, Ry is yaw, Rz is roll.
MAX_PITCH = 40.0
MAX_ROLL = 30.0
MAX_YAW = 50.0

# An expression counts as high intensity when any single AU exceeds this.
HIGH_INTENSITY = 4.0
# An AU counts as active above this. OpenFace rarely reports an exact zero, so
# counting everything above 0.0 inflates the count with sub-perceptual noise
# (8.94 mean active AUs on the clean set, against 5.47 at this threshold).
ACTIVE = 0.5

# AUs the Unity AU->ARKit map splits into separate left/right blendshapes. Only
# these can carry asymmetry; the rest drive a single shared target and stay
# bilateral. AU45/AU06/AU07/AU04/AU15/AU20 have L/R ARKit targets too (blink,
# cheekSquint, eyeSquint, browDown, mouthFrown, mouthStretch) and can be added
# here once the map gains AU45L/AU45R-style keys for them.
LATERAL_AUS = ('AU01', 'AU02', 'AU09', 'AU10', 'AU12', 'AU14')
# Share of the selected frames that get lateralized.
ASYM_FRAC = 0.35
# Of those, the share that get a near-unilateral action (a smirk, a single brow)
# rather than the mild hemiface imbalance most real expressions carry.
STRONG_ASYM_FRAC = 0.35
# Within a lateralized frame, the chance each active lateral AU joins the split.
# Below 1.0 so an asymmetric frame is usually asymmetric in only part of the
# face, which is what unilateral action actually looks like.
PER_AU_ASYM_PROB = 0.6


def parse_segment_name(csv_file):
    """'10021_to_10176_segment_speaker.csv' -> ('10021_to_10176', 'speaker')."""
    stem = os.path.splitext(csv_file)[0]
    segment, _, role = stem.rpartition('_segment_')
    return segment, role


def load_csv_dir(csv_dir):
    """Read every CSV into flat per-frame arrays plus the AU column names.

    Returns (aus, pose, frame_idx, track_id, role_id, csv_row, au_names,
    tracks, roles) where aus is (N, n_aus) float32 intensities and pose is
    (N, 3) degrees ordered pitch/yaw/roll.

    One CSV is one track: a segment name appears twice in the directory, once
    per role, and the two files hold two different faces of the same
    conversation. track_id is therefore per file, not per segment name -- the
    grouping the downstream single-face assumptions need -- and tracks[track_id]
    gives back the segment name, duplicated across the two roles that share it.

    csv_row is the 0-based data row, kept because a repeated frame index is not
    on its own a unique key inside a track (see repeated_frames) and provenance
    should survive that.
    """
    csv_files = sorted(f for f in os.listdir(csv_dir) if f.endswith('.csv'))
    if not csv_files:
        raise SystemExit('no CSVs found in %s' % csv_dir)

    with open(os.path.join(csv_dir, csv_files[0]), 'r') as f:
        header = f.readline().strip().split(',')
    au_cols = [i for i, col in enumerate(header) if col.startswith('AU') and col.endswith('_r')]
    au_names = [header[i].replace('_r', '') for i in au_cols]
    pose_cols = [header.index(c) for c in ('pose_Rx', 'pose_Ry', 'pose_Rz')]
    frame_col = header.index('frame')

    aus, pose, frame_idx, track_id, role_id, csv_row = [], [], [], [], [], []
    tracks, roles = [], []
    role_lookup = {}

    for csv_file in csv_files:
        segment, role = parse_segment_name(csv_file)
        if role not in role_lookup:
            role_lookup[role] = len(roles)
            roles.append(role)
        tid = len(tracks)
        tracks.append(segment)

        with open(os.path.join(csv_dir, csv_file), 'r') as f:
            f.readline()  # header, assumed identical across files
            row = 0
            for line in f:
                if not line.strip():
                    continue
                values = line.split(',')
                aus.append([float(values[i]) for i in au_cols])
                pose.append([float(values[i]) for i in pose_cols])
                frame_idx.append(int(values[frame_col]))
                track_id.append(tid)
                role_id.append(role_lookup[role])
                csv_row.append(row)
                row += 1

    return (
        np.asarray(aus, dtype=np.float32),
        np.degrees(np.asarray(pose, dtype=np.float32)),
        np.asarray(frame_idx, dtype=np.int32),
        np.asarray(track_id, dtype=np.int32),
        np.asarray(role_id, dtype=np.int8),
        np.asarray(csv_row, dtype=np.int32),
        au_names,
        tracks,
        roles,
    )


def repeated_frames(frame_idx, track_id):
    """Mask of rows whose frame index is not unique within its track.

    A track covers exactly one face, so a frame index can legitimately appear
    only once. In 226 of the CSVs (all listener) it appears twice or three times
    from some point onward, interleaved in non-decreasing order: the tracker
    spawned a spurious extra detection, and the export carries no face_id to say
    which row is the real subject.

    Every occurrence is dropped, not just the extras. Occurrence order is not
    reliably bound to identity here -- at the transitions where it can be tested,
    the first occurrence continues the established head trajectory in only about
    three quarters of cases -- so keeping one would be a guess that silently
    admits the wrong face a quarter of the time.
    """
    key = track_id.astype(np.int64) * (int(frame_idx.max()) + 1) + frame_idx
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    return counts[inverse] > 1


def pose_ok(pose_deg):
    """Boolean mask of frames whose head pose is inside the plausible range."""
    pitch, yaw, roll = pose_deg[:, 0], pose_deg[:, 1], pose_deg[:, 2]
    return (
        (np.abs(pitch) <= MAX_PITCH)
        & (np.abs(roll) <= MAX_ROLL)
        & (np.abs(yaw) <= MAX_YAW)
    )


def temporal_dedup(aus, frame_idx, group_id, tau, reset_on_gap=False):
    """Keep a frame only once its AU vector has moved tau from the last kept one.

    Neighbouring frames of 30fps video are near-duplicates, so the reference is
    the last *kept* frame rather than the previous frame: that drops slow drift
    as well as literal repeats. Rows are walked in frame order within a group_id
    (a track, i.e. one face) and the reference resets between groups, so one
    person is never compared against another.

    Gaps in the frame index (dropped detections, ~8% of transitions) do not
    reset the reference by default: redundancy here is a claim about AU space,
    not about time, so a frame that lands back inside tau is still a duplicate
    of what we already kept. Resetting on gaps too keeps ~1% more frames.
    """
    keep = np.zeros(len(aus), dtype=bool)
    tau_sq = tau * tau
    order = np.lexsort((frame_idx, group_id))
    ref = None
    prev_group = None
    prev_frame = 0

    for i in order:
        contiguous = group_id[i] == prev_group and (
            frame_idx[i] == prev_frame + 1 or not reset_on_gap)
        if not contiguous:
            ref = None
        if ref is None or float(np.sum((aus[i] - ref) ** 2)) > tau_sq:
            keep[i] = True
            ref = aus[i]
        prev_group = group_id[i]
        prev_frame = frame_idx[i]

    return keep


def farthest_point_order(points, k, rng):
    """Greedy max-min selection; returns k indices into points (the action units values), in pick order.

    Each prefix of the result is itself the farthest-point set of that size,
    which is what lets the hybrid reuse a single pass.
    """
    n = len(points)
    k = min(k, n)
    picked = np.empty(k, dtype=np.int64)
    if k == 0:
        return picked

    current = int(rng.integers(n)) #first, pick one random point
    picked[0] = current
    min_dist_sq = np.sum((points - points[current]) ** 2, axis=1) #returns the index of the furthest point from current, largest distance

    for i in range(1, k):
        current = int(np.argmax(min_dist_sq))
        picked[i] = current
        np.minimum(min_dist_sq, np.sum((points - points[current]) ** 2, axis=1), out=min_dist_sq)

    return picked


def select_hybrid(points, k, fps_frac, rng):
    """fps_frac of k by farthest-point, the rest uniform from what is left."""
    k = min(k, len(points))
    n_fps = min(int(round(k * fps_frac)), k)
    order = farthest_point_order(points, n_fps, rng)

    remaining = np.setdiff1d(np.arange(len(points)), order, assume_unique=False)
    n_random = min(k - n_fps, len(remaining))
    extra = rng.choice(remaining, size=n_random, replace=False) if n_random else np.empty(0, np.int64)

    return np.concatenate([order, extra.astype(np.int64)]), n_fps


def mean_nn_distance(points):
    """Mean distance from each selected point to its nearest other selection.

    Higher is more spread out. Computed in blocks so K in the thousands stays
    cheap without materializing the full pairwise matrix.
    """
    n = len(points)
    if n < 2:
        return 0.0
    total = 0.0
    for start in range(0, n, 512):
        block = points[start:start + 512]
        d = np.sqrt(np.maximum(
            np.sum(block ** 2, axis=1)[:, None]
            + np.sum(points ** 2, axis=1)[None, :]
            - 2.0 * block @ points.T, 0.0))
        d[np.arange(len(block)), np.arange(start, start + len(block))] = np.inf
        total += float(np.sum(np.min(d, axis=1)))
    return total / n


def coverage_distance(reference, selected):
    """Mean distance from reference frames to the nearest selected frame.

    Lower is better: it measures whether the selection still represents the
    dense core of the distribution, which is exactly what pure farthest-point
    sampling gives up in exchange for spread.
    """
    total = 0.0
    for start in range(0, len(reference), 2048):
        block = reference[start:start + 2048]
        d = np.sqrt(np.maximum(
            np.sum(block ** 2, axis=1)[:, None]
            + np.sum(selected ** 2, axis=1)[None, :]
            - 2.0 * block @ selected.T, 0.0))
        total += float(np.sum(np.min(d, axis=1)))
    return total / len(reference)


def describe(aus):
    """Summary stats used for both the selections and the reference rows."""
    return {
        'frames': int(len(aus)),
        'high_intensity_frac': float(np.mean(np.max(aus, axis=1) > HIGH_INTENSITY)),
        'mean_active_aus': float(np.mean(np.sum(aus > ACTIVE, axis=1))),
        'mean_intensity': float(np.mean(aus)),
    }


def allocate(pool_sizes, k):
    """Split k across strata proportional to pool size, largest remainder first."""
    total = sum(pool_sizes)
    if total == 0:
        return [0] * len(pool_sizes)
    exact = [k * size / total for size in pool_sizes]
    counts = [min(int(math.floor(e)), size) for e, size in zip(exact, pool_sizes)]
    order = sorted(range(len(exact)), key=lambda i: exact[i] - math.floor(exact[i]), reverse=True)
    leftover = k - sum(counts)
    for i in order:
        if leftover <= 0:
            break
        if counts[i] < pool_sizes[i]:
            counts[i] += 1
            leftover -= 1
    return counts


def choose_asymmetric(eligible, n_fps, frac):
    """Mask of rows to lateralize, split across the FPS and random halves.

    The asymmetry budget is allocated between the two halves of the hybrid in
    proportion to how many eligible frames each contributed, so the asymmetric
    subset carries the same 70/30 composition as the selection it comes from --
    otherwise the L/R half of the label space would be covered only by whichever
    half of the selection happened to get picked.

    Inside the FPS half the picks are evenly spaced over *pick order*. That
    order ranks frames by how much they extended the spread, so a stride keeps
    the asymmetric subset spanning the whole range rather than piling onto the
    handful of extremes at the front.
    """
    mask = np.zeros(len(eligible), dtype=bool)
    halves = [np.flatnonzero(eligible[:n_fps]),
              np.flatnonzero(eligible[n_fps:]) + n_fps]
    quotas = allocate([len(h) for h in halves], int(round(len(eligible) * frac)))
    for half, quota in zip(halves, quotas):
        if quota:
            mask[half[np.round(np.linspace(0, len(half) - 1, quota)).astype(int)]] = True
    return mask


def lateralize(aus, lateral_cols, n_fps, frac, rng):
    """Split a share of the frames into unequal left/right AU intensities.

    Returns (left, right, asym_mask). Both sides start as the observed vector,
    so a frame that is not lateralized comes out exactly as it went in.

    The dominant side keeps the measured intensity and only the weak side is
    scaled down. Scaling both would pull the frame toward neutral and give back
    the high-intensity coverage the hybrid selection was built to buy. One
    weak side and one base ratio are drawn per frame, not per AU: real
    asymmetry is one hemiface trailing the other, not each muscle acting
    independently, so a frame with a weak-left smile and a weak-right sneer
    would read as tracking noise rather than expression.

    Only AUs already active in the frame are eligible -- lifting an inactive AU
    on one side would invent an action the subject never made.
    """
    left, right = aus.copy(), aus.copy()
    # index:     0 ............... 3499 | 3500 ......... 4999
    #        |<--- FPS, in pick --->|<--- random,  -->|
    #        |     order            |    arbitrary    |
    eligible = np.any(aus[:, lateral_cols] > ACTIVE, axis=1) # boolean of length 5000 in exactly that order 
    targets = choose_asymmetric(eligible, n_fps, frac) if frac > 0 else np.zeros(len(aus), bool)

    for i in np.flatnonzero(targets):
        active = [c for c in lateral_cols if aus[i, c] > ACTIVE]
        take = [c for c in active if rng.random() < PER_AU_ASYM_PROB] or \
            [max(active, key=lambda c: aus[i, c])] #If no AU is selected, pick the strongest one
        base = (rng.uniform(0.0, 0.25) if rng.random() < STRONG_ASYM_FRAC
                else rng.uniform(0.30, 0.75))
        weak = left if rng.random() < 0.5 else right
        for c in take:
            ratio = float(np.clip(base + rng.uniform(-0.10, 0.10), 0.0, 0.85))
            weak[i, c] = aus[i, c] * ratio

    return left, right, targets


def asymmetry_report(left, right, asym, methods, au_names, lateral_cols):
    """Counts and laterality ratios for the lateralized frames."""
    weaker = np.minimum(left, right)
    stronger = np.maximum(left, right)
    split = stronger > weaker
    ratios = weaker[split] / np.maximum(stronger[split], 1e-6)
    per_au = {au_names[c]: int(split[:, c].sum()) for c in lateral_cols}
    fps = np.asarray([m == 'fps' for m in methods])
    return {
        'lateral_aus': [au_names[c] for c in lateral_cols],
        'frames': int(asym.sum()),
        'frac': round(float(asym.mean()), 3) if len(asym) else 0.0,
        'from_fps': int((asym & fps).sum()),
        'from_random': int((asym & ~fps).sum()),
        'split_aus_per_frame': round(float(split.sum() / max(int(asym.sum()), 1)), 2),
        'mean_laterality_ratio': round(float(ratios.mean()), 3) if len(ratios) else 0.0,
        'near_unilateral_frac': round(float(np.mean(ratios < 0.25)), 3) if len(ratios) else 0.0,
        'split_frames_per_au': per_au,
    }


# def build_records(indices, aus, pose, frame_idx, track_id, role_id, csv_row,
#                   au_names, tracks, roles, methods):
#     """One JSON record per selected frame: identity, AU intensities, pose, role."""
#     records = []
#     for i, method in zip(indices, methods):
#         records.append({
#             'segment': tracks[track_id[i]],
#             'frame': int(frame_idx[i]),
#             'role': roles[role_id[i]],
#             'source_csv': '%s_segment_%s.csv' % (tracks[track_id[i]], roles[role_id[i]]),
#             'csv_row': int(csv_row[i]),
#             'selection': method,
#             'head_pose_deg': {
#                 'pitch': round(float(pose[i, 0]), 2),
#                 'yaw': round(float(pose[i, 1]), 2),
#                 'roll': round(float(pose[i, 2]), 2),
#             },
#             'aus': {name: round(float(v), 2) for name, v in zip(au_names, aus[i])},
#         })
#     records.sort(key=lambda r: (r['segment'], r['role'], r['frame']))
#     return records

def build_records(indices, left, right, frame_idx, au_names, lateral_cols):
    """One JSON record per selected frame: frame index and AU intensities.

    An AU that ended up lateralized is written as the AU01L/AU01R pair the Unity
    AU->ARKit map already understands; everything else keeps the bare AU01 key
    that drives both sides. The keys are the record of which frames are
    asymmetric -- no extra field for the Unity side to learn.
    """
    records = []
    for pos, i in enumerate(indices):
        values = {}
        for c, name in enumerate(au_names):
            l, r = round(float(left[pos, c]), 2), round(float(right[pos, c]), 2)
            if c in lateral_cols and l != r:
                values[name + 'L'], values[name + 'R'] = l, r
            else:
                values[name] = l
        records.append({'frame': int(frame_idx[i]), 'aus': values})
    return records

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv-dir', default=CSV_DIR)
    parser.add_argument('--out-dir', default=OUT_DIR)
    parser.add_argument('--k', type=int, default=2000, help='frames to select')
    parser.add_argument('--tau', type=float, default=1.0, help='temporal dedup radius in AU space')
    parser.add_argument('--fps-frac', type=float, default=0.7,
                        help='fraction of K taken by farthest-point sampling')
    parser.add_argument('--asym-frac', type=float, default=ASYM_FRAC,
                        help='fraction of the selected frames given left/right asymmetry')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--reset-on-gap', action='store_true',
                        help='restart the dedup reference after a dropped detection')
    parser.add_argument('--report', action='store_true',
                        help='also score uniform-random and pure-FPS baselines')
    parser.add_argument('--reference-size', type=int, default=20000,
                        help='clean frames sampled to measure coverage')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    # Separate stream for lateralization, so changing --asym-frac cannot shift
    # which frames the hybrid picks.
    asym_rng = np.random.default_rng([args.seed, 1])

    (aus, pose, frame_idx, track_id, role_id, csv_row,
     au_names, tracks, roles) = load_csv_dir(args.csv_dir)
    print('loaded %d frames from %d tracks over %d segments'
          % (len(aus), len(tracks), len(set(tracks))))

    # Two independent signs of a failed track: an implausible head pose, and a
    # frame index appearing more than once in a track that holds one face.
    bad_pose = ~pose_ok(pose)
    duplicated = repeated_frames(frame_idx, track_id)
    ok = ~(bad_pose | duplicated)
    n_pose, n_dup, n_failed = int(bad_pose.sum()), int(duplicated.sum()), int((~ok).sum())
    print('tracking failures: %d (%.1f%%)  [pose %d, repeated frame index %d]'
          % (n_failed, 100.0 * n_failed / len(aus), n_pose, n_dup))
    print('clean frames: %d' % int(ok.sum()))

    clean = np.flatnonzero(ok)
    keep = temporal_dedup(aus[clean], frame_idx[clean], track_id[clean], args.tau,
                          reset_on_gap=args.reset_on_gap)
    pool = clean[keep]
    print('after temporal dedup (tau=%.2f): %d (%.0f%% of clean)'
          % (args.tau, len(pool), 100.0 * len(pool) / len(clean)))

    lateral_cols = [au_names.index(a) for a in LATERAL_AUS if a in au_names]

    # Role is not a stratum. Speaker and listener are two ways of arriving at an
    # expression, not two label spaces, and the AU vector is the same 17-D space
    # either way -- so the hybrid runs once over the whole deduped pool and lets
    # the geometry decide the mix. Whatever the roles contribute is recorded
    # below, but nothing is reserved for either.
    local, n_fps = select_hybrid(aus[pool], args.k, args.fps_frac, rng)
    chosen = pool[local]
    methods = ['fps'] * n_fps + ['random'] * (len(local) - n_fps)

    # After selection, so the frames themselves are unchanged.
    left, right, asym = lateralize(aus[chosen], lateral_cols, n_fps,
                                   args.asym_frac, asym_rng)
    print('selected %d of %d (%d fps + %d random, %d asymmetric)'
          % (len(chosen), len(pool), n_fps, len(local) - n_fps, int(asym.sum())))
    for r, role in enumerate(roles):
        in_role = role_id[chosen] == r
        print('  %-8s pool=%6d selected=%4d (%d fps + %d random)'
              % (role, int((role_id[pool] == r).sum()), int(in_role.sum()),
                 int(in_role[:n_fps].sum()), int(in_role[n_fps:].sum())))

    reference = aus[rng.choice(clean, size=min(args.reference_size, len(clean)), replace=False)]
    rows = [('hybrid %d%% FPS + %d%% random' % (round(args.fps_frac * 100),
                                                round((1 - args.fps_frac) * 100)), chosen)]

    if args.report:
        # Baselines are drawn from the same deduped pool, so the comparison
        # isolates the selection rule itself.
        uniform = pool[rng.choice(len(pool), size=min(args.k, len(pool)), replace=False)]
        pure = pool[farthest_point_order(aus[pool], args.k, np.random.default_rng(args.seed))]
        rows = [('uniform random', uniform), ('farthest-point', pure)] + rows

    metrics = []
    for name, idx in rows:
        stats = describe(aus[idx])
        stats.update({
            'name': name,
            'spread': round(mean_nn_distance(aus[idx]), 3),
            'coverage': round(coverage_distance(reference, aus[idx]), 3),
        })
        metrics.append(stats)

    clean_stats = describe(aus[clean])
    clean_stats['name'] = 'full clean set (reference)'
    metrics.append(clean_stats)

    print('\n%-34s %8s %10s %13s %10s' % ('', 'spread', 'coverage', 'AU>4 frames', 'mean AUs'))
    for m in metrics:
        print('%-34s %8s %10s %12.1f%% %10.2f'
              % (m['name'],
                 '%.3f' % m['spread'] if 'spread' in m else '',
                 '%.3f' % m['coverage'] if 'coverage' in m else '',
                 100.0 * m['high_intensity_frac'],
                 m['mean_active_aus']))

    asym_stats = asymmetry_report(left, right, asym, methods, au_names, lateral_cols)
    print('\nlateralized %d/%d frames (%d fps + %d random), %.2f AUs split per frame, '
          'mean weak/strong ratio %.2f'
          % (asym_stats['frames'], len(chosen), asym_stats['from_fps'],
             asym_stats['from_random'], asym_stats['split_aus_per_frame'],
             asym_stats['mean_laterality_ratio']))

    records = build_records(chosen, left, right, frame_idx, au_names, lateral_cols)
    os.makedirs(args.out_dir, exist_ok=True)
    config_path = os.path.join(args.out_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
           'config': {
                        'k': args.k,
                        'tau': args.tau,
                        'fps_frac': args.fps_frac,
                        'asym_frac': args.asym_frac,
                        'seed': args.seed,
                        'reset_on_gap': args.reset_on_gap,
                        'active_au_threshold': ACTIVE,
                        'high_intensity_threshold': HIGH_INTENSITY,
                        'pose_limits_deg': {'pitch': MAX_PITCH, 'yaw': MAX_YAW, 'roll': MAX_ROLL},
                        'au_names': au_names,
                    },
                    'counts': {
                        'total_frames': int(len(aus)),
                        'tracking_failures': n_failed,
                        'failures_pose': n_pose,
                        'failures_repeated_frame': n_dup,
                        'clean_frames': int(len(clean)),
                        'deduped_pool': int(len(pool)),
                        'selected': int(len(records)),
                    },
                    'metrics': metrics,
                    'asymmetry': asym_stats,
        }, f, indent=2)
    frames_path = os.path.join(args.out_dir, 'frames.json')
    with open(frames_path, 'w') as f:
        json.dump({
            'frames': records,
        }, f, indent=2)
    print('\nwrote %d frames to %s' % (len(records), frames_path))


if __name__ == '__main__':
    main()
