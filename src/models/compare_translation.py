"""Score facemesh -> deformation translations against the paired ground truth.

The two domains are paired by file stem -- delta_maps/00042.npy was rasterized
from the same frame that rendered DeformationMaps/00042.exr -- so a translation
has a ground truth to be checked against, which neither --eval (reconstruction
only) nor --preview (no reference) in infer_comb.py does.

For N randomly drawn stems this reports, per sample and in aggregate:

  * value differences: MAE / RMSE / max|err| per channel, both in the network's
    normalized units and in the source units the deformation .exr stores;
  * the same numbers for the zero predictor (i.e. the mean |ground truth|),
    because an error is only small relative to the signal it is meant to carry;
  * heatmaps: facemesh input, ground truth, prediction, |error|, signed error.

All pixel statistics are taken over the ground truth's own coverage mask. The
prediction is masked with the deformation domain's canvas mask (sample 0, the
same stand-in translate_map() uses, since the decoder predicts no mask); the
disagreement between the two masks is reported as `mask_iou`.

Usage:
    python src/models/compare_translation.py \
        --ckpt models/checkpoints/run_11656/comb_step470000.pt -n 10
"""

import argparse
import csv
import os
import random
import sys

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_comb import load_checkpoint, render_map, target_mask  # noqa: E402
from train_model import _load_offset_map  # noqa: E402

CHANNELS = ("dx", "dy", "dz")


def paired_stems(cfg, src="facemesh", dst="deformation"):
    """Stems present in both domains, sorted -- the pairing is by file name."""
    src_dir, dst_dir = cfg["domains"][src]["path"], cfg["domains"][dst]["path"]
    src_stems = {os.path.splitext(f)[0] for f in os.listdir(src_dir) if f.endswith(".npy")}
    dst_stems = {os.path.splitext(f)[0] for f in os.listdir(dst_dir) if f.endswith(".exr")}
    return sorted(src_stems & dst_stems), src_dir, dst_dir


@torch.no_grad()
def translate(model, scales, device, src_paths, src="facemesh", dst="deformation"):
    """Normalize, encode, decode into `dst`; returns normalized (N, 3, H, W).

    Exactly the preprocessing DomainImageDataset does -- divide by the source
    domain's scale, clamp, mask -- because that is the only input form the
    shared encoder has ever seen.
    """
    xs = []
    for path in src_paths:
        offsets, mask = _load_offset_map(path, src)
        x = torch.from_numpy(offsets).permute(2, 0, 1).contiguous()
        m = torch.from_numpy(mask).permute(2, 0, 1).contiguous()
        xs.append((x / scales[src]).clamp(-1.0, 1.0) * m)
    x = torch.stack(xs).to(device)
    return x, model.translate(x, dst)


def stats(err, gt, pred, mask):
    """Per-channel and overall error statistics over `mask` (H, W) bool.

    `err`, `gt` and `pred` are (3, H, W) in the same units. Two columns ride
    along to keep the error interpretable:

      * `gt_mae`, the zero predictor's error -- an MAE is only small relative
        to the signal it is meant to reproduce;
      * `corr` and `gain`, the correlation with the ground truth and the
        least-squares gain <pred.gt>/<gt.gt>. They separate the two ways a
        prediction can be wrong: gain < 1 with high corr is the right
        deformation drawn too faintly, low corr is the wrong deformation.
    """
    e, g, p = err[:, mask], gt[:, mask], pred[:, mask]  # (3, P)
    ec, gc = e.ravel(), g.ravel()
    out = {
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "max": float(np.abs(e).max()),
        "gt_mae": float(np.abs(g).mean()),              # zero-predictor error
        "gt_max": float(np.abs(g).max()),
        "bias": float(ec.mean()),
        "corr": float(np.corrcoef(p.ravel(), gc)[0, 1]),
        "gain": float((p.ravel() @ gc) / max(float(gc @ gc), 1e-20)),
    }
    for c, name in enumerate(CHANNELS):
        out[f"mae_{name}"] = float(np.abs(e[c]).mean())
        out[f"gt_mae_{name}"] = float(np.abs(g[c]).mean())
    return out


def figure(rows, out_path, step, scale, err_vmax):
    """One row per sample: input, ground truth, prediction, |err|, signed err.

    Maps are drawn in normalized units with the render_delta_map convention
    (mid-gray zero, darker negative, brighter positive, off-mask black). The
    error panels share one scale across all rows so the rows are comparable to
    each other, not each to itself.
    """
    n = len(rows)
    fig, axes = plt.subplots(n, 5, figsize=(3.2 * 5, 3.35 * n))
    axes = np.atleast_2d(axes)

    for r, row in enumerate(rows):
        m_gt = row["mask"]
        panels = [
            ("facemesh input", render_map(row["x"], m_gt[None] * 0 + 1)),
            ("deformation GT", render_map(row["gt_n"], m_gt[None])),
            ("prediction", render_map(row["pred_n"], m_gt[None])),
            (None, None),                                   # |error|, own colormap
            ("signed error", render_map(row["err_n"] / err_vmax, m_gt[None])),
        ]
        for c, (title, img) in enumerate(panels):
            ax = axes[r][c]
            if img is None:
                heat = np.abs(row["err_n"]).mean(axis=0)
                heat[~m_gt] = np.nan
                im = ax.imshow(heat, cmap="inferno", vmin=0, vmax=err_vmax)
                title = f"|error|  0 .. {err_vmax:.3f}"
                if r == n - 1:
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            else:
                ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=11)
        s = row["stats_n"]
        axes[r][0].set_ylabel(f"{row['stem']}\nMAE {s['mae']:.4f} "
                              f"({s['mae'] / s['gt_mae']:.0%} of |GT|)\n"
                              f"corr {s['corr']:.2f}  gain {s['gain']:.2f}", fontsize=9)

    fig.suptitle(f"facemesh -> deformation vs. ground truth  @ cycle {step}  "
                 f"-- normalized units (x{scale:.4g} = source units); "
                 f"signed error panel amplified x{1 / err_vmax:.0f}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / n))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def summary_figure(rows, out_path, err_vmax):
    """Per-sample MAE against the zero predictor, and the error distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    stems = [r["stem"] for r in rows]
    idx = np.arange(len(rows))

    mae = [r["stats_n"]["mae"] for r in rows]
    gt_mae = [r["stats_n"]["gt_mae"] for r in rows]
    axes[0].bar(idx - 0.2, gt_mae, 0.4, label="|GT| (zero predictor)", color="#bbb")
    axes[0].bar(idx + 0.2, mae, 0.4, label="model MAE", color="#c1443c")
    axes[0].set_xticks(idx); axes[0].set_xticklabels(stems, rotation=60, fontsize=8)
    axes[0].set_ylabel("normalized units"); axes[0].legend(fontsize=9)
    axes[0].set_title("error vs. signal, per sample")

    width = 0.26
    for k, name in enumerate(CHANNELS):
        axes[1].bar(idx + (k - 1) * width, [r["stats_n"][f"mae_{name}"] for r in rows],
                    width, label=name)
    axes[1].set_xticks(idx); axes[1].set_xticklabels(stems, rotation=60, fontsize=8)
    axes[1].set_ylabel("MAE (normalized)"); axes[1].legend(fontsize=9)
    axes[1].set_title("per-channel MAE")

    allerr = np.concatenate([r["err_n"][:, r["mask"]].ravel() for r in rows])
    axes[2].hist(allerr, bins=160, range=(-4 * err_vmax, 4 * err_vmax), color="#3c6ec1")
    axes[2].axvline(0, color="k", lw=0.8)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("signed error (normalized)"); axes[2].set_ylabel("pixels (log)")
    axes[2].set_title(f"error distribution  (bias {allerr.mean():+.4f})")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="models/checkpoints/run_11656/comb_step470000.pt")
    ap.add_argument("-n", type=int, default=10, help="how many maps to draw")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the draw")
    ap.add_argument("--stems", nargs="*", help="use these stems instead of a random draw")
    ap.add_argument("-o", "--out", default="logs/inference/compare")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, scales, step = load_checkpoint(args.ckpt, device)
    print(f"device: {device}   checkpoint: {args.ckpt} (cycle {step})")
    print(f"scales: " + "  ".join(f"{k}={v:.6g}" for k, v in scales.items()))

    stems, src_dir, dst_dir = paired_stems(cfg)
    if args.stems:
        chosen = args.stems
    else:
        chosen = sorted(random.Random(args.seed).sample(stems, min(args.n, len(stems))))
    print(f"{len(stems)} paired maps; using {len(chosen)}: {', '.join(chosen)}\n")

    src_paths = [os.path.join(src_dir, f"{s}.npy") for s in chosen]
    x, pred_n = translate(model, scales, device, src_paths)

    # The prediction lands on the deformation canvas, which the decoder cannot
    # predict a mask for; sample 0's mask is the stand-in translate_map() uses.
    m_canvas = target_mask(cfg, scales, "deformation", device)[0, 0].cpu().numpy() > 0
    pred_n = pred_n.cpu().numpy()
    x = x.cpu().numpy()
    scale = scales["deformation"]

    rows = []
    for i, stem in enumerate(chosen):
        gt, gt_mask = _load_offset_map(os.path.join(dst_dir, f"{stem}.exr"), "deformation")
        gt = gt.transpose(2, 0, 1)                       # (3, H, W), source units
        m_gt = gt_mask[:, :, 0] > 0

        pred_src = pred_n[i] * scale
        err_src = pred_src - gt                          # source units, unclamped GT
        rows.append({
            "stem": stem,
            "x": x[i],
            "gt_n": gt / scale,
            "pred_n": pred_n[i],
            "err_n": err_src / scale,
            "mask": m_gt,
            "stats_n": stats(err_src / scale, gt / scale, pred_n[i], m_gt),
            "stats_s": stats(err_src, gt, pred_src, m_gt),
            "mask_iou": float((m_gt & m_canvas).sum() / (m_gt | m_canvas).sum()),
        })

    # --- value differences -------------------------------------------------
    hdr = (f"{'stem':>8} | {'MAE':>7} {'RMSE':>7} {'max':>7} {'|GT|':>7} {'rel':>5} "
           f"{'corr':>6} {'gain':>6} | {'dx':>7} {'dy':>7} {'dz':>7} | "
           f"{'MAE(src)':>10} {'|GT|(src)':>10} {'IoU':>5}")
    print("normalized units (network space, x%.6g = source units)" % scale)
    print(hdr); print("-" * len(hdr))
    for r in rows:
        n, s = r["stats_n"], r["stats_s"]
        print(f"{r['stem']:>8} | {n['mae']:7.4f} {n['rmse']:7.4f} {n['max']:7.4f} "
              f"{n['gt_mae']:7.4f} {n['mae'] / n['gt_mae']:5.0%} "
              f"{n['corr']:6.3f} {n['gain']:6.3f} | "
              + " ".join(f"{n[f'mae_{c}']:7.4f}" for c in CHANNELS)
              + f" | {s['mae']:10.3e} {s['gt_mae']:10.3e} {r['mask_iou']:5.2f}")
    agg = {k: float(np.mean([r["stats_n"][k] for r in rows]))
           for k in rows[0]["stats_n"]}
    agg_s = {k: float(np.mean([r["stats_s"][k] for r in rows])) for k in rows[0]["stats_s"]}
    print("-" * len(hdr))
    print(f"{'mean':>8} | {agg['mae']:7.4f} {agg['rmse']:7.4f} "
          f"{max(r['stats_n']['max'] for r in rows):7.4f} {agg['gt_mae']:7.4f} "
          f"{agg['mae'] / agg['gt_mae']:5.0%} {agg['corr']:6.3f} {agg['gain']:6.3f} | "
          + " ".join(f"{agg[f'mae_{c}']:7.4f}" for c in CHANNELS)
          + f" | {agg_s['mae']:10.3e} {agg_s['gt_mae']:10.3e} "
          f"{np.mean([r['mask_iou'] for r in rows]):5.2f}")
    print("\nrel  = MAE / mean|GT|; 100% means no better than predicting zero."
          "\ncorr = correlation with the ground truth over the face mask."
          "\ngain = least-squares <pred,gt>/<gt,gt>; 1.0 = right magnitude, "
          "<1 = under-shot.")

    # --- figures + csv -----------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    err_vmax = max(float(np.percentile(
        np.concatenate([np.abs(r["err_n"][:, r["mask"]]).ravel() for r in rows]), 99.0)), 1e-6)

    grid = figure(rows, os.path.join(args.out, f"compare_step{step}_maps.png"),
                  step, scale, err_vmax)
    summ = summary_figure(rows, os.path.join(args.out, f"compare_step{step}_summary.png"),
                          err_vmax)
    csv_path = os.path.join(args.out, f"compare_step{step}.csv")
    with open(csv_path, "w", newline="") as fh:
        keys = list(rows[0]["stats_n"])
        w = csv.writer(fh)
        w.writerow(["stem", "mask_iou"] + [f"{k}_norm" for k in keys]
                   + [f"{k}_src" for k in keys])
        for r in rows:
            w.writerow([r["stem"], f"{r['mask_iou']:.4f}"]
                       + [f"{r['stats_n'][k]:.6g}" for k in keys]
                       + [f"{r['stats_s'][k]:.6g}" for k in keys])
    print(f"\nWrote {grid}\n      {summ}\n      {csv_path}")


if __name__ == "__main__":
    main()
