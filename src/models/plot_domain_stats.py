"""Plot the losses and per-domain output statistics from a training log.

train_model.py logs a stats line per domain just before each cycle line. The
statistics cover the face pixels only -- the ones inside the domain's mask,
which is also all the reconstruction loss scores:

    INFO Domain 'facemesh': max 0.9593, min -0.4210, mean 0.0500, zeros: 38328 / 3145728 (1.22%)
    INFO Domain 'deformation': max 0.3870, min -0.2284, mean 0.0014, zeros: 3 / 3145728 (0.00%)
    INFO cycle 200/6000 :: facemesh: L1=0.0329 CCL=0.0002 | deformation: L1=0.0021 CCL=0.0000 | |g|=0.776

The cycle line carries the per-domain reconstruction loss, the latent
cycle-consistency loss and the gradient norm of the update; the stats lines
carry no cycle number, so they are buffered and attributed to the next
`cycle N` line that follows them.

Together these are what make an output-saturation collapse visible: the loss
pins to a constant while max/min rail at +/-1 (both domains end in a tanh).

The log can be from a run that is still going -- everything written so far is
plotted, and the status panel reports the rate and the projected finish.

Usage:
    python src/models/plot_domain_stats.py logs/comb-11656.err
    python src/models/plot_domain_stats.py logs/comb-11656.err --from 50000
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # headless: compute nodes have no display
import matplotlib.pyplot as plt  # noqa: E402

STATS_RE = re.compile(
    r"Domain '(\w+)':\s*max\s*([-\d.eE+]+),\s*min\s*([-\d.eE+]+),\s*"
    r"mean\s*([-\d.eE+]+),\s*zeros:\s*(\d+)\s*/\s*(\d+)"
)
CYCLE_RE = re.compile(r"cycle\s+(\d+)/(\d+)")
# One "<domain>: L1=... CCL=..." segment of the cycle line, per domain.
LOSS_RE = re.compile(r"(\w+):\s*(L1|L2|MSE)=([-\d.eE+]+)\s+CCL=([-\d.eE+]+)")
GRAD_RE = re.compile(r"\|g\|=([-\d.eE+]+|nan)")
TIME_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def parse_log(path):
    """Parse one training log.

    Returns a dict with
        stats:  {domain: {"step", "max", "min", "mean", "zeros"}} (lists)
        loss:   {domain: {"step", "pix", "ccl"}}                  (lists)
        grad:   {"step", "norm"}
        crit:   {domain: criterion name, e.g. "L1"}
        total:  the run's configured iteration count
        times:  (first timestamp, last timestamp) as datetimes, or (None, None)
        notes:  WARNING/ERROR lines, which is how a spike abort announces itself
    """
    stats = defaultdict(lambda: defaultdict(list))
    loss = defaultdict(lambda: defaultdict(list))
    grad = {"step": [], "norm": []}
    crit, notes = {}, []
    total = None
    first_time = last_time = None
    pending = {}

    with open(path, "r", errors="replace") as f:
        for line in f:
            t = TIME_RE.match(line)
            if t:
                stamp = datetime.strptime(t.group(1), "%Y-%m-%d %H:%M:%S")
                first_time = first_time or stamp
                last_time = stamp

            if " WARNING " in line or " ERROR " in line:
                notes.append(line.strip())

            m = STATS_RE.search(line)
            if m:
                d, mx, mn, mean, z, tot = m.groups()
                pending[d] = (float(mx), float(mn), float(mean),
                              100.0 * int(z) / int(tot))
                continue

            c = CYCLE_RE.search(line)
            if not c:
                continue
            step, total = int(c.group(1)), int(c.group(2))

            for d, (mx, mn, mean, zpct) in pending.items():
                stats[d]["step"].append(step)
                stats[d]["max"].append(mx)
                stats[d]["min"].append(mn)
                stats[d]["mean"].append(mean)
                stats[d]["zeros"].append(zpct)
            pending = {}

            for d, name, pix, ccl in LOSS_RE.findall(line):
                crit[d] = name
                loss[d]["step"].append(step)
                loss[d]["pix"].append(float(pix))
                loss[d]["ccl"].append(float(ccl))

            g = GRAD_RE.search(line)
            if g:
                grad["step"].append(step)
                grad["norm"].append(float(g.group(1)))

    return {"stats": stats, "loss": loss, "grad": grad, "crit": crit,
            "total": total, "times": (first_time, last_time), "notes": notes}


def _window(steps, lo, hi):
    """Indices of the steps inside [lo, hi]."""
    return [i for i, v in enumerate(steps) if lo <= v <= hi]


def _kilo(v, _pos):
    """Tick label as 50k rather than 50000 -- six-digit cycle counts collide."""
    return f"{v / 1000:.0f}k" if abs(v) >= 1000 else f"{v:.0f}"


def _series(ax, per_domain, key, lo, hi, log=False):
    for d in sorted(per_domain):
        s = per_domain[d]["step"]
        keep = _window(s, lo, hi)
        if not keep:
            continue
        ax.plot([s[i] for i in keep], [per_domain[d][key][i] for i in keep],
                lw=1.3, marker=".", ms=3, label=d)
    # A log scale drops non-positive points silently, which would hide a whole
    # series once its loss reaches the log line's 4-decimal resolution.
    if log and all(v > 0 for d in per_domain for v in per_domain[d][key]):
        ax.set_yscale("log")
    _decorate(ax)


def _decorate(ax, legend=True):
    ax.set_xlabel("cycle")
    ax.grid(alpha=.3, lw=.5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_kilo))
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=9)


def status_lines(log, path):
    """Human-readable run status: progress, rate, projected finish, health."""
    loss, total = log["loss"], log["total"]
    steps = sorted({s for d in loss for s in loss[d]["step"]})
    if not steps:
        return ["no cycle lines yet"]

    last, first_time, last_time = steps[-1], *log["times"]
    out = [os.path.basename(path), f"cycles logged: {steps[0]} .. {last}"]
    if total:
        out.append(f"progress: {last}/{total}  ({100 * last / total:.1f}%)")

    if first_time and last_time and last > steps[0]:
        elapsed = (last_time - first_time).total_seconds()
        rate = last / elapsed if elapsed else 0  # cycles per second, whole run
        out.append(f"elapsed: {elapsed / 3600:.1f} h at {3600 * rate:.0f} cycles/h")
        if total and rate:
            out.append(f"remaining: ~{(total - last) / rate / 3600:.1f} h")

    for d in sorted(loss):
        pix = loss[d]["pix"]
        best = min(range(len(pix)), key=lambda i: pix[i])
        name = log["crit"].get(d, "L1")
        out.append(f"{d}: {name}={pix[-1]:.4f} now, best {pix[best]:.4f} "
                   f"@{loss[d]['step'][best]}")

    # Saturation check. Touching +/-1 is normal for a healthy tanh on data that
    # reaches the ends of its range; a collapse is that *plus* a loss pinned at
    # ~1.0, so report the reach and leave the verdict to the loss above.
    for d in sorted(log["stats"]):
        s = log["stats"][d]
        railed = sum(1 for mx, mn in zip(s["max"], s["min"])
                     if mx > 0.9999 and mn < -0.9999)
        if railed:
            out.append(f"{d}: tanh reached +/-1 at {railed}/{len(s['max'])} points")

    out.append(f"warnings/errors: {len(log['notes'])}")
    return out


def plot(path, out_path, lo=0, hi=float("inf")):
    log = parse_log(path)
    if not log["loss"] and not log["stats"]:
        sys.exit(f"No cycle or domain-stats lines found in {path}.")

    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
    crit = "/".join(sorted(set(log["crit"].values()))) or "L1"

    _series(axes[0][0], log["loss"], "pix", lo, hi, log=True)
    axes[0][0].set_title(f"reconstruction loss ({crit}, face pixels)")

    _series(axes[0][1], log["loss"], "ccl", lo, hi, log=True)
    axes[0][1].set_title("latent cycle-consistency loss")

    g = log["grad"]
    keep = _window(g["step"], lo, hi)
    axes[0][2].plot([g["step"][i] for i in keep], [g["norm"][i] for i in keep],
                    lw=1.3, marker=".", ms=3, color="tab:purple")
    axes[0][2].set_title("gradient norm |g|")
    _decorate(axes[0][2], legend=False)

    _series(axes[1][0], log["stats"], "max", lo, hi)
    axes[1][0].set_title("max output")
    _series(axes[1][1], log["stats"], "min", lo, hi)
    axes[1][1].set_title("min output")
    _series(axes[1][2], log["stats"], "mean", lo, hi)
    axes[1][2].set_title("mean output")
    _series(axes[1][3], log["stats"], "zeros", lo, hi)
    axes[1][3].set_title("zeros (% of face pixels)")

    # Saturation bounds: tanh rails at +/-1.
    for ax in (axes[1][0], axes[1][1]):
        for y in (1.0, -1.0):
            ax.axhline(y, color="crimson", ls="--", lw=1, alpha=.6)

    lines = status_lines(log, path)
    axes[0][3].axis("off")
    axes[0][3].text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
                    family="monospace", fontsize=9.5,
                    transform=axes[0][3].transAxes)

    fig.suptitle(f"Comb training -- {os.path.basename(path)}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"Wrote {out_path}")
    print("\n".join(lines))
    for note in log["notes"][:10]:
        print(note)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--from", dest="lo", type=int, default=0)
    ap.add_argument("--to", dest="hi", type=int, default=None)
    args = ap.parse_args()

    out = args.out or os.path.splitext(args.log)[0] + "_stats.png"
    plot(args.log, out, lo=args.lo,
         hi=args.hi if args.hi is not None else float("inf"))


if __name__ == "__main__":
    main()
