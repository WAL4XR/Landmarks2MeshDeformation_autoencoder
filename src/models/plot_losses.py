"""Plot the training loss curves from a comb-model training log.

train_model.py logs one line per `log_interval` cycles, e.g.

    2026-08-13 13:13:28,088 INFO cycle 447900/500000 :: deformation: L1=1.0021 CCL=0.0000 | heatmap: L2=0.1575 CCL=0.0000

Newer runs also log a per-domain translation loss and the global gradient norm:

    ... :: facemesh: L1=0.0097 CCL=0.000261 TRANS=0.0041 | deformation: L1=0.0020 CCL=0.000037 TRANS=0.0149 | |g|=1.275

Both fields are optional -- their panels are only drawn when the log has them,
so older logs still plot as before.

The domain order within a line is randomized by the N-batch cycle, so lines are
parsed by domain name rather than by position.

Usage:
    python src/models/plot_losses.py logs/comb-10620.err
    python src/models/plot_losses.py logs/comb-10620.err -o loss.png --linear

Note the loss lines go to stderr (logging's default stream), so the input is
usually the .err file, not .out.
"""

import argparse
import math
import os
import re
import sys
from collections import defaultdict

import matplotlib

# Select a non-interactive backend before pyplot is imported: compute nodes are
# headless, so the default Qt backend would fail to initialize.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# "cycle 447900/500000 :: <rest>"
CYCLE_RE = re.compile(r"cycle\s+(\d+)/(\d+)\s*::\s*(.+)")
# "deformation: L1=1.0021 CCL=0.0000" (TRANS only present in newer runs)
ENTRY_RE = re.compile(
    r"(\w+):\s*(\w+)=([-\d.eE+]+)\s+CCL=([-\d.eE+]+)(?:\s+TRANS=([-\d.eE+]+))?"
)
# "|g|=1.275" -- the global gradient norm, once per line. Nothing clips or
# rescales it, so `max_grad_norm` in the config line is not a ceiling on it.
GRAD_RE = re.compile(r"\|g\|=([-\d.eE+]+)")


def parse_log(path):
    """Return (steps, pix, ccl, trans, crit, grad) keyed by domain.

    steps[d] -- cycle numbers, pix[d]/ccl[d]/trans[d] -- loss values,
    crit[d]  -- the pixel criterion name ("L1"/"L2") for axis labels,
    grad     -- (steps, values) for the global gradient norm, empty if unlogged.

    trans[d] is all-NaN for logs written before the translation loss existed.
    """
    steps = defaultdict(list)
    pix = defaultdict(list)
    ccl = defaultdict(list)
    trans = defaultdict(list)
    crit = {}
    gsteps, gvals = [], []

    with open(path, "r", errors="replace") as f:
        for line in f:
            m = CYCLE_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            for domain, name, pval, cval, tval in ENTRY_RE.findall(m.group(3)):
                try:
                    p, c = float(pval), float(cval)
                except ValueError:
                    continue
                steps[domain].append(step)
                pix[domain].append(p)
                ccl[domain].append(c)
                crit[domain] = name
                # NaN for logs without TRANS, so trans[d] stays index-aligned
                # with steps[d]; matplotlib skips NaN and the panel is dropped
                # below when a domain has no finite value at all.
                trans[domain].append(float(tval) if tval else float("nan"))
            g = GRAD_RE.search(m.group(3))
            if g:
                gsteps.append(step)
                gvals.append(float(g.group(1)))
    return steps, pix, ccl, trans, crit, (gsteps, gvals)


def _clip_range(steps, pix, ccl, trans, grad, lo, hi):
    """Keep only the points whose cycle falls in [lo, hi]."""
    for d in list(steps):
        keep = [i for i, s in enumerate(steps[d]) if lo <= s <= hi]
        for vals in (pix[d], ccl[d], trans[d]):
            vals[:] = [vals[i] for i in keep]
        steps[d] = [steps[d][i] for i in keep]
        if not steps[d]:
            del steps[d], pix[d], ccl[d], trans[d]

    gsteps, gvals = grad
    keep = [i for i, s in enumerate(gsteps) if lo <= s <= hi]
    grad = ([gsteps[i] for i in keep], [gvals[i] for i in keep])
    return steps, pix, ccl, trans, grad


def _summarize(name, s, v, fmt=".4f"):
    """One line of first/best/last -- enough to spot a divergence without the png."""
    s, v = zip(*[(a, b) for a, b in zip(s, v) if math.isfinite(b)])
    best_i = min(range(len(v)), key=lambda i: v[i])
    print(f"{name:18s}: first={v[0]:{fmt}} @{s[0]}  "
          f"best={v[best_i]:{fmt}} @{s[best_i]}  last={v[-1]:{fmt}} @{s[-1]}")


def plot(path, out_path, log_scale=True, collapse=None, lo=0, hi=float("inf")):
    steps, pix, ccl, trans, crit, grad = parse_log(path)
    if not steps:
        sys.exit(f"No 'cycle N/M ::' lines found in {path}. "
                 "The loss lines go to the .err file, not .out.")

    if lo > 0 or hi < float("inf"):
        steps, pix, ccl, trans, grad = _clip_range(
            steps, pix, ccl, trans, grad, lo, hi)
        if not steps:
            sys.exit(f"No logged cycles in range [{lo}, {hi}].")

    domains = sorted(steps)
    gsteps, gvals = grad

    # Only draw the newer panels when the log actually carries those fields.
    panels = [(pix, "Reconstruction loss (L_pix)"),
              (ccl, "Latent cycle consistency (L_CCL)")]
    if any(math.isfinite(v) for d in domains for v in trans[d]):
        panels.append((trans, "Translation loss (L_trans)"))
    npanels = len(panels) + bool(gvals)

    ncols = min(npanels, 2)
    nrows = -(-npanels // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5 * nrows),
                             squeeze=False)
    flat = axes.ravel()

    for ax, (series, title) in zip(flat, panels):
        for d in domains:
            # The criterion name only varies on the pixel panel.
            label = f"{d} ({crit[d]})" if series is pix else d
            ax.plot(steps[d], series[d], lw=1.2, label=label)
        if collapse is not None and series is pix:
            ax.axhline(collapse, color="crimson", ls="--", lw=1.2,
                       label=f"all-zeros floor ({collapse:g})")
        ax.set_title(title)
        ax.set_ylabel("loss")
        if log_scale:
            # Log scale keeps an early collapse visible once later values
            # differ by orders of magnitude; non-positive values are dropped.
            ax.set_yscale("log")

    if gvals:
        # Linear, and never log: the norm sits in a narrow band, and what makes
        # a blow-up visible is a spike out of that band, not its decades.
        ax = flat[len(panels)]
        ax.plot(gsteps, gvals, lw=1.2, color="tab:green", label="|g|")
        ax.set_title("Gradient norm (|g|, unclipped)")
        ax.set_ylabel("norm")

    for ax in flat[:npanels]:
        ax.set_xlabel("cycle")
        ax.grid(alpha=.3, lw=.5)
        ax.legend(fontsize=9)
    for ax in flat[npanels:]:  # odd panel count leaves a blank cell
        ax.set_visible(False)

    fig.suptitle(os.path.basename(path))
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"Wrote {out_path}")

    for d in domains:
        _summarize(f"{d} {crit[d]}", steps[d], pix[d])
    for d in domains:
        if any(math.isfinite(v) for v in trans[d]):
            _summarize(f"{d} TRANS", steps[d], trans[d])
    if gvals:
        _summarize("|g|", gsteps, gvals, fmt=".3f")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="training log (usually logs/comb-<jobid>.err)")
    ap.add_argument("-o", "--out", default=None, help="output png (default: <log>.png)")
    ap.add_argument("--linear", action="store_true", help="linear y axis instead of log")
    ap.add_argument("--collapse", type=float, default=None,
                    help="draw a reference line at the all-zeros loss value")
    ap.add_argument("--from", dest="lo", type=int, default=0,
                    help="first cycle to plot")
    ap.add_argument("--to", dest="hi", type=int, default=None,
                    help="last cycle to plot (e.g. --to 30000 to zoom on the start)")
    args = ap.parse_args()

    out = args.out or os.path.splitext(args.log)[0] + "_loss.png"
    plot(args.log, out, log_scale=not args.linear, collapse=args.collapse,
         lo=args.lo, hi=args.hi if args.hi is not None else float("inf"))


if __name__ == "__main__":
    main()
