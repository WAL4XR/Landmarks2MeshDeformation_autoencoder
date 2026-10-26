"""Evaluate comb-model checkpoints and render what they actually output.

Two things the training log cannot tell you:

  * Which checkpoint is best. The `cycle N :: ... L1=...` line is one 16-image
    batch, and on this data the batch-to-batch spread (0.0010 - 0.0018 for the
    facemesh domain) is wider than any difference between late checkpoints.
    `--eval` re-scores each checkpoint on the same fixed, evenly-spaced sample
    of the dataset, which makes the comparison meaningful.
  * What the maps look like. `--preview` renders, per domain, the input, the
    autoencoder reconstruction, the error, and the cross-domain translation
    (the encoder is shared, so any map can be decoded by the other domain's
    decoder -- that translation is the point of the whole model).

`--export` writes those facemesh -> deformation translations out as .exr in
the same layout the training data uses, so they can be loaded in Unity.

Maps are rendered the way MediaPipeFacemesh.render_delta_map does it: mid-gray
is zero, darker is a negative offset, brighter positive, and the three offset
channels are shown as RGB. Everything is drawn in the *normalized* units the
network sees ([-1, 1] after the per-domain scale), with off-mask pixels black.

Usage:
    # rank checkpoints on 256 images per domain
    python src/models/infer_comb.py --eval models/checkpoints/checkpoints/comb_step{250000,300000,400000}.pt

    # render previews from the winner
    python src/models/infer_comb.py --preview models/checkpoints/checkpoints/comb_step250000.pt

    # write every translated deformation map to data/translated/DeformationMap
    python src/models/infer_comb.py --export models/checkpoints/checkpoints/comb_step250000.pt
"""

import argparse
import os
import sys

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from architectures import CombTranslationModel  # noqa: E402
from train_model import (DomainImageDataset, _load_offset_map,  # noqa: E402
                         reconstruction_loss)

# Float32 + RLE is what the training EXRs use, so Unity sees one format only.
EXR_PARAMS = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT,
              cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_RLE]


def write_exr(path, offsets, alpha):
    """Write one (3, H, W) map + (H, W, 1) mask as the deformation domain's .exr.

    The layout is the one _load_offset_map() reads back and the Unity side
    already consumes: (H, W, 4) float32 BGRA, the three offsets stored BGR
    (i.e. dx, dy, dz reversed) with the coverage mask in alpha. The [2, 1, 0, 3]
    reorder is its own inverse, so this is exactly the read step undone.

    `offsets` must already be in *source* units, not the normalized units the
    network works in -- see translate_map().
    """
    rgba = np.concatenate([np.asarray(offsets).transpose(1, 2, 0), alpha], axis=2)
    if not cv2.imwrite(path, rgba[:, :, [2, 1, 0, 3]], EXR_PARAMS):
        raise IOError(f"Could not write EXR: {path}")
    return path


def target_mask(cfg, scales, dst, device):
    """The coverage mask a translation into `dst` should be written with.

    A translation lands in the target domain's canvas, so it needs that
    domain's mask, and the decoder has no branch that predicts one. Every map
    in the deformation domain carries a byte-identical mask, so sample 0 is
    not a stand-in there -- it is the mask.
    """
    ds = DomainImageDataset(cfg["domains"][dst]["path"], kind=dst, scale=scales[dst])
    return ds[0][1].unsqueeze(0).to(device)


def load_checkpoint(path, device):
    """Rebuild the model exactly as it was trained, from the checkpoint itself.

    The checkpoint carries the config and the per-domain normalization scales,
    so nothing here has to agree with the current configs/train_config.yaml --
    which matters when comparing checkpoints from different runs.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    domain_cfg = cfg["domains"]

    model = CombTranslationModel(
        domains={n: {"channels": dc["channels"], "out_activation": dc["out_activation"]}
                 for n, dc in domain_cfg.items()},
        latent_dim=cfg["latent_dim"],
        input_size=cfg["input_size"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # "scales" is the current per-domain form; older checkpoints stored a
    # single "deform_scale" and left the other domain unnormalized.
    scales = ckpt.get("scales")
    if scales is None:
        scales = {n: (ckpt.get("deform_scale") or 1.0) if n == "deformation" else 1.0
                  for n in domain_cfg}
    return model, cfg, scales, ckpt.get("step")


def sample_indices(count, wanted):
    """`wanted` evenly-spaced indices, so every checkpoint sees one same set."""
    if count <= wanted:
        return list(range(count))
    step = count / wanted
    return [int(i * step) for i in range(wanted)]


@torch.no_grad()
def evaluate(model, cfg, scales, device, n_images=256, batch=16):
    """Masked reconstruction loss per domain over a fixed sample."""
    out = {}
    for name, dc in cfg["domains"].items():
        ds = DomainImageDataset(dc["path"], kind=name, scale=scales[name])
        idx = sample_indices(len(ds), n_images)
        total, seen = 0.0, 0
        for i in range(0, len(idx), batch):
            chunk = idx[i:i + batch]
            xs, ms = zip(*(ds[j] for j in chunk))
            x, m = torch.stack(xs).to(device), torch.stack(ms).to(device)
            recon = model(x, name)
            # Weight by batch size so a short final chunk doesn't count double.
            total += float(reconstruction_loss(recon, x, m, dc["pixel_loss"])) * len(chunk)
            seen += len(chunk)
        out[name] = total / seen
    return out


def render_map(arr, mask=None, value_range=1.0):
    """Signed (3, H, W) map -> (H, W, 3) uint8 RGB, mid-gray = zero.

    Same convention as MediaPipeFacemesh.render_delta_map: darker than gray is
    a negative offset, brighter is positive, off-mask pixels are black so the
    holes read as holes rather than as skin that did not move.
    """
    t = np.clip(np.asarray(arr) / value_range, -1.0, 1.0)
    rgb = np.round((t * 0.5 + 0.5) * 255).astype(np.uint8).transpose(1, 2, 0)
    if mask is not None:
        rgb[np.asarray(mask)[0] <= 0] = 0
    return rgb


@torch.no_grad()
def preview(model, cfg, scales, device, out_dir, step, n_samples=4, stems=None):
    """Render input / reconstruction / error / translation for a few samples.

    `stems` names the frames to render (e.g. ["00023"]), which is how you look
    at one specific frame; the domains are paired by stem, so the same names
    select the corresponding map in every domain. Without it the rows are
    `n_samples` evenly-spaced frames instead.
    """
    os.makedirs(out_dir, exist_ok=True)
    names = list(cfg["domains"])
    written = []

    for name, dc in cfg["domains"].items():
        other = next(d for d in names if d != name)
        ds = DomainImageDataset(dc["path"], kind=name, scale=scales[name])
        if stems:
            at = {os.path.splitext(os.path.basename(f))[0]: i
                  for i, f in enumerate(ds.files)}
            missing = [t for t in stems if t not in at]
            if missing:
                raise SystemExit(f"{name}: no such frame(s) under {dc['path']}: "
                                 + ", ".join(missing))
            idx = [at[t] for t in stems]
        else:
            idx = sample_indices(len(ds), n_samples)
        rows = len(idx)

        xs, ms = zip(*(ds[j] for j in idx))
        x, m = torch.stack(xs).to(device), torch.stack(ms).to(device)
        recon = model(x, name)

        # A translation lands in the *other* domain's canvas, so it has to be
        # masked with that domain's coverage, not the source's. The facemesh
        # mask is a constant (one raster of the neutral mesh) and the
        # deformation masks barely vary, so one representative sample is a
        # fair stand-in for the mask the decoder has no branch to predict.
        other_ds = DomainImageDataset(cfg["domains"][other]["path"], kind=other,
                                      scale=scales[other]) #refrence the whole dataset, represents all N files of the other domain
        m_other = other_ds[0][1].unsqueeze(0).to(device)
        translated = model.translate(x, other) * m_other

        err = (recon - x).abs() * m
        # Error is non-negative, so the signed mid-gray rendering wastes half
        # its range on values that cannot occur: show it as heat instead, on a
        # scale set by the errors actually present.
        err_mean = err.mean(dim=1).cpu().numpy()
        vmax = max(float(np.percentile(err_mean, 99.9)), 1e-6)

        fig, axes = plt.subplots(rows, 4, figsize=(3.3 * 4, 3.3 * rows))
        axes = np.atleast_2d(axes)
        for r in range(rows):
            panels = [
                ("Input", render_map(x[r].cpu(), m[r].cpu())),
                ("Reconstruction", render_map(recon[r].cpu(), m[r].cpu())),
                (None, None),  # error, drawn below with its own colormap
                (other, render_map(translated[r].cpu(), m_other[0].cpu())),
            ]
            for c, (title, img) in enumerate(panels):
                ax = axes[r][c]
                if img is None:
                    im = ax.imshow(err_mean[r], cmap="inferno", vmin=0, vmax=vmax)
                    title = "|error|"
                    # The bar carries the scale, so the range stays out of the
                    # heading; horizontal keeps it clear of the panel to its
                    # right rather than wedged between the two columns.
                    if r == rows - 1:
                        cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                                            fraction=0.046, pad=0.04)
                        # Label the end of the bar with vmax itself, so the
                        # ceiling of the scale is readable off the figure.
                        # Auto ticks crowding it are dropped rather than
                        # overprinted.
                        ticks = [t for t in cbar.get_ticks() if 0 <= t <= vmax * 0.88]
                        cbar.set_ticks(ticks + [vmax])
                        cbar.set_ticklabels([f"{t:.4f}" for t in ticks + [vmax]])
                        cbar.ax.tick_params(labelsize=14)
                        for t in cbar.ax.get_xticklabels():
                            t.set_fontweight("bold")
                else:
                    ax.imshow(img)
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(title, fontsize=20, fontweight="bold")

        fig.tight_layout()
        tag = f"_{'_'.join(stems)}" if stems else ""
        path = os.path.join(out_dir, f"infer_step{step}_{name}{tag}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)
        print(f"Wrote {path}")
    return written

#python src/models/infer_comb.py --translate data/Train/FaceMeshes/00002.npy     models/checkpoints/comb_step500000.pt
#python src/models/infer_comb.py --translate logs/inference/disfa_SN001/SN001_B8_AU25_TrailNo_2_042.npy     models/Mike2Omda_checkpoints/comb_step500000.pt
@torch.no_grad()
def translate_map(model, cfg, scales, device, src_path,
                  out_dir="data/translated/deformation",
                  src="facemesh", dst="deformation"):
    """Run one facemesh map through the model and write the deformation .exr.

    `src_path` is a single file of the source domain -- a (H, W, 4) .npy for
    facemesh -- and the result is written to out_dir under the same stem, so
    data/.../delta_maps/00042.npy becomes data/translated/deformation/00042.exr.

    The input is normalized exactly as DomainImageDataset does it (divide by
    the domain's scale, clamp, mask), because that is the only form the
    encoder has ever seen. On the way out the reverse happens: the decoder
    speaks the *target* domain's normalized units, so the output is multiplied
    by scales[dst] to land back in the units Unity expects.

    Returns the path written.
    """
    os.makedirs(out_dir, exist_ok=True)

    offsets, mask = _load_offset_map(src_path, src)
    x = torch.from_numpy(offsets).permute(2, 0, 1).contiguous()
    m = torch.from_numpy(mask).permute(2, 0, 1).contiguous()
    x = ((x / scales[src]).clamp(-1.0, 1.0) * m).unsqueeze(0).to(device)

    m_dst = target_mask(cfg, scales, dst, device)
    y = model.translate(x, dst) * m_dst * scales[dst]

    stem = os.path.splitext(os.path.basename(src_path))[0]
    path = write_exr(os.path.join(out_dir, f"{stem}.exr"),
                     y[0].cpu().numpy(),
                     m_dst[0].permute(1, 2, 0).cpu().numpy())
    print(f"Wrote {path}")
    return path


# python src/models/infer_comb.py --export \
#   --source-dir data/Test/FaceMeshes \
#   --export-dir data/Test/PredictedDeformationMaps \
#   models/checkpoints/comb_step500000.pt

@torch.no_grad()
def export_translations(model, cfg, scales, device, out_dir, src_dir=None,
                        src="facemesh", dst="deformation", limit=0, batch=16):
    """Write the src -> dst translation of every src map as an .exr.

    This is translate_map() over the whole source domain, batched -- see
    write_exr() for the file layout and translate_map() for the two
    conversions (units and coverage) that every translation needs.

    `src_dir` overrides where the source maps are read from; it defaults to
    the source domain's training root recorded in the checkpoint. Point it at
    a held-out split (data/Test/FaceMeshes) to translate maps the model never
    saw. The target mask still comes from the checkpoint's deformation root --
    every map in that domain carries a byte-identical mask, so the split it is
    read from does not matter.
    """
    os.makedirs(out_dir, exist_ok=True)
    ds = DomainImageDataset(src_dir or cfg["domains"][src]["path"],
                            kind=src, scale=scales[src])

    m_dst = target_mask(cfg, scales, dst, device)         # (1, 1, H, W)
    alpha = m_dst[0].permute(1, 2, 0).cpu().numpy()       # (H, W, 1)

    idx = list(range(len(ds) if limit <= 0 else min(limit, len(ds))))
    for i in range(0, len(idx), batch):
        chunk = idx[i:i + batch]
        x = torch.stack([ds[j][0] for j in chunk]).to(device)
        y = model.translate(x, dst) * m_dst * scales[dst]

        for j, out in zip(chunk, y.cpu().numpy()):
            stem = os.path.splitext(os.path.basename(ds.files[j]))[0]
            write_exr(os.path.join(out_dir, f"{stem}.exr"), out, alpha)

    print(f"Wrote {len(idx)} {dst} maps to {out_dir} "
          f"(x{scales[dst]:.4g}, i.e. source units)")
    return len(idx)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--eval", action="store_true", help="score each checkpoint")
    ap.add_argument("--preview", action="store_true", help="render sample outputs")
    ap.add_argument("--export", action="store_true",
                    help="write translated deformation maps as .exr for Unity")
    ap.add_argument("--translate", metavar="MAP",
                    help="translate one facemesh .npy and write its .exr")
    ap.add_argument("--images", type=int, default=256, help="images per domain for --eval")
    ap.add_argument("--samples", type=int, default=4, help="rows per --preview figure")
    ap.add_argument("--frame", nargs="+", metavar="STEM",
                    help="--preview these frames by stem (e.g. 00023) instead "
                         "of an evenly-spaced sample")
    ap.add_argument("--data-root", metavar="DIR",
                    help="read the domains from DIR instead of the root the "
                         "checkpoint was trained on (e.g. data/Test)")
    ap.add_argument("--export-limit", type=int, default=0,
                    help="stop --export after N maps (0 = the whole domain)")
    ap.add_argument("-o", "--out", default="logs/inference")
    ap.add_argument("--export-dir", default="data/translated/DeformationMap")
    ap.add_argument("--source-dir",
                    help="read the facemesh maps for --export from here "
                         "instead of the checkpoint's training root")
    ap.add_argument("--translate-dir", default="data/translated/deformation")
    args = ap.parse_args()
    if not (args.eval or args.preview or args.export or args.translate):
        args.eval = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    scored = []
    for path in args.checkpoints:
        model, cfg, scales, step = load_checkpoint(path, device)
        if args.data_root:
            # Same domain directory names, different split -- the checkpoint
            # stores the training root, which is not where held-out frames are.
            for dc in cfg["domains"].values():
                dc["path"] = os.path.join(args.data_root,
                                          os.path.basename(dc["path"].rstrip("/")))
                print(f"reading {dc['path']}")
        if args.eval:
            losses = evaluate(model, cfg, scales, device, n_images=args.images)
            scored.append((step, path, losses))
            msg = "  ".join(f"{d}: {v:.5f}" for d, v in sorted(losses.items()))
            print(f"step {step:>7}  {msg}   ({os.path.basename(path)})")
        if args.preview:
            preview(model, cfg, scales, device, args.out, step,
                    n_samples=args.samples, stems=args.frame)
        if args.translate:
            translate_map(model, cfg, scales, device, args.translate,
                          out_dir=args.translate_dir)
        if args.export:
            # One name per input map, so several checkpoints would overwrite
            # each other -- give each its own subdirectory instead.
            out_dir = (args.export_dir if len(args.checkpoints) == 1
                       else os.path.join(args.export_dir, f"step{step}"))
            export_translations(model, cfg, scales, device, out_dir,
                                src_dir=args.source_dir, limit=args.export_limit)
        del model
        torch.cuda.empty_cache()

    if len(scored) > 1:
        # Rank by each domain's loss relative to the best seen, averaged, so
        # neither domain dominates just by having a larger absolute loss.
        best = {d: min(s[2][d] for s in scored) for d in scored[0][2]}
        ranked = sorted(scored, key=lambda s: sum(s[2][d] / best[d] for d in best))
        print("\nranked (1.00 = best on every domain):")
        for step, path, losses in ranked:
            score = sum(losses[d] / best[d] for d in best) / len(best)
            print(f"  step {step:>7}  score {score:.4f}  "
                  + "  ".join(f"{d}={losses[d]:.5f}" for d in sorted(losses)))
        print(f"\nbest: step {ranked[0][0]}  ({ranked[0][1]})")


if __name__ == "__main__":
    main()
