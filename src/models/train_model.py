"""Comb-model image-to-image training (Moser et al. 2021).

Concurrently trains one autoencoder per domain (facemesh, deformation) that
share a single encoder and the shared decoder stem (up to f_out). Both domains
are dense maps of signed offsets -- the facemesh domain holds the per-pixel
expression deltas rasterized from the MediaPipe mesh, the deformation domain
the rig's deformation maps -- so both are normalized to ~[-1, 1], reconstructed
through a tanh and scored with L1. Training follows the paper's strategy:

  * Single-domain batches: each batch holds images of exactly one domain.
  * N-batch cycle: for N domains, every N consecutive batches cover each
    domain once, in random order.
  * Gradient accumulation: gradients are accumulated across the N batches;
    at the end of the cycle the *shared* components' gradients are averaged
    (divided by N) and a single optimizer step is applied. Domain-specific
    decoders only receive gradient from their own batch.
  * Loss (Eq. 1, adapted): L = L_pix + w_ccl * L_CCL + w_trans * L_trans.
      - L_pix: per-domain reconstruction (Eq. 2), masked by the background
        mask. Both domains ship that mask with the data -- the 4th channel of
        every map -- so, exactly as in Eq. 2, the input and the reconstruction
        are both multiplied by it and only face pixels are scored. The paper's
        SSIM term is omitted: it targets perceptual RGB face images and is
        ill-suited to our sparse, signed offset maps. The pixel criterion is
        L1 for both domains.
      - L_CCL: latent cycle-consistency (Eq. 3); decoder weights are excluded
        from its gradient (handled by CombTranslationModel.cycle_latents).
      - L_trans: *supervised* translation. Our dataset is paired -- a facemesh
        map and the deformation map it was captured with share a file stem --
        so the cross-domain decode of an input has a known ground truth: the
        counterpart map of the same stem. The translated output is scored
        against it directly, on the target domain's mask. This is what makes
        the run supervised rather than unpaired: without it nothing ties a
        latent to the *corresponding* map in the other domain, only to some
        plausible one. Set `translation_weight: 0` to fall back to the paper's
        unsupervised objective.
"""

import os

# OpenEXR support in OpenCV must be enabled before cv2 is imported.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import glob
import logging
import math
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import optim
from torch.utils.data import DataLoader, Dataset

# Import the architecture whether run as a script or imported as a module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from architectures import CombTranslationModel  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Masked reconstruction loss (Eq. 2; SSIM omitted).
# ---------------------------------------------------------------------------

def reconstruction_loss(recon, target, mask, pixel_loss: str):
    """Masked pixel reconstruction, Eq. 2: f_c(recon (*) m, target (*) m).

    pixel_loss selects the criterion f_c per domain. Both domains are signed
    offset maps and use "l1"; "l2"/"mse" is kept for domains added later.

    `mask` is the background mask of Eq. 2, a (B, 1, H, W) tensor that is 1.0
    on face pixels and 0.0 elsewhere; it is broadcast over the channels the
    way the paper multiplies it into every image channel. Only the covered
    pixels are scored, which is what focuses the network on the face: for the
    facemesh domain the mask leaves 38% of the canvas, so an unmasked average
    would spend nearly two thirds of the loss on background the decoder can
    match by emitting zeros.

    The sum is divided by the number of covered pixels rather than by the
    canvas area, so the value reads as "mean error per face pixel" and does
    not shift when a domain's coverage does.
    """
    if pixel_loss == "l1":
        err = (recon - target).abs()
    elif pixel_loss in ("l2", "mse"):
        err = (recon - target) ** 2
    else:
        raise ValueError(f"Unknown pixel_loss: {pixel_loss!r} (expected 'l1' or 'l2')")

    covered = mask.sum().clamp(min=1.0) * recon.shape[1]  # face pixels * channels
    return (err * mask).sum() / covered

def translation_loss(translated, ground_truth, mask, pixel_loss: str):
    """Masked pixel translation loss, scored the same way as Eq. 2.

    `translated` is the cross-domain decode of a source map and `ground_truth`
    the target-domain map of the *same identity* -- the paired file that shares
    the source's stem. Both `mask` and `pixel_loss` are the target domain's,
    not the source's: the comparison happens entirely in the target domain, so
    it is that domain's coverage that says which pixels exist and its criterion
    f_c that scores them.
    """
    if pixel_loss == "l1":
        err = (translated - ground_truth).abs()
    elif pixel_loss in ("l2", "mse"):
        err = (translated - ground_truth) ** 2
    else:
        raise ValueError(f"Unknown pixel_loss: {pixel_loss!r} (expected 'l1' or 'l2')")

    covered = mask.sum().clamp(min=1.0) * translated.shape[1]  # face pixels * channels
    return (err * mask).sum() / covered

# ---------------------------------------------------------------------------
# Per-domain datasets. Both domains are dense maps of *signed offsets*: the
# facemesh domain holds the per-pixel expression deltas rasterized by
# src/MediaPipeFacemesh.py, the deformation domain the rig's deformation maps.
#
# The two sets correspond: `00042.npy` and `00042.exr` are the same frame seen
# in the two domains. DomainImageDataset ignores that and loads one domain in
# isolation (what the paper's unpaired objective needs, and what inference
# uses); PairedDomainDataset indexes by the shared stem and hands back every
# domain's map for one frame at once, which is what the supervised translation
# loss needs.
# ---------------------------------------------------------------------------

# Glob pattern per domain kind.
DOMAIN_PATTERN = {"facemesh": "*.npy", "deformation": "*.exr"}


def _load_offset_map(path: str, kind: str):
    """Read one map off disk as ((H, W, 3) offsets, (H, W, 1) mask) float32.

    Both kinds store the same 4 channels: three signed offsets plus a coverage
    mask that is 1.0 where there is face and 0.0 on the background (and, for
    the facemesh domain, in the eye apertures and the mouth opening, which the
    mesh genuinely does not cover). That 4th channel is the background mask
    m_x of the paper's Eq. 2 -- it is returned separately rather than fed to
    the 3-channel encoder as if it were an offset.

    facemesh:     .npy of shape (H, W, 4), channels (dx, dy, dz, mask), written
                  by MediaPipeFacemesh.rasterize_deltas(). The offsets are raw
                  and unscaled, in pose-normalized units (inter-eye distance
                  = 1.0), so they are small -- this dataset stays inside ~0.1.
                  The mask leaves ~38% of the canvas covered.
    deformation:  .exr (H, W, 4) BGRA float32; the first 3 channels are the
                  offsets and are reordered BGR->RGB(=XYZ), the 4th is the
                  mask (~96% covered -- the render fills most of the frame).
    """
    if kind == "facemesh":
        arr = np.load(path)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Could not read EXR: {path}")
        arr = img[:, :, [2, 1, 0, 3]]  # BGRA -> RGBA, i.e. (dx, dy, dz, mask)

    offsets = np.ascontiguousarray(arr[:, :, :3], dtype=np.float32)
    mask = np.ascontiguousarray(arr[:, :, 3:4], dtype=np.float32)
    return offsets, mask


class DomainImageDataset(Dataset):
    """Loads a domain as (masked (3, H, W) map in ~[-1, 1], (1, H, W) mask).

    The raw offsets of both domains are far too small to be reconstructed
    through a tanh (the facemesh deltas live inside ~0.1 of the tanh's [-1, 1]
    range, where the loss barely distinguishes them from zero), so each domain
    is divided by its own `scale` -- see compute_domain_scale().

    The map handed back is already masked, so what the encoder sees is the
    face on an exactly-zero background (Eq. 2 masks the input as well as the
    reconstruction). The mask rides along because the loss needs it too.
    """

    def __init__(self, root: str, kind: str, scale: float = 1.0): #called once when instantiating the Dataset object
        if kind not in DOMAIN_PATTERN:
            raise ValueError(f"Unknown domain kind: {kind}")
        self.kind = kind
        self.scale = scale
        self.files = sorted(glob.glob(os.path.join(root, DOMAIN_PATTERN[kind])))
        if not self.files:
            raise FileNotFoundError(f"No {kind} files found under {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx): #add any needed preprocesing here, e.g. normalization, resizing, etc. This method is called by the DataLoader to get a single sample from the dataset.
        offsets, mask = _load_offset_map(self.files[idx], self.kind)
        # PyTorch's convolutional layers strictly expect CHW (Channels, Height, Width)
        x = torch.from_numpy(offsets).permute(2, 0, 1).contiguous()
        m = torch.from_numpy(mask).permute(2, 0, 1).contiguous()
        return (x / self.scale).clamp(-1.0, 1.0) * m, m


class PairedDomainDataset(Dataset):
    """One frame in *every* domain at once: {domain: (map, mask)}.

    The pairing is by file stem -- `00042.npy` under the facemesh root is the
    same frame as `00042.exr` under the deformation root -- so the index is the
    sorted intersection of the stems present in all domains. Stems that are
    missing from any domain are dropped rather than silently mispaired: an
    index built per-domain by position would slide every later frame against
    its counterpart the moment one file is missing, and the translation loss
    would then be trained on mismatched pairs.

    Each domain's map is normalized and masked exactly as DomainImageDataset
    does it, so the two datasets hand the model identical tensors.
    """

    def __init__(self, roots: dict, scales: dict):
        self.roots = dict(roots)
        self.scales = {name: float(scales[name]) for name in roots}

        stems = {}
        for name, root in self.roots.items():
            if name not in DOMAIN_PATTERN:
                raise ValueError(f"Unknown domain kind: {name}")
            found = glob.glob(os.path.join(root, DOMAIN_PATTERN[name]))
            if not found:
                raise FileNotFoundError(f"No {name} files found under {root}")
            stems[name] = {os.path.splitext(os.path.basename(p))[0] for p in found}

        common = set.intersection(*stems.values())
        if not common:
            raise FileNotFoundError(
                "No stems shared by all domains "
                + ", ".join(f"{n} ({len(s)} files)" for n, s in stems.items())
                + ". The supervised translation loss pairs a facemesh map with "
                "the deformation map of the same name, so the domains must use "
                "matching file stems."
            )
        for name, s in stems.items():
            dropped = len(s) - len(common)
            if dropped:
                logger.warning(
                    f"Domain '{name}': {dropped} of {len(s)} files have no "
                    f"counterpart in every other domain and are unpaired; "
                    f"they are excluded from training."
                )
        self.stems = sorted(common)

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        sample = {}
        for name, root in self.roots.items():
            ext = os.path.splitext(DOMAIN_PATTERN[name])[1]
            offsets, mask = _load_offset_map(os.path.join(root, stem + ext), name)
            x = torch.from_numpy(offsets).permute(2, 0, 1).contiguous()
            m = torch.from_numpy(mask).permute(2, 0, 1).contiguous()
            sample[name] = ((x / self.scales[name]).clamp(-1.0, 1.0) * m, m)
        return sample


def compute_domain_scale(root: str, kind: str, percentile: float = 99.9) -> float:
    """Scale that maps a domain's signed offsets to ~[-1, 1].

    Uses a high percentile of |value| over the 3 offset channels so a few
    outliers don't crush the useful range; the values past it are clipped by
    the dataset. Store this (it goes into the checkpoint) to invert the
    normalization at inference -- a prediction is only meaningful in the
    original units after multiplying by the scale of its own domain.

    The domain's own mask selects which pixels count. Off-face pixels are
    exact zeros and cover most of a facemesh delta map, so including them
    would put the percentile deep in the background rather than on the face.
    """
    files = sorted(glob.glob(os.path.join(root, DOMAIN_PATTERN[kind])))
    if not files:
        raise FileNotFoundError(f"No {kind} files found under {root}")
    mags = []
    for path in files:
        offsets, mask = _load_offset_map(path, kind)
        mags.append(np.abs(offsets[mask[:, :, 0] > 0]).ravel())
    scale = float(np.percentile(np.concatenate(mags), percentile))
    return scale if scale > 0 else 1.0


def _infinite(loader):
    """Yield batches forever so single-domain loaders can be cycled."""
    while True:
        for batch in loader:
            yield batch


# ---------------------------------------------------------------------------
# Training.
# ---------------------------------------------------------------------------

def _default_cfg() -> dict:
    return {
        "input_size": 256,
        "latent_dim": 512,
        "domains": {
            # Per-pixel expression deltas from MediaPipeFacemesh.py
            "facemesh": {
                "path": "data/Train/FaceMeshes",
                "channels": 3,
                "out_activation": "tanh",
                "pixel_loss": "l1",
                # "scale": pin the normalization instead of measuring it.
            },
            "deformation": {
                "path": "data/Omda/Train/DeformationMaps",
                "channels": 3,
                "out_activation": "tanh",
                "pixel_loss": "l1",
            },
        },
        # Loss weights. L = L_pix + ccl_weight * L_CCL + translation_weight *
        # L_trans. translation_weight: 0 drops the supervision and trains the
        # paper's unpaired objective.
        "ccl_weight": 1.0,
        "translation_weight": 1.0,
        "iterations": 50000,    # number of N-batch cycles (shared updates)
        "max_grad_norm": 1.0,   # 0 disables gradient clipping
        "spike_abort": {},      # domain -> collapse loss threshold
        "spike_patience": 3,    # consecutive logged cycles above it before aborting
        "spike_warmup": 2000,   # cycles before the spike check is armed
        "batch_size": 16,
        "learning_rate": 5e-5,
        "beta1": 0.5,
        "beta2": 0.999,
        "num_workers": 2,
        "log_interval": 100,
        "save_interval": 1000,
        "checkpoint_dir": "models/Omda_checkpoints",
        "seed": 0,
    }


def train(config_path: str = "configs/train_config.yaml"):
    """Train the comb model per the paper's N-batch-cycle strategy."""
    cfg = _default_cfg()
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            loaded = yaml.safe_load(f) or {}
        cfg.update(loaded.get("comb", {}))  # comb-specific keys override defaults
    logger.info(f"Comb training config: {cfg}")

    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    domain_cfg = cfg["domains"]
    domain_names = list(domain_cfg)
    # Per-domain pixel criterion f_c (Eq. 2): L1 for both domains, whose
    # targets are signed offset maps reconstructed through a tanh output.
    pixel_loss = {n: dc["pixel_loss"] for n, dc in domain_cfg.items()}

    ccl_weight = float(cfg.get("ccl_weight", 1.0))
    translation_weight = float(cfg.get("translation_weight", 1.0))

    # Every domain gets its own normalization scale, measured from its data
    # unless the config pins one.
    scales = {}
    for name, dc in domain_cfg.items():
        scale = dc.get("scale")
        if scale is None:
            scale = compute_domain_scale(dc["path"], kind=name)
            logger.info(f"Domain '{name}' normalization scale (p99.9 |value|): {scale:.6f}")
        else:
            scale = float(scale)
            logger.info(f"Domain '{name}' normalization scale (pinned): {scale:.6f}")
        scales[name] = scale

    # One paired dataset, read through N independently shuffled loaders -- one
    # per slot of the N-batch cycle. Each batch still holds a single domain's
    # images as far as the encoder is concerned (the cycle's slot picks which
    # entry of the sample is the input); the other entries ride along as the
    # ground truth the translation loss needs. Keeping a loader per domain,
    # rather than reusing one batch for the whole cycle, preserves the paper's
    # independent per-domain sampling: the deformation batch of a cycle is not
    # the same set of frames as its facemesh batch.
    dataset = PairedDomainDataset({n: dc["path"] for n, dc in domain_cfg.items()}, scales)
    bs = min(cfg["batch_size"], len(dataset))
    loaders = {}
    for name in domain_names:
        loader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=True,
            drop_last=True,
            num_workers=cfg["num_workers"],
            # _infinite() restarts the loader every epoch; keep the workers
            # alive so they aren't respawned at each boundary.
            persistent_workers=cfg["num_workers"] > 0,
        )
        loaders[name] = _infinite(loader)
    logger.info(f"Paired dataset: {len(dataset)} frames per domain, batch size {bs}")

    # Model.
    model = CombTranslationModel(
        domains={n: {"channels": dc["channels"], "out_activation": dc["out_activation"]}
                 for n, dc in domain_cfg.items()},
        latent_dim=cfg["latent_dim"],
        input_size=cfg["input_size"],
    ).to(device)

    # Shared components whose gradients get averaged over the N-batch cycle.
    shared_params = list(model.encoder.parameters()) + list(model.decoder_stem.parameters())

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        betas=(cfg["beta1"], cfg["beta2"]),
    )

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    model.train() #sets the model to training mode, which is important for layers like dropout and batch normalization that behave differently during training and inference.
    n = len(domain_names)

    # Read for the clipping block below, which is currently disabled; the
    # gradient norm is still measured and logged every logged cycle.
    max_grad_norm = float(cfg.get("max_grad_norm", 0.0))  # 0 disables clipping
    # Spike abort: stop once a domain's reconstruction loss stays at or above
    # its collapse threshold for `spike_patience` consecutive logged cycles.
    # Set a threshold per domain in `spike_abort`; domains left out are not
    # watched. See the config for how to compute the value.
    spike_abort = cfg.get("spike_abort") or {}
    spike_patience = int(cfg.get("spike_patience", 3))
    # An untrained model starts *above* the collapse thresholds (its outputs
    # are near zero-mean noise, costing more than a saturated output), so the
    # check has to stay disarmed until training has descended past them.
    spike_warmup = int(cfg.get("spike_warmup", 2000))
    spike_hits = {d: 0 for d in spike_abort}

    for step in range(1, cfg["iterations"] + 1):
        optimizer.zero_grad(set_to_none=True) # clears gradients to avoid accumlation from multiple passes.

        # Every N consecutive batches cover each domain once, in random order.
        order = random.sample(domain_names, n)
        cycle_logs = {}
        for domain in order:
            # One paired frame batch. `domain` is this slot's input domain; the
            # sample's other entries are the same frames in the other domains,
            # i.e. the ground truth for translating out of `domain`.
            sample = next(loaders[domain])
            # x arrives already masked; m is the background mask of Eq. 2.
            x, m = (t.to(device) for t in sample[domain])

            # Reconstruction (Eq. 2): encode -> stem -> own decoder, scored on
            # the face pixels only. The latent is kept so the translation and
            # the cycle term can reuse it instead of re-encoding x twice more.
            z = model.encode(x)
            recon = model.decode(z, domain)
            pix = reconstruction_loss(recon, x, m, pixel_loss=pixel_loss[domain])

            if step % cfg["log_interval"] == 0:
                # Stats over the face pixels: outside the mask the decoder is
                # unconstrained, so whole-canvas min/max says nothing about
                # whether the reconstruction is healthy.
                face = recon.masked_select(m.expand_as(recon) > 0)
                zeros = int((face == 0).sum())
                logger.info(f"Domain '{domain}': max {face.max():.4f}, min {face.min():.4f}, mean {face.mean():.4f}, zeros: {zeros} / {face.numel()} ({100 * zeros / face.numel():.2f}%)")

            # For N domains, pick a random other domain as the opposite
            # decoder j; both cross-domain terms below use it.
            other = random.choice([d for d in domain_names if d != domain])

            # Supervised translation. Decoding z with the *other* domain's tail
            # should reproduce that domain's map of this very frame, which the
            # paired sample carries. Scored on the target domain's mask and
            # criterion, since the comparison lives in the target domain.
            trans = torch.zeros((), device=device)
            if translation_weight != 0.0:
                y, m_other = (t.to(device) for t in sample[other])
                translated = model.decode(z, other)
                trans = translation_loss(
                    translated, y, m_other, pixel_loss=pixel_loss[other]
                )

            # Cross-domain latent cycle consistency (Eq. 3). The translated map
            # is masked before re-encoding so the encoder sees the same zero
            # background on both sides of the loop. This decodes into `other`
            # a second time on purpose: Eq. 3 excludes the decoder weights from
            # its gradient, so it needs its own graph, recorded with the
            # decoder frozen, and cannot share the translation's decode above.
            _, z_star = model.cycle_latents(x, other, mask=m, z=z)
            ccl = F.mse_loss(z_star, z)

            loss = pix + ccl_weight * ccl + translation_weight * trans
            loss.backward()  # accumulate into .grad across the N batches
            cycle_logs[domain] = (pix.item(), ccl.item(), trans.detach().item()) #This only works for tensors with one element, which is the case for the loss values here.

        # Average the shared components' accumulated gradients over the cycle.
        for p in shared_params:
            if p.grad is not None:
                p.grad.div_(n)

        # # Clip after the shared-gradient averaging and before the step, so the
        # # cap applies to the update that is actually taken. A single outsized
        # # gradient here is what destroyed run 11294: at cycle ~16000 the shared
        # # encoder blew up, both output activations saturated (tanh -> +/-1)
        # # and training never recovered.
        # grad_norm = float("nan")
        # if max_grad_norm > 0:
        #     grad_norm = float(
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        #     ) #returns the norm before clipping
        #     # A non-finite gradient would poison every weight through the step.
        #     if not math.isfinite(grad_norm):
        #         logger.error(
        #             f"cycle {step}: non-finite gradient norm ({grad_norm}); "
        #             "skipping this update"
        #         )
        #         optimizer.zero_grad(set_to_none=True)
        #         continue

        # Norm of the update actually being taken, measured after the shared-
        # gradient averaging and before the step. Nothing clips or rescales it
        # -- this is purely the number in the log line, and it is what makes a
        # blow-up like run 11294's visible: the norm spikes a cycle or two
        # before the output activations saturate. Only computed on logged
        # cycles, since nothing else uses it.
        grad_norm = float("nan")
        if step % cfg["log_interval"] == 0:
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if grads:
                grad_norm = float(
                    torch.linalg.vector_norm(
                        torch.stack([torch.linalg.vector_norm(g) for g in grads])
                    )
                )

        optimizer.step()

        if step % cfg["log_interval"] == 0:
            msg = " | ".join(
                f"{d}: {pixel_loss[d].upper()}={p:.4f} CCL={c:.6f} TRANS={t:.4f}"
                for d, (p, c, t) in cycle_logs.items()
            )
            msg += f" | |g|={grad_norm:.3f}"
            logger.info(f"cycle {step}/{cfg['iterations']} :: {msg}")
            

            # Spike abort. Once the output activations saturate the loss pins
            # to a constant and no further training is possible, so continuing
            # only wastes the reservation.
            for d, threshold in spike_abort.items():
                if d not in cycle_logs or step <= spike_warmup:
                    continue
                if cycle_logs[d][0] >= threshold:
                    spike_hits[d] += 1
                    logger.warning(
                        f"cycle {step}: '{d}' loss {cycle_logs[d][0]:.4f} >= collapse "
                        f"threshold {threshold} ({spike_hits[d]}/{spike_patience})"
                    )
                else:
                    spike_hits[d] = 0  # only consecutive hits count
                if spike_hits[d] >= spike_patience:
                    logger.error(
                        f"Aborting at cycle {step}: '{d}' has been collapsed for "
                        f"{spike_patience} consecutive logged cycles. The last "
                        f"checkpoint before this point is the one to inspect."
                    )
                    return model

        if step % cfg["save_interval"] == 0:
            ckpt = os.path.join(cfg["checkpoint_dir"], f"comb_step{step}.pt")
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    # Per-domain normalization; a prediction only means
                    # anything in the original units once multiplied by these.
                    "scales": scales,
                    "config": cfg,
                },
                ckpt,
            )
            logger.info(f"Saved checkpoint: {ckpt}")
            #logger.info(f"Saved parameters: {model.state_dict()}")

    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train()
