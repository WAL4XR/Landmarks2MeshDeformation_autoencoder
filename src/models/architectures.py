"""Model architecture definitions."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# !pip install torchvision
import torchvision

import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Comb model (Moser et al. 2021, "Semi-supervised video-driven facial
# animation transfer for production"): a single shared encoder feeding
# domain-specific decoders that share their first layers (the shared
# decoder stem / f_out), per the asterisked rows of the paper's Table 2.
#
# Adapted to our data:
#   - facemesh domain:     256x256x3 signed expression deltas (dx, dy, dz),
#                          rasterized from the MediaPipe mesh
#   - deformation domain:  256x256x3 signed deformation channels
#
# Both domains have the same 3-channel format, so a single fully-shared
# encoder processes both (as in the paper). The encoder and the decoder
# stem (up to f_out, 512x16x16) are shared; each domain then has its own
# decoder tail back to a 3x256x256 map.
# ---------------------------------------------------------------------------

DEFAULT_DOMAINS = {
    "facemesh": {"channels": 3, "out_activation": "tanh"},
    "deformation": {"channels": 3, "out_activation": "tanh"},
}


class ResidualBlock(nn.Module):
    """Two 3x3 convs with a skip connection (paper Table 2 conv pairs)."""

    def __init__(self, channels: int, slope: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(slope, inplace=True),
        )

    def forward(self, x):
        return x + self.block(x)


class SharedEncoder(nn.Module):
    """Single shared encoder (paper Table 1), adapted for 256x256 inputs.

    Because both domains are 3-channel, one encoder (identical weights)
    processes both, exactly as in the paper. 256 -> 8 spatially, doubling
    channels each step (32 -> 64 -> 128 -> 256 -> 512), then flatten -> z.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        latent_dim: int = 512,
        input_size: int = 256,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        size = input_size // 2
        ch = base_channels
        while size > 8:
            layers += [
                nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
            ]
            ch *= 2
            size //= 2
        self.body = nn.Sequential(*layers) # (*)unpacks the list into positional arguments
        self.out_channels = ch
        self.spatial = size  # 8
        self.bottleneck = nn.Linear(ch * size * size, latent_dim)

    def forward(self, x):
        h = self.body(x)
        return self.bottleneck(h.flatten(1))


class SharedDecoderStem(nn.Module):
    """Shared decoder layers (paper Table 2, the asterisked rows).

    Common across all domain decoders: latent -> Dense -> 512x8x8 -> Conv3x3
    -> PixelShuffle -> 512x16x16 (the paper's `f_out`). This is the last
    shared representation, which the paper (Sec. 5.5) found to be the best
    input for downstream geometry regression.
    """

    def __init__(self, latent_dim: int = 512, channels: int = 512, start_size: int = 8):
        super().__init__()
        self.channels = channels
        self.start_size = start_size
        self.dense = nn.Linear(latent_dim, channels * start_size * start_size)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.PixelShuffle(2),  # 8 -> 16, channels 4C -> C
        )
        self.out_channels = channels
        self.out_size = start_size * 2  # 16

    def forward(self, z):
        h = self.dense(z).view(-1, self.channels, self.start_size, self.start_size)
        return self.conv(h)  # -> channels x 16 x 16 (f_out)


class DomainDecoder(nn.Module):
    """Domain-specific decoder tail (paper Table 2, the non-asterisked rows).

    Consumes the shared f_out feature map (512x16x16) and upsamples with
    [Conv + PixelShuffle + ResidualBlock] stages to a 3x256x256 output map.
    """

    def __init__(
        self,
        out_channels: int,
        out_activation: str = "tanh",
        in_channels: int = 512,
        in_size: int = 16,
        output_size: int = 256,
    ):
        super().__init__()
        num_ups = int(np.log2(output_size // in_size))  # 16 -> output_size, 4 upsampling stages
        stages = []
        ch = in_channels
        for _ in range(num_ups):
            nxt = ch // 2
            stages += [
                nn.Conv2d(ch, nxt * 4, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.PixelShuffle(2),
                ResidualBlock(nxt),
            ]
            ch = nxt
        self.trunk = nn.Sequential(*stages)
        self.trunk_out = ch

        self.image_out = nn.Conv2d(ch, out_channels, 1)
        if out_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif out_activation == "tanh":
            self.activation = nn.Tanh()
        elif out_activation in (None, "none", "linear"):
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unknown out_activation: {out_activation}")

    def forward(self, f):
        return self.activation(self.image_out(self.trunk(f)))


class CombTranslationModel(nn.Module):
    """Comb model: shared encoder + shared decoder stem + one decoder tail
    per domain (paper Fig. 3 / Table 2).

    Translation goes source image -> shared latent z -> shared decoder stem
    (f_out) -> target decoder tail. Because the encoder is shared, encoding
    is domain-agnostic.

    domains maps a domain name to a dict with keys:
        channels (int), out_activation (str)
    """

    def __init__(
        self,
        domains: dict = None,
        latent_dim: int = 512,
        input_size: int = 256,
        in_channels: int = 3,
    ):
        super().__init__()
        domains = domains or DEFAULT_DOMAINS
        self.domain_names = list(domains)
        self.encoder = SharedEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            input_size=input_size,
        )
        self.decoder_stem = SharedDecoderStem(latent_dim=latent_dim)
        self.decoders = nn.ModuleDict(
            {
                name: DomainDecoder(
                    out_channels=cfg["channels"],
                    out_activation=cfg["out_activation"],
                    in_channels=self.decoder_stem.out_channels,
                    in_size=self.decoder_stem.out_size,
                    output_size=input_size,
                )
                for name, cfg in domains.items()
            }
        )

    def encode(self, x):
        return self.encoder(x)

    def shared_features(self, z):
        """Last shared decoder layer (f_out) — the paper's preferred input
        for downstream regression (Sec. 5.5)."""
        return self.decoder_stem(z)

    def decode(self, z, domain: str):
        return self.decoders[domain](self.decoder_stem(z))

    def forward(self, x, domain: str):
        """Autoencode within one domain (used for the reconstruction loss)."""
        return self.decode(self.encode(x), domain)

    def translate(self, x, target: str):
        """Map an image into the target domain (encoder is shared)."""
        return self.decode(self.encode(x), target)

    def cycle_latents(self, x, target: str, mask=None, z=None):
        """Latents for the cycle consistency loss (paper Eq. 3): z from the
        input, z* from re-encoding the cross-domain decode.

        As in the paper, the decoder weights (shared stem + target tail) are
        excluded from this term: their requires_grad is switched off while
        the translation graph is recorded, so gradients still flow through
        the decoder activations back to the encoder, but the decoder
        parameters do not accumulate them.

        `mask` (B, 1, H, W), when given, is applied to the translated map
        before it is re-encoded. The encoder is only ever trained on masked
        inputs, and the reconstruction loss leaves the decoder free outside
        the mask, so an unmasked re-encode would feed it a background it has
        never seen and make z* reflect that instead of the expression.

        `z`, when given, is the caller's already-computed encoding of `x` --
        reused rather than recomputed, so a training step that also needs the
        latent for other terms encodes `x` once. It must carry its gradient
        graph; passing a detached latent would cut the encoder out of Eq. 3.
        """
        z = self.encode(x) if z is None else z
        dec_params = list(self.decoder_stem.parameters()) + list(
            self.decoders[target].parameters()
        )
        states = [p.requires_grad for p in dec_params]
        for p in dec_params:
            p.requires_grad_(False)  #freeze decoder weights for L_CCL
        try:
            translated = self.decoders[target](self.decoder_stem(z))
        finally:
            for p, s in zip(dec_params, states):
                p.requires_grad_(s)
        if mask is not None:
            translated = translated * mask
        z_star = self.encode(translated)
        return z, z_star
