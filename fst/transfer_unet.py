"""Feature-space transfer UNet: MRI h64 → SPECT h64 (no diffusion)."""

from __future__ import annotations

import torch

from model import UNet
from code.volume_shapes import get_latent_shape


class FeatureTransferUNet(UNet):
    """64→64 spatial UNet. Ignores diffusion-style condition concat.

    IL-CLDM ``UNet(with_time_emb=False)`` breaks on ``label_emb(time_dim=None)``,
    so we keep time_emb enabled and feed zero timesteps.
    """

    def __init__(self, image_size: int | None = None):
        if image_size is None:
            image_size = int(get_latent_shape()[0])
        super().__init__(
            in_channel=64,
            out_channel=64,
            with_time_emb=True,
            film_mode="off",
            image_size=image_size,
        )

    def _concat_inputs(self, x, cond):
        # Transfer maps a single h64 volume; do not concatenate a second stream.
        return x

    def forward(self, h64: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if h64.ndim != 5 or h64.shape[1] != 64:
            raise ValueError(f"Expected (B,64,D,H,W), got {tuple(h64.shape)}")
        time = torch.zeros(h64.shape[0], device=h64.device, dtype=h64.dtype)
        return super().forward(
            h64,
            h64,
            time,
            label=None,
            precomputed_condition=True,
            mri_film=None,
        )


def build_transfer_unet() -> FeatureTransferUNet:
    return FeatureTransferUNet()
