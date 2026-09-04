"""UNet 3D residual pequena para h64 (64,24,28,29). Enfoque A, v2.

Correcciones frente a v1 (`FeatureTransferUNet`, la UNet de difusion de IL-CLDM):
  * v1 tenia ~600M parametros (checkpoint de 2.4 GB) para 147 sujetos. Aqui ~0.3M.
  * residual con la conv final a CERO -> arranca EXACTAMENTE en la identidad,
    que en h64 ya vale AUC 0.646 (v1 arrancaba en ruido y nunca la alcanzo).
  * salida ReLU: h64 es la activacion oculta del AAE, no negativa (53% ceros).
  * sin embedding de tiempo ni FiLM: esto no es difusion, es una traduccion.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _blk(cin, cout, groups=8):
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(groups, cout), nn.SiLU(inplace=True),
        nn.Conv3d(cout, cout, 3, padding=1, bias=False),
        nn.GroupNorm(groups, cout), nn.SiLU(inplace=True),
    )


class ResidualTransferUNet3D(nn.Module):
    def __init__(self, channels: int = 64, mid: int = 32, dropout: float = 0.1):
        super().__init__()
        self.enc = _blk(channels, mid)
        self.bott = _blk(mid, mid)
        self.dec = _blk(2 * mid, mid)
        self.drop = nn.Dropout3d(dropout)
        self.out = nn.Conv3d(mid, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)          # delta == 0 en el paso 0

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.ndim != 5 or h.shape[1] != 64:
            raise ValueError(f"Se esperaba (B,64,D,H,W), llego {tuple(h.shape)}")
        s = self.enc(h)
        x = self.bott(F.avg_pool3d(s, 2))
        x = F.interpolate(x, size=s.shape[2:], mode="trilinear", align_corners=False)
        x = self.dec(torch.cat([x, s], dim=1))
        delta = self.out(self.drop(x))
        return F.relu(h + delta)


def build_transfer_unet_v2(**kw) -> ResidualTransferUNet3D:
    return ResidualTransferUNet3D(**kw)
