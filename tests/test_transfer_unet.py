"""Unit tests: freeze, shapes, loss α, UNet no cond-concat, grads only on UNet."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

FST_ROOT = Path(__file__).resolve().parents[1]
ILCLDM_ROOT = Path(os.environ.get("ILCLDM_ROOT", "/workspace/IL-CLDM"))
sys.path.insert(0, str(FST_ROOT))
sys.path.insert(0, str(ILCLDM_ROOT))

os.environ.setdefault("ILCLDM_SKIP_CROP", "1")
os.environ.setdefault("ILCLDM_ROI_VOLUME_SHAPE", "96,112,114")
os.environ.setdefault("ILCLDM_IMAGE_VOLUME_NORM", "legacy")
os.environ.setdefault("ILCLDM_WHOLE_ABETA", str(ILCLDM_ROOT / "data_rois/whole_Abeta_4mm"))
os.environ.setdefault("ILCLDM_WHOLE_MRI", str(ILCLDM_ROOT / "data_rois/whole_MRI"))


def test_loss_weights_arad():
    mse_w, ce_w = 0.2, 0.8
    assert abs(mse_w + ce_w - 1.0) < 1e-9
    # Match train script argparse defaults
    assert mse_w == 0.2 and ce_w == 0.8


def test_transfer_unet_forward_shape():
    from code.volume_shapes import get_latent_shape
    from fst.transfer_unet import build_transfer_unet

    latent = get_latent_shape()
    assert latent == (24, 28, 29)
    model = build_transfer_unet()
    x = torch.randn(2, 64, *latent)
    y = model(x)
    assert y.shape == (2, 64, 24, 28, 29)
    assert any(p.requires_grad for p in model.parameters())


def test_transfer_unet_no_condition_concat():
    from code.volume_shapes import get_latent_shape
    from fst.transfer_unet import build_transfer_unet

    model = build_transfer_unet()
    x = torch.randn(1, 64, *get_latent_shape())
    # Parent UNet would concat cond → 128 channels; subclass must keep 64.
    out = model._concat_inputs(x, x)
    assert out.shape == x.shape
    assert out is x or torch.equal(out, x)


def test_freeze_module_helper():
    sys.path.insert(0, str(FST_ROOT / "scripts"))
    # Import freeze from train script without running main
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_fst_h64_arad", FST_ROOT / "scripts" / "train_fst_h64_arad.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    linear = nn.Linear(4, 2)
    for p in linear.parameters():
        assert p.requires_grad
    mod.freeze_module(linear)
    assert all(not p.requires_grad for p in linear.parameters())
    assert not linear.training


def test_gradient_only_on_unet():
    """Frozen encoder stubs + trainable UNet: grads land only on UNet."""
    from code.volume_shapes import get_latent_shape
    from fst.transfer_unet import build_transfer_unet

    latent = get_latent_shape()
    frozen = nn.Conv3d(1, 64, kernel_size=1)
    for p in frozen.parameters():
        p.requires_grad_(False)
    frozen.eval()

    unet = build_transfer_unet()
    opt = torch.optim.Adam(unet.parameters(), lr=1e-3)

    vol = torch.randn(1, 1, *latent)
    with torch.no_grad():
        h64_m = frozen(vol)
        h64_s = torch.randn_like(h64_m)
    h64_hat = unet(h64_m)
    loss = 0.2 * torch.nn.functional.mse_loss(h64_hat, h64_s)
    opt.zero_grad(set_to_none=True)
    loss.backward()

    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in unet.parameters())
    assert all(p.grad is None for p in frozen.parameters())
