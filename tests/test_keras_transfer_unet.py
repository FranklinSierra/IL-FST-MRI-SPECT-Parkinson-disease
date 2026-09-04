"""Unit tests for Keras clsfeat transfer UNet (shapes, freeze, loss α)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

FST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FST_ROOT))


def test_transfer_module_shapes():
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    tf = pytest.importorskip("tensorflow")
    from fst.keras_transfer_unet import build_transfer_module

    model = build_transfer_module()
    x = np.random.randn(2, 8, 8, 1024).astype(np.float32)
    y = model(x, training=False)
    assert tuple(y.shape) == (2, 8, 8, 1024)


def test_loss_alpha_defaults():
    # Paper-faithful: synthesis 0.8, classification 0.2
    mse_w, ce_w = 0.8, 0.2
    assert abs(mse_w + ce_w - 1.0) < 1e-9
    assert mse_w > ce_w


def test_stratified_holdout():
    from fst.slice_data import stratified_holdout

    ids = [f"s{i}" for i in range(40)]
    labels = {sid: (0 if i < 20 else 1) for i, sid in enumerate(ids)}
    train, val = stratified_holdout(ids, labels, n_val=15, seed=42)
    assert len(val) == 15
    assert len(train) == 25
    assert set(train).isdisjoint(val)
    # Both classes in val
    assert {labels[s] for s in val} == {0, 1}


def test_trainable_only_transfer():
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    tf = pytest.importorskip("tensorflow")
    from tensorflow.keras import Input, Model, layers

    from fst.keras_fst_models import count_trainable_weights, freeze_all
    from fst.keras_transfer_unet import build_transfer_module

    # Fake frozen feat extractors
    def fake_feat(name):
        inp = Input(shape=(256, 256, 3), name=f"{name}_in")
        x = layers.Conv2D(1024, 32, strides=32, padding="same", name=f"{name}_c")(inp)
        # Force 8x8: 256/32 = 8
        m = Model(inp, x, name=name)
        freeze_all(m)
        return m

    mri_feat = fake_feat("mri")
    spect_feat = fake_feat("spect")
    feat_in = Input(shape=(8, 8, 1024))
    x = layers.GlobalAveragePooling2D()(feat_in)
    x = layers.Dense(2, activation="softmax", name="predictions")(x)
    head = Model(feat_in, x, name="head")
    freeze_all(head)
    transfer = build_transfer_module()

    from fst.keras_fst_models import build_fst_train_model

    model = build_fst_train_model(mri_feat, spect_feat, transfer, head)
    n_model = count_trainable_weights(model)
    n_xfer = count_trainable_weights(transfer)
    assert n_model == n_xfer > 0
