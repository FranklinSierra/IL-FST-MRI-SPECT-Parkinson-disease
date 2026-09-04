"""Small Keras UNet for classifier-feature transfer (8x8x1024 via 64-ch body)."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers


def conv_block(x, filters: int, name: str):
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.ReLU(name=f"{name}_relu2")(x)
    return x


def build_transfer_module(
    in_channels: int = 1024,
    mid_channels: int = 64,
    name: str = "fst_transfer",
) -> Model:
    """
    Proj↓ 1024→64 → UNet body (8↔4) → Proj↑ 64→1024.

    Body: Conv3x3x2(+BN,ReLU) + skip → MaxPool → Conv3x3x2 bottleneck →
    Upsample → concat → Conv3x3+BN+ReLU → (8,8,64).
    """
    inp = layers.Input(shape=(8, 8, in_channels), name=f"{name}_in")
    x = layers.Conv2D(mid_channels, 1, padding="same", name=f"{name}_proj_down")(inp)

    skip = conv_block(x, mid_channels, name=f"{name}_enc")
    x = layers.MaxPooling2D(2, name=f"{name}_pool")(skip)

    x = conv_block(x, mid_channels, name=f"{name}_bottleneck")

    x = layers.UpSampling2D(2, interpolation="bilinear", name=f"{name}_up")(x)
    x = layers.Concatenate(name=f"{name}_concat")([x, skip])
    # After concat (8,8,128): 3x3 + BN + ReLU back to 64 (UNet-like).
    x = layers.Conv2D(mid_channels, 3, padding="same", name=f"{name}_dec_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_dec_bn")(x)
    x = layers.ReLU(name=f"{name}_dec_relu")(x)

    out = layers.Conv2D(in_channels, 1, padding="same", name=f"{name}_proj_up")(x)
    return Model(inputs=inp, outputs=out, name=name)
