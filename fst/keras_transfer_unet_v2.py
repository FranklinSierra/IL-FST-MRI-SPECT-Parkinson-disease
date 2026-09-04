"""UNet de transferencia v2 para features conv_pw_13_relu (8,8,1024).

Correcciones frente a v1:
  * residual con `proj_up` inicializada a CERO -> el modelo ARRANCA en la identidad
    (baseline AUC 0.711) en lugar de en ruido; solo puede mejorar desde ahi.
  * salida ReLU6: el target f_spect es ReLU6, vive en [0,6] con ~79% de ceros.
    v1 tenia salida lineal -> distribucion imposible para la head congelada.
  * cuello de botella configurable (v1 comprimia 1024->64, tirando el 94% del ancho).
  * regularizacion L2 + dropout: 147 sujetos no soportan un traductor libre.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers


def _conv_block(x, filters, name, l2):
    reg = regularizers.l2(l2) if l2 else None
    x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg, name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg, name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.ReLU(name=f"{name}_relu2")(x)
    return x


def build_transfer_module_v2(
    in_channels: int = 1024,
    mid_channels: int = 192,
    *,
    l2: float = 1e-4,
    dropout: float = 0.2,
    residual: bool = True,
    name: str = "fst_transfer_v2",
) -> Model:
    reg = regularizers.l2(l2) if l2 else None
    inp = layers.Input(shape=(8, 8, in_channels), name=f"{name}_in")

    x = layers.Conv2D(mid_channels, 1, padding="same", kernel_regularizer=reg,
                      name=f"{name}_proj_down")(inp)
    x = layers.BatchNormalization(name=f"{name}_proj_bn")(x)
    x = layers.ReLU(name=f"{name}_proj_relu")(x)

    skip = _conv_block(x, mid_channels, f"{name}_enc", l2)
    x = layers.MaxPooling2D(2, name=f"{name}_pool")(skip)
    x = _conv_block(x, mid_channels, f"{name}_bottleneck", l2)
    x = layers.UpSampling2D(2, interpolation="bilinear", name=f"{name}_up")(x)
    x = layers.Concatenate(name=f"{name}_concat")([x, skip])
    x = layers.Conv2D(mid_channels, 3, padding="same", kernel_regularizer=reg,
                      name=f"{name}_dec_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_dec_bn")(x)
    x = layers.ReLU(name=f"{name}_dec_relu")(x)
    if dropout:
        x = layers.Dropout(dropout, name=f"{name}_drop")(x)

    # Zero-init => delta == 0 en el paso 0 => salida == identidad.
    delta = layers.Conv2D(
        in_channels, 1, padding="same",
        kernel_initializer="zeros", bias_initializer="zeros",
        kernel_regularizer=reg, name=f"{name}_proj_up",
    )(x)

    out = layers.Add(name=f"{name}_res")([inp, delta]) if residual else delta
    out = layers.ReLU(max_value=6.0, name=f"{name}_relu6")(out)
    return Model(inputs=inp, outputs=out, name=name)
