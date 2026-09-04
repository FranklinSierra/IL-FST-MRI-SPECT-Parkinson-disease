"""Frozen MobileNet feature extractors + SPECT head for clsfeat FST."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model, layers


FEATURE_LAYER = "conv_pw_13_relu"
HEAD_LAYERS = ("fc1", "fc2", "predictions")


def _enable_gpu_memory_growth() -> None:
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


def freeze_all(model: Model) -> None:
    model.trainable = False
    for layer in model.layers:
        layer.trainable = False


def load_classifier(path: Path | str) -> Model:
    _enable_gpu_memory_growth()
    model = tf.keras.models.load_model(str(path), compile=False)
    freeze_all(model)
    return model


def assert_classifier_integrity(model: Model, *, expect_feature: bool = True) -> None:
    names = {layer.name for layer in model.layers}
    if expect_feature and FEATURE_LAYER not in names:
        raise ValueError(f"Missing {FEATURE_LAYER} in {model.name}; layers={sorted(names)[:20]}...")
    for h in HEAD_LAYERS:
        if h not in names:
            raise ValueError(f"Missing head layer {h} in {model.name}")
    # Probe feature shape when Keras exposes it (some layers omit output_shape).
    feat_layer = model.get_layer(FEATURE_LAYER)
    shape = getattr(feat_layer, "output_shape", None)
    if shape is not None and isinstance(shape, (tuple, list)) and len(shape) == 4:
        if tuple(shape[1:]) != (8, 8, 1024):
            raise ValueError(f"{FEATURE_LAYER} shape {shape}, expected (None,8,8,1024)")


def build_feature_extractor(classifier: Model, name: str | None = None) -> Model:
    assert_classifier_integrity(classifier)
    feat = classifier.get_layer(FEATURE_LAYER).output
    return Model(inputs=classifier.input, outputs=feat, name=name or f"{classifier.name}_feat")


def _spect_head_forward(classifier: Model, feat_in: tf.Tensor) -> tf.Tensor:
    """GAP → fc1 → Dropout → fc2 on feature maps (shared weights from classifier)."""
    x = layers.GlobalAveragePooling2D(name="gap")(feat_in)
    fc1 = classifier.get_layer("fc1")
    drop = None
    for layer in classifier.layers:
        if isinstance(layer, layers.Dropout):
            drop = layer
            break
    fc2 = classifier.get_layer("fc2")

    x = fc1(x)
    if drop is not None:
        x = drop(x, training=False)
    else:
        x = layers.Dropout(0.5, name="dropout_head")(x, training=False)
    return fc2(x)


def build_fc2_extractor(classifier: Model, name: str = "spect_fc2") -> Model:
    """Rebuild GAP→fc1→Dropout→fc2 (512-D embedding) sharing weights from classifier."""
    assert_classifier_integrity(classifier)
    feat_in = layers.Input(shape=(8, 8, 1024), name="feat_in")
    emb = _spect_head_forward(classifier, feat_in)
    head = Model(inputs=feat_in, outputs=emb, name=name)
    freeze_all(head)
    return head


def build_spect_head(classifier: Model, name: str = "spect_head") -> Model:
    """Rebuild GAP→fc1→Dropout→fc2→predictions sharing weights from classifier."""
    assert_classifier_integrity(classifier)
    feat_in = layers.Input(shape=(8, 8, 1024), name="feat_in")
    x = _spect_head_forward(classifier, feat_in)
    pred = classifier.get_layer("predictions")
    logits = pred(x)
    head = Model(inputs=feat_in, outputs=logits, name=name)
    freeze_all(head)
    return head


def build_fst_train_model(
    mri_feat: Model,
    spect_feat: Model,
    transfer: Model,
    spect_head: Model,
    *,
    name: str = "fst_clsfeat",
) -> Model:
    """
    Inputs: (mri_tile, spect_tile).
    Outputs: f_hat (8x8x1024), predictions (2,).
    SPECT features for MSE are produced outside or via a second forward — see train script.
    """
    freeze_all(mri_feat)
    freeze_all(spect_feat)
    freeze_all(spect_head)
    transfer.trainable = True

    mri_in = layers.Input(shape=(256, 256, 3), name="mri_in")
    spect_in = layers.Input(shape=(256, 256, 3), name="spect_in")

    f_mri = mri_feat(mri_in)
    f_spect = spect_feat(spect_in)
    # Stop gradient into frozen SPECT encoder for the GT path is automatic (frozen).
    f_hat = transfer(f_mri)
    logits = spect_head(f_hat)

    model = Model(
        inputs=[mri_in, spect_in],
        outputs={"f_hat": f_hat, "f_spect": f_spect, "predictions": logits},
        name=name,
    )
    return model


def build_fst_infer_model(
    mri_feat: Model,
    transfer: Model,
    spect_head: Model,
    *,
    name: str = "fst_clsfeat_infer",
) -> Model:
    freeze_all(mri_feat)
    freeze_all(spect_head)
    mri_in = layers.Input(shape=(256, 256, 3), name="mri_in")
    f_hat = transfer(mri_feat(mri_in))
    logits = spect_head(f_hat)
    return Model(inputs=mri_in, outputs={"f_hat": f_hat, "predictions": logits}, name=name)


def count_trainable_weights(model: Model) -> int:
    return int(sum(int(tf.size(w)) for w in model.trainable_weights))
