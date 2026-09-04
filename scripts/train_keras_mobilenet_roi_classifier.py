#!/usr/bin/env python3
"""Train Keras MobileNet v1 (conv_pw_13_relu / fc1/fc2/predictions) on ROI 4 mm.

Outputs model.keras under --out-dir for FST clsfeat gates.
Default axial range z=[78, 112] (35 slices). Softmax 2-class head.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

FST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FST_ROOT))
sys.path.insert(0, "/workspace/CoCoLIT/src")

from cocolit.classification.cuda_env import ensure_nvidia_libs_for_tf  # noqa: E402

ensure_nvidia_libs_for_tf(reexec=True)

import tensorflow as tf  # noqa: E402
from tensorflow.keras.utils import Sequence, to_categorical  # noqa: E402

from cocolit.classification.spect_classifier import (  # noqa: E402
    evaluate_subject_level,
    make_model,
    train_classifier,
)
from fst.slice_data import (  # noqa: E402
    load_id_list,
    load_labels,
    map_volume_to_unit_interval,
    slice_indices,
    slice_to_rgb,
    stratified_holdout,
    write_id_list,
)

DEFAULT_SLICE_RANGE = (78, 112)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modality", choices=["mri", "spect"], required=True)
    p.add_argument("--volume-root", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--train-list", type=Path, default=Path(os.environ.get("FST_TRAIN_LIST", "")))
    p.add_argument("--test-list", type=Path, default=Path(os.environ.get("FST_TEST_LIST", "")))
    p.add_argument("--meta-csv", type=Path, default=Path(os.environ.get("FST_META_CSV", "")))
    p.add_argument("--fit-list", type=Path, default=None)
    p.add_argument("--val-list", type=Path, default=None)
    p.add_argument("--n-val-holdout", type=int, default=15)
    p.add_argument("--z-min", type=int, default=DEFAULT_SLICE_RANGE[0])
    p.add_argument("--z-max", type=int, default=DEFAULT_SLICE_RANGE[1])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tile-cache", type=Path, default=None, help="Optional dir for .npy tiles")
    return p.parse_args()


def default_volume_root(modality: str) -> Path:
    if modality == "mri":
        return Path(os.environ.get("ILCLDM_WHOLE_MRI", "/workspace/IL-CLDM/data_rois/whole_MRI"))
    return Path(os.environ.get("ILCLDM_WHOLE_ABETA", "/workspace/IL-CLDM/data_rois/whole_Abeta_4mm"))


def resolve_splits(args, labels: dict[str, int]):
    train_all = [s for s in load_id_list(args.train_list) if s in labels]
    test_ids = [s for s in load_id_list(args.test_list) if s in labels]
    if args.fit_list and args.val_list and args.fit_list.is_file() and args.val_list.is_file():
        return load_id_list(args.fit_list), load_id_list(args.val_list), test_ids
    fit, val = stratified_holdout(train_all, labels, n_val=args.n_val_holdout, seed=args.seed)
    return fit, val, test_ids


def materialize_manifest(
    subject_ids: list[str],
    labels: dict[str, int],
    volume_root: Path,
    modality: str,
    z_min: int,
    z_max: int,
    tile_cache: Path,
) -> pd.DataFrame:
    """Write 256x256x3 float32 .npy tiles and return CoCoLIT-style manifest."""
    tile_cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in subject_ids:
        path = volume_root / f"{sid}.nii.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
        unit = map_volume_to_unit_interval(raw, modality)  # type: ignore[arg-type]
        zs = slice_indices(z_min, z_max, unit.shape, axis=2)
        sub_dir = tile_cache / sid
        sub_dir.mkdir(parents=True, exist_ok=True)
        y = int(labels[sid])
        for z in zs:
            out = sub_dir / f"{sid}_slice_{z:03d}.npy"
            if not out.is_file():
                np.save(out, slice_to_rgb(unit[:, :, z]))
            rows.append({"path": str(out), "label": y, "subject_id": sid, "slice_idx": z})
    return pd.DataFrame(rows)


class RamSliceSequence(Sequence):
    """In-memory tiles (no disk). Built once from volumes."""

    def __init__(self, xs: np.ndarray, ys: np.ndarray, sids: list[str], batch_size: int, shuffle: bool, seed: int):
        self.xs = xs
        self.ys = ys
        self.sids = sids
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.indices = np.arange(len(xs))
        self.on_epoch_end()

    def __len__(self):
        return max(1, int(np.ceil(len(self.xs) / self.batch_size)))

    def __getitem__(self, index):
        batch_idx = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        x = self.xs[batch_idx]
        y = to_categorical(self.ys[batch_idx], num_classes=2)
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(self.indices)

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame({"path": [""] * len(self.sids), "label": self.ys, "subject_id": self.sids})


def load_tiles_ram(
    subject_ids: list[str],
    labels: dict[str, int],
    volume_root: Path,
    modality: str,
    z_min: int,
    z_max: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    tiles = []
    ys = []
    sids = []
    zs_out = []
    for sid in subject_ids:
        path = volume_root / f"{sid}.nii.gz"
        raw = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
        unit = map_volume_to_unit_interval(raw, modality)  # type: ignore[arg-type]
        zs = slice_indices(z_min, z_max, unit.shape, axis=2)
        y = int(labels[sid])
        for z in zs:
            tiles.append(slice_to_rgb(unit[:, :, z]))
            ys.append(y)
            sids.append(sid)
            zs_out.append(z)
    return np.stack(tiles, axis=0).astype(np.float32), np.asarray(ys, dtype=np.int32), sids, zs_out


def train_from_ram(args, fit_ids, val_ids, test_ids, labels, volume_root):
    """Train without writing slice cache (saves disk); still saves model.keras."""
    from cocolit.classification.spect_classifier import SubjectLevelEarlyStopping

    print("Loading fit tiles into RAM…", flush=True)
    x_tr, y_tr, sid_tr, zs = load_tiles_ram(
        fit_ids, labels, volume_root, args.modality, args.z_min, args.z_max
    )
    print("Loading val tiles into RAM…", flush=True)
    x_va, y_va, sid_va, _ = load_tiles_ram(
        val_ids, labels, volume_root, args.modality, args.z_min, args.z_max
    )
    print(
        f"tiles fit/val={len(x_tr)}/{len(x_va)} | z={zs[0] if zs else '?'}… "
        f"unique_z={len(set(zs))}",
        flush=True,
    )

    train_seq = RamSliceSequence(x_tr, y_tr, sid_tr, args.batch_size, shuffle=True, seed=args.seed)
    valid_seq = RamSliceSequence(x_va, y_va, sid_va, args.batch_size, shuffle=False, seed=args.seed)
    valid_df = pd.DataFrame({"path": [""] * len(sid_va), "label": y_va, "subject_id": sid_va})

    # Patch Sequence used by evaluate: need real predict path via arrays
    class ArrayPredictSequence(Sequence):
        def __init__(self, x, y, sids, batch_size):
            self.x, self.y, self.sids = x, y, sids
            self.batch_size = batch_size
            self.df = pd.DataFrame({"path": [""] * len(sids), "label": y, "subject_id": sids})

        def __len__(self):
            return max(1, int(np.ceil(len(self.x) / self.batch_size)))

        def __getitem__(self, index):
            sl = slice(index * self.batch_size, (index + 1) * self.batch_size)
            return self.x[sl], to_categorical(self.y[sl], 2)

    model = make_model("MobileNet", 256, 256, 2)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "model.keras"

    # Monkey-patch evaluate to use arrays: wrap model.predict
    def evaluate_arrays(mdl, x, y, sids, batch_size):
        logits = mdl.predict(x, batch_size=batch_size, verbose=0)
        pred_class = np.argmax(logits, axis=1)
        df = pd.DataFrame(
            {
                "subject_id": sids,
                "label": y,
                "pred_class": pred_class,
                "pred_prob_pd": logits[:, 1],
            }
        )
        rows = []
        for sid, grp in df.groupby("subject_id"):
            votes = grp["pred_class"].value_counts()
            rows.append(
                {
                    "subject_id": str(sid),
                    "label": int(grp["label"].iloc[0]),
                    "pred_class": int(votes.idxmax()),
                    "n_slices": len(grp),
                    "mean_prob_pd": float(grp["pred_prob_pd"].mean()),
                }
            )
        subj = pd.DataFrame(rows)
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            roc_auc_score,
        )

        yt, yp = subj["label"].values, subj["pred_class"].values
        metrics = {
            "n_subjects": int(len(subj)),
            "accuracy": float(accuracy_score(yt, yp)),
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(yt, yp).tolist(),
            "classification_report": classification_report(yt, yp, output_dict=True, zero_division=0),
            "per_subject": subj.to_dict(orient="records"),
        }
        if len(np.unique(yt)) > 1:
            metrics["auc"] = float(roc_auc_score(yt, subj["mean_prob_pd"].values))
        return metrics

    class ArrayEarlyStop(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best = -np.inf
            self.best_epoch = 0
            self.wait = 0
            self.best_weights = None

        def on_epoch_end(self, epoch, logs=None):
            metrics = evaluate_arrays(self.model, x_va, y_va, sid_va, args.batch_size)
            score = float(metrics["balanced_accuracy"])
            if logs is not None:
                logs["val_subject_balanced_accuracy"] = score
            print(
                f"  val_subject_BA={score:.4f} auc={metrics.get('auc', float('nan')):.4f}",
                flush=True,
            )
            if score > self.best:
                self.best = score
                self.best_epoch = int(epoch) + 1
                self.wait = 0
                self.best_weights = self.model.get_weights()
                self.model.save(model_path)
            else:
                self.wait += 1
                if self.wait >= args.patience:
                    self.model.stop_training = True
                    if self.best_weights is not None:
                        self.model.set_weights(self.best_weights)

    cb = ArrayEarlyStop()
    history = model.fit(
        train_seq,
        validation_data=valid_seq,
        epochs=args.epochs,
        verbose=1,
        callbacks=[cb],
    )

    if model_path.exists():
        model = tf.keras.models.load_model(str(model_path))

    val_metrics = evaluate_arrays(model, x_va, y_va, sid_va, args.batch_size)
    print("Loading test tiles…", flush=True)
    x_te, y_te, sid_te, _ = load_tiles_ram(
        test_ids, labels, volume_root, args.modality, args.z_min, args.z_max
    )
    test_metrics = evaluate_arrays(model, x_te, y_te, sid_te, args.batch_size)

    # Integrity for FST
    names = {l.name for l in model.layers}
    assert "conv_pw_13_relu" in names, names
    assert "fc1" in names and "fc2" in names and "predictions" in names

    config = {
        "framework": "keras",
        "architecture": "MobileNet",
        "modality": args.modality,
        "feature_layer": "conv_pw_13_relu",
        "feature_shape": [8, 8, 1024],
        "head_layers": ["fc1", "fc2", "predictions"],
        "num_classes": 2,
        "activation_out": "softmax",
        "height": 256,
        "width": 256,
        "slice_axis": 2,
        "slice_range": [args.z_min, args.z_max],
        "n_slices_per_subject": len(set(zs)),
        "preprocess": "mri_zscore_clip3sigma_to_unit"
        if args.modality == "mri"
        else "spect_p1_p99_to_unit",
        "best_epoch": cb.best_epoch,
        "best_val_balanced_accuracy": cb.best,
        "n_train_subjects": len(fit_ids),
        "n_val_subjects": len(val_ids),
        "n_test_subjects": len(test_ids),
        "checkpoint": str(model_path),
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (args.out_dir / "history.json").write_text(
        json.dumps({k: [float(v) for v in vals] for k, vals in history.history.items()}, indent=2)
    )
    (args.out_dir / "metrics_valid.json").write_text(json.dumps(val_metrics, indent=2))
    (args.out_dir / "metrics_test.json").write_text(json.dumps(test_metrics, indent=2))

    report = test_metrics.get("classification_report", {})
    lines = ["", "              precision    recall  f1-score   support", ""]
    # Keep a sklearn text report too
    from sklearn.metrics import classification_report as cr

    pred = [r["pred_class"] for r in test_metrics["per_subject"]]
    lab = [r["label"] for r in test_metrics["per_subject"]]
    text = cr(lab, pred, digits=4, zero_division=0)
    (args.out_dir / "classification_report_test.txt").write_text(text)

    print("\n=== TEST subject-level ===", flush=True)
    print(text, flush=True)
    print(json.dumps({k: test_metrics[k] for k in ("accuracy", "balanced_accuracy", "auc", "f1_macro") if k in test_metrics}, indent=2), flush=True)
    print(f"Saved {model_path}", flush=True)
    return model


def main():
    args = parse_args()
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    volume_root = args.volume_root or default_volume_root(args.modality)
    labels = load_labels(args.meta_csv)
    fit_ids, val_ids, test_ids = resolve_splits(args, labels)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_id_list(args.out_dir / "train_fit.txt", fit_ids)
    write_id_list(args.out_dir / "val_holdout.txt", val_ids)
    write_id_list(args.out_dir / "test.txt", test_ids)

    print(
        f"modality={args.modality} out={args.out_dir} "
        f"slice=[{args.z_min},{args.z_max}] fit/val/test={len(fit_ids)}/{len(val_ids)}/{len(test_ids)}",
        flush=True,
    )
    vc = sum(1 for s in val_ids if labels[s] == 0)
    print(f"val balance: control={vc} PD={len(val_ids) - vc}", flush=True)

    if args.tile_cache is not None:
        # Disk-backed path using CoCoLIT train_classifier
        train_df = materialize_manifest(
            fit_ids, labels, volume_root, args.modality, args.z_min, args.z_max, args.tile_cache / "train"
        )
        valid_df = materialize_manifest(
            val_ids, labels, volume_root, args.modality, args.z_min, args.z_max, args.tile_cache / "valid"
        )
        test_df = materialize_manifest(
            test_ids, labels, volume_root, args.modality, args.z_min, args.z_max, args.tile_cache / "test"
        )
        model, _ = train_classifier(
            train_df,
            valid_df,
            args.out_dir,
            architecture="MobileNet",
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.lr,
            modality=args.modality,
            preprocess="mri_zscore_clip3sigma_to_unit"
            if args.modality == "mri"
            else "spect_p1_p99_to_unit",
            slice_range=(args.z_min, args.z_max),
        )
        from cocolit.classification.spect_classifier import save_test_metrics

        save_test_metrics(model, test_df, args.out_dir, batch_size=args.batch_size)
    else:
        train_from_ram(args, fit_ids, val_ids, test_ids, labels, volume_root)


if __name__ == "__main__":
    main()
