#!/usr/bin/env python3
"""Extract fc2(512) SPECT-head embeddings (fold 2) and train classic ML classifiers.

Pipeline: f_mri → transfer_fold2 → f_hat → frozen SPECT head → fc2.
Labels: control=0, pd_nhy1=1, pd_nhy2=2 (from nhy_distribution stratum).
Train on train_fit slices; select best model on val_holdout (subject majority vote);
report on testFold2 (independent).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FST = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FST))

from fst.keras_fst_models import build_fc2_extractor, load_classifier  # noqa: E402

EXP = FST / "result" / "fst_clsfeat_mnet2d_v2_z78_112" / "ownKFold"
DEFAULT_OUT = EXP / "models" / "fold2" / "embeddings"
STRATUM_TO_LABEL = {"control": 0, "pd_nhy1": 1, "pd_nhy2": 2}
LABEL_NAMES = {0: "control", 1: "pd_nhy1", 2: "pd_nhy2"}
MODEL_ORDER = [
    "knn_k3",
    "knn_k5",
    "knn_k7",
    "logistic_regression",
    "svm_rbf",
    "dnn_mlp",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, default=2)
    p.add_argument("--feat-cache", type=Path, default=FST / "data" / "feat_cache_z78_112")
    p.add_argument("--transfer", type=Path, default=EXP / "models" / "fold2" / "transfer_fold2.keras")
    p.add_argument(
        "--spect-classifier",
        type=Path,
        default=FST / "classifiers" / "classifiers_roi_4mm" / "spect" / "model.keras",
    )
    p.add_argument("--models-dir", type=Path, default=EXP / "models")
    p.add_argument("--nhy-csv", type=Path, default=FST / "data" / "nhy_distribution.csv")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--z-min", type=int, default=78)
    p.add_argument("--z-max", type=int, default=112)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dnn-epochs", type=int, default=80)
    p.add_argument("--skip-extract", action="store_true", help="reuse saved embeddings if present")
    return p.parse_args()


def load_ids(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def load_strata_labels(nhy_csv: Path) -> dict[str, int]:
    df = pd.read_csv(nhy_csv, dtype={"ID": str})
    df["stratum"] = df["stratum"].fillna("control").astype(str)
    out: dict[str, int] = {}
    for _, row in df.iterrows():
        sid = str(row["ID"])
        strat = row["stratum"]
        if strat not in STRATUM_TO_LABEL:
            raise ValueError(f"Unknown stratum {strat!r} for subject {sid}")
        out[sid] = STRATUM_TO_LABEL[strat]
    return out


def configure_gpu() -> None:
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass


def build_fc2_pipeline(transfer: tf.keras.Model, spect_classifier: tf.keras.Model) -> tf.keras.Model:
    fc2_head = build_fc2_extractor(spect_classifier, name="spect_fc2_fold2")
    f_in = tf.keras.Input(shape=(8, 8, 1024), name="f_mri_in")
    f_hat = transfer(f_in)
    emb = fc2_head(f_hat)
    return tf.keras.Model(f_in, emb, name="fc2_embed_fold2")


@tf.function
def _fwd(model, x):
    return model(x, training=False)


def extract_embeddings(model, fm, row_idx: np.ndarray, batch_size: int) -> np.ndarray:
    out = []
    for i in range(0, len(row_idx), batch_size):
        b = row_idx[i : i + batch_size]
        x = np.asarray(fm[b], dtype=np.float32)
        out.append(_fwd(model, x).numpy())
    return np.concatenate(out, axis=0)


def assign_split(subject_id: str, train_fit: set[str], val_ids: set[str], test_ids: set[str]) -> str:
    if subject_id in test_ids:
        return "test"
    if subject_id in val_ids:
        return "val"
    if subject_id in train_fit:
        return "train_fit"
    return "other"


class KerasMLPClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible 512→128→3 MLP with internal validation split for early stopping."""

    def __init__(
        self,
        hidden: int = 128,
        dropout: float = 0.3,
        epochs: int = 80,
        batch_size: int = 64,
        seed: int = 42,
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.model_ = None
        self.classes_ = None

    def fit(self, X, y):
        tf.keras.utils.set_random_seed(self.seed)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        with tf.device("/CPU:0"):
            self.model_ = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(X.shape[1],)),
                    tf.keras.layers.Dense(self.hidden, activation="relu"),
                    tf.keras.layers.Dropout(self.dropout),
                    tf.keras.layers.Dense(n_classes, activation="softmax"),
                ],
                name="mlp_fc2_3class",
            )
            self.model_.compile(
                optimizer=tf.keras.optimizers.Adam(1e-3),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            cw = {}
            counts = {int(c): int(np.sum(y == c)) for c in self.classes_}
            total = len(y)
            for c, n in counts.items():
                cw[c] = total / (n_classes * max(n, 1))
            self.model_.fit(
                X,
                y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                class_weight=cw,
                verbose=0,
            )
        return self

    def predict_proba(self, X):
        with tf.device("/CPU:0"):
            return self.model_.predict(X, verbose=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def build_candidates(seed: int, dnn_epochs: int) -> dict[str, object]:
    return {
        "knn_k3": Pipeline(
            [("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=3))]
        ),
        "knn_k5": Pipeline(
            [("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]
        ),
        "knn_k7": Pipeline(
            [("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=7))]
        ),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced")),
            ]
        ),
        "dnn_mlp": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KerasMLPClassifier(epochs=dnn_epochs, seed=seed)),
            ]
        ),
    }


def aggregate_subject_predictions(
    df: pd.DataFrame, y_pred_slice: np.ndarray, y_proba_slice: np.ndarray
) -> pd.DataFrame:
    """Majority vote for hard labels; mean probabilities for AUC."""
    d = df.copy()
    d["pred_slice"] = y_pred_slice
    for c in range(y_proba_slice.shape[1]):
        d[f"proba_{c}"] = y_proba_slice[:, c]

    def _vote(s):
        vals, counts = np.unique(s, return_counts=True)
        return int(vals[np.argmax(counts)])

    g = d.groupby("subject_id", as_index=False).agg(
        label=("label_3class", "first"),
        pred=("pred_slice", _vote),
        **{f"proba_{c}": (f"proba_{c}", "mean") for c in range(y_proba_slice.shape[1])},
    )
    proba_cols = [f"proba_{c}" for c in range(y_proba_slice.shape[1])]
    g["proba"] = g[proba_cols].values.tolist()
    return g


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    m: dict = {
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }
    try:
        if len(np.unique(y_true)) > 1:
            m["macro_auc"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            )
        else:
            m["macro_auc"] = float("nan")
    except ValueError:
        m["macro_auc"] = float("nan")
    return m


def select_best_model(rows: list[dict]) -> str:
    def key(r):
        return (
            r["val_macro_recall"],
            r["val_macro_f1"],
            -MODEL_ORDER.index(r["model"]),
        )

    return max(rows, key=key)["model"]


def main():
    a = parse_args()
    configure_gpu()
    tf.keras.utils.set_random_seed(a.seed)
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)

    fold_dir = a.models_dir / f"fold{a.fold}"
    train_fit_ids = set(load_ids(fold_dir / "train_fit.txt"))
    val_ids = set(load_ids(fold_dir / "val_holdout.txt"))
    test_ids = set(load_ids(fold_dir / "testFold.txt"))
    strata_labels = load_strata_labels(a.nhy_csv)

    idx = pd.read_csv(a.feat_cache / "index.csv", dtype={"subject_id": str})
    keep = (idx.slice_idx >= a.z_min) & (idx.slice_idx <= a.z_max)
    idx = idx.loc[keep].reset_index(drop=True)
    idx["global_row"] = np.arange(len(idx))

    all_ids = train_fit_ids | val_ids | test_ids
    missing = sorted(all_ids - set(idx.subject_id))
    if missing:
        raise SystemExit(f"Subjects missing from feature cache: {missing[:10]}")

    idx["label_3class"] = idx.subject_id.map(strata_labels)
    if idx["label_3class"].isna().any():
        bad = idx.loc[idx.label_3class.isna(), "subject_id"].unique()[:5]
        raise SystemExit(f"Missing stratum labels for subjects: {bad}")

    idx["split"] = idx.subject_id.map(
        lambda s: assign_split(s, train_fit_ids, val_ids, test_ids)
    )
    pool = idx[idx.subject_id.isin(all_ids)].copy().reset_index(drop=True)
    row_idx = pool.global_row.values

    emb_path = out / "fc2_embeddings.npy"
    index_path = out / "embeddings_index.csv"

    if a.skip_extract and emb_path.is_file() and index_path.is_file():
        embeddings = np.load(emb_path)
        pool = pd.read_csv(index_path, dtype={"subject_id": str})
        print(f"Loaded cached embeddings {embeddings.shape}", flush=True)
    else:
        print("Loading transfer + SPECT classifier...", flush=True)
        transfer = tf.keras.models.load_model(str(a.transfer), compile=False)
        spect = load_classifier(a.spect_classifier)
        pipe = build_fc2_pipeline(transfer, spect)
        fm = np.load(a.feat_cache / "f_mri.npy", mmap_mode="r")
        print(f"Extracting fc2 for {len(row_idx)} slices...", flush=True)
        embeddings = extract_embeddings(pipe, fm, row_idx, a.batch_size)
        np.save(emb_path, embeddings.astype(np.float32))
        nhy = pd.read_csv(a.nhy_csv, dtype={"ID": str})
        nhy_map = dict(zip(nhy["ID"].astype(str), nhy["stratum"].fillna("control")))
        pool = pool.assign(stratum=pool.subject_id.map(nhy_map))
        pool[["subject_id", "slice_idx", "label_3class", "stratum", "split", "global_row"]].to_csv(
            index_path, index=False
        )
        print(f"Saved embeddings {embeddings.shape} -> {emb_path}", flush=True)
        del transfer, spect, pipe, fm
        tf.keras.backend.clear_session()

    m_train = pool.split.values == "train_fit"
    m_val = pool.split.values == "val"
    m_test = pool.split.values == "test"

    train_df = pool.loc[m_train].reset_index(drop=True)
    val_df = pool.loc[m_val].reset_index(drop=True)
    test_df = pool.loc[m_test].reset_index(drop=True)

    X_train = embeddings[m_train]
    y_train = train_df.label_3class.values.astype(int)
    X_val = embeddings[m_val]
    X_test = embeddings[m_test]

    candidates = build_candidates(a.seed, a.dnn_epochs)
    comparison_rows = []
    fitted: dict[str, object] = {}

    for name, model in candidates.items():
        print(f"Training {name}...", flush=True)
        model.fit(X_train, y_train)
        fitted[name] = model

        proba_val = model.predict_proba(X_val)
        pred_val = np.argmax(proba_val, axis=1)
        subj_val = aggregate_subject_predictions(val_df, pred_val, proba_val)
        yv = subj_val.label.values.astype(int)
        yp = subj_val.pred.values.astype(int)
        proba_subj = np.stack(subj_val.proba.values)
        val_m = compute_metrics(yv, yp, proba_subj)

        proba_test = model.predict_proba(X_test)
        pred_test = np.argmax(proba_test, axis=1)
        subj_test = aggregate_subject_predictions(test_df, pred_test, proba_test)
        yt = subj_test.label.values.astype(int)
        ypt = subj_test.pred.values.astype(int)
        proba_test_subj = np.stack(subj_test.proba.values)
        test_m = compute_metrics(yt, ypt, proba_test_subj)

        comparison_rows.append(
            {
                "model": name,
                "val_macro_recall": val_m["macro_recall"],
                "val_macro_f1": val_m["macro_f1"],
                "val_balanced_accuracy": val_m["balanced_accuracy"],
                "val_macro_auc": val_m["macro_auc"],
                "test_macro_recall": test_m["macro_recall"],
                "test_macro_f1": test_m["macro_f1"],
                "test_balanced_accuracy": test_m["balanced_accuracy"],
                "test_macro_auc": test_m["macro_auc"],
            }
        )
        print(
            f"  val: recall={val_m['macro_recall']:.4f} f1={val_m['macro_f1']:.4f} "
            f"auc={val_m['macro_auc']:.4f}",
            flush=True,
        )

    best_name = select_best_model(comparison_rows)
    for row in comparison_rows:
        row["selected"] = row["model"] == best_name

    comp_df = pd.DataFrame(comparison_rows).sort_values(
        ["selected", "val_macro_recall", "val_macro_f1"], ascending=[False, False, False]
    )
    comp_path = out / "model_comparison.csv"
    comp_df.to_csv(comp_path, index=False)

    best_model = fitted[best_name]
    joblib.dump(best_model, out / "best_model.joblib")

    proba_test = best_model.predict_proba(X_test)
    pred_test = np.argmax(proba_test, axis=1)
    subj_test = aggregate_subject_predictions(test_df, pred_test, proba_test)
    yt = subj_test.label.values.astype(int)
    ypt = subj_test.pred.values.astype(int)
    proba_test_subj = np.stack(subj_test.proba.values)
    test_m = compute_metrics(yt, ypt, proba_test_subj)

    report_txt = classification_report(
        yt,
        ypt,
        target_names=[LABEL_NAMES[i] for i in [0, 1, 2]],
        digits=4,
        zero_division=0,
    )
    report_path = out / "classification_report_fold2.txt"
    header = (
        f"Best model (selected on val_holdout): {best_name}\n"
        f"Selection: macro recall (tie-break macro F1)\n"
        f"Test subjects: n={len(yt)} (majority vote over slices)\n"
        f"macro AUC (OvR): {test_m['macro_auc']:.4f}\n"
        f"macro recall: {test_m['macro_recall']:.4f}\n"
        f"macro F1: {test_m['macro_f1']:.4f}\n"
        f"balanced accuracy: {test_m['balanced_accuracy']:.4f}\n\n"
    )
    report_path.write_text(header + report_txt)

    subj_test.assign(
        true_label=subj_test.label.map(LABEL_NAMES),
        pred_label=subj_test.pred.map(LABEL_NAMES),
    ).to_csv(out / "test_predictions_fold2.csv", index=False)

    config = {
        "fold": a.fold,
        "transfer": str(a.transfer),
        "spect_classifier": str(a.spect_classifier),
        "feat_cache": str(a.feat_cache),
        "z_range": [a.z_min, a.z_max],
        "embedding_dim": 512,
        "embedding_layer": "fc2",
        "label_map": STRATUM_TO_LABEL,
        "train_fit_subjects": len(train_fit_ids),
        "val_subjects": len(val_ids),
        "test_subjects": len(test_ids),
        "train_slices": int(len(train_df)),
        "val_slices": int(len(val_df)),
        "test_slices": int(len(test_df)),
        "best_model": best_name,
        "selection_metric": "macro_recall",
        "tiebreak_metric": "macro_f1",
        "aggregation": "majority_vote_slices",
        "proba_aggregation": "mean",
        "seed": a.seed,
    }
    metrics = {
        "best_model": best_name,
        "test": test_m,
        "n_test_subjects": int(len(yt)),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "metrics_fold2.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nBest model: {best_name}", flush=True)
    print(f"Test macro recall={test_m['macro_recall']:.4f} macro_f1={test_m['macro_f1']:.4f}", flush=True)
    print(f"Artifacts in {out}", flush=True)


if __name__ == "__main__":
    main()
