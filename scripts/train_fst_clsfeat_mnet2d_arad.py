#!/usr/bin/env python3
"""Train ARAD-style FST on MobileNet conv_pw_13 features (MRI→SPECT), Keras 2D UNet.

Experiment id: fst_clsfeat_mnet2d_arad
L = 0.8 * MSE(f_hat, f_spect) + 0.2 * CE(Head_SPECT(f_hat), y)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

FST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FST_ROOT))

# GPU by default. Set FST_TF_FORCE_GPU=0 to force CPU.
if os.environ.get("FST_TF_FORCE_GPU", "1") == "0":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _ensure_tf_cuda() -> None:
    """Prepend pip NVIDIA libs and re-exec once if needed (CoCoLIT helper)."""
    try:
        sys.path.insert(0, "/workspace/CoCoLIT/src")
        from cocolit.classification.cuda_env import ensure_nvidia_libs_for_tf

        ensure_nvidia_libs_for_tf(reexec=True)
    except Exception as exc:
        print(f"WARN: TF CUDA lib bootstrap skipped: {exc}")


_ensure_tf_cuda()

import tensorflow as tf  # noqa: E402

from fst.keras_fst_models import (  # noqa: E402
    build_feature_extractor,
    build_fst_infer_model,
    build_fst_train_model,
    build_spect_head,
    count_trainable_weights,
    freeze_all,
    load_classifier,
)
from fst.keras_transfer_unet import build_transfer_module  # noqa: E402
from fst.slice_data import (  # noqa: E402
    DEFAULT_SLICE_RANGE,
    attach_labels,
    ensure_subject_slices,
    load_id_list,
    load_labels,
    make_paired_sequence,
    stratified_holdout,
    write_id_list,
)

EXP_ID = "fst_clsfeat_mnet2d_arad"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mri-root", type=Path, default=Path(os.environ.get("ILCLDM_WHOLE_MRI", "")))
    p.add_argument("--spect-root", type=Path, default=Path(os.environ.get("ILCLDM_WHOLE_ABETA", "")))
    p.add_argument(
        "--mri-classifier",
        type=Path,
        default=Path(
            os.environ.get(
                "FST_CLSFEAT_MRI_CKPT",
                str(
                    FST_ROOT
                    / "models/classifiers_roi_4mm/mri/mobilenet_ilcldm_test34/folds/fold3/model.keras"
                ),
            )
        ),
    )
    p.add_argument(
        "--spect-classifier",
        type=Path,
        default=Path(
            os.environ.get(
                "FST_CLSFEAT_SPECT_CKPT",
                str(FST_ROOT / "models/classifiers_roi_4mm/spect/mobilenet/model.keras"),
            )
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("FST_CLSFEAT_OUT_DIR", str(FST_ROOT / f"result/{EXP_ID}"))),
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("FST_CLSFEAT_CACHE", str(FST_ROOT / "data/cache_slices_roi4mm"))),
    )
    p.add_argument("--train-list", type=Path, default=Path(os.environ.get("FST_TRAIN_LIST", "")))
    p.add_argument("--test-list", type=Path, default=Path(os.environ.get("FST_TEST_LIST", "")))
    p.add_argument("--meta-csv", type=Path, default=Path(os.environ.get("FST_META_CSV", "")))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--mse-weight", type=float, default=0.8)
    p.add_argument("--ce-weight", type=float, default=0.2)
    p.add_argument("--n-val-holdout", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--ckpt", type=Path, default=None, help="Transfer weights (.keras) for eval-only")
    return p.parse_args()


def refuse_paths(args) -> None:
    sr = str(args.spect_root).replace("\\", "/")
    if "whole_Abeta_4mm" not in sr:
        raise SystemExit(f"ERROR: expected whole_Abeta_4mm, got {args.spect_root}")
    for path, label in (
        (args.mri_root, "MRI root"),
        (args.spect_root, "SPECT root"),
        (args.mri_classifier, "MRI classifier"),
        (args.spect_classifier, "SPECT classifier"),
        (args.train_list, "train list"),
        (args.test_list, "test list"),
        (args.meta_csv, "meta csv"),
    ):
        p = Path(path)
        if label.endswith("root"):
            if not p.is_dir():
                raise SystemExit(f"ERROR: missing {label}: {p}")
        elif not p.is_file():
            raise SystemExit(f"ERROR: missing {label}: {p}")


def build_manifest(subject_ids: list[str], args, labels: dict[str, int]) -> pd.DataFrame:
    raw = ensure_subject_slices(
        subject_ids,
        mri_root=args.mri_root,
        spect_root=args.spect_root,
        cache_root=args.cache_dir,
        z_min=DEFAULT_SLICE_RANGE[0],
        z_max=DEFAULT_SLICE_RANGE[1],
    )
    return attach_labels(raw, labels)


def subject_level_from_slices(slice_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in slice_df.groupby("subject_id"):
        votes = g["pred_class"].value_counts()
        pred = int(votes.idxmax())
        rows.append(
            {
                "subject_id": str(sid),
                "label": int(g["label"].iloc[0]),
                "pred_class": pred,
                "n_slices": len(g),
                "mean_prob_pd": float(g["prob_pd"].mean()),
            }
        )
    return pd.DataFrame(rows)


def metrics_from_subjects(subj: pd.DataFrame) -> dict:
    y_true = subj["label"].to_numpy(dtype=int)
    y_pred = subj["pred_class"].to_numpy(dtype=int)
    prob = subj["mean_prob_pd"].to_numpy(dtype=float)
    out = {
        "n_subjects": int(len(subj)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
        "per_subject": subj.to_dict(orient="records"),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
    else:
        out["roc_auc"] = float("nan")
    return out


@tf.function
def train_step(model, opt, mri, spect, y, mse_w, ce_w):
    with tf.GradientTape() as tape:
        outs = model([mri, spect], training=True)
        f_hat = outs["f_hat"]
        f_spect = outs["f_spect"]
        logits = outs["predictions"]
        mse = tf.reduce_mean(tf.square(f_hat - f_spect))
        ce = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, logits))
        loss = mse_w * mse + ce_w * ce
    vars_ = model.trainable_weights
    grads = tape.gradient(loss, vars_)
    opt.apply_gradients(zip(grads, vars_))
    return loss, mse, ce


@tf.function
def eval_batch(model, mri, spect, y):
    outs = model([mri, spect], training=False)
    f_hat = outs["f_hat"]
    f_spect = outs["f_spect"]
    logits = outs["predictions"]
    mse = tf.reduce_mean(tf.square(f_hat - f_spect))
    ce = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, logits))
    return mse, ce, logits


def run_epoch_train(model, opt, seq, mse_w, ce_w):
    total_loss = total_mse = total_ce = 0.0
    n = 0
    for (mri, spect), y, _sids in seq:
        mri_t = tf.convert_to_tensor(mri)
        spect_t = tf.convert_to_tensor(spect)
        y_t = tf.convert_to_tensor(y)
        loss, mse, ce = train_step(model, opt, mri_t, spect_t, y_t, mse_w, ce_w)
        bsz = int(mri.shape[0])
        total_loss += float(loss) * bsz
        total_mse += float(mse) * bsz
        total_ce += float(ce) * bsz
        n += bsz
    return total_loss / max(n, 1), total_mse / max(n, 1), total_ce / max(n, 1)


def evaluate_split(model, seq) -> tuple[dict, pd.DataFrame]:
    rows = []
    mse_total = 0.0
    n = 0
    for (mri, spect), y, sids in seq:
        mse, _ce, logits = eval_batch(
            model,
            tf.convert_to_tensor(mri),
            tf.convert_to_tensor(spect),
            tf.convert_to_tensor(y),
        )
        probs = logits.numpy()
        preds = probs.argmax(axis=1)
        labels = y.argmax(axis=1)
        bsz = int(mri.shape[0])
        mse_total += float(mse) * bsz
        n += bsz
        for i, sid in enumerate(sids):
            rows.append(
                {
                    "subject_id": sid,
                    "label": int(labels[i]),
                    "pred_class": int(preds[i]),
                    "prob_pd": float(probs[i, 1]),
                }
            )
    slice_df = pd.DataFrame(rows)
    subj = subject_level_from_slices(slice_df)
    metrics = metrics_from_subjects(subj)
    metrics["mse_feat"] = mse_total / max(n, 1)
    return metrics, slice_df


def assemble_models(args):
    mri_cls = load_classifier(args.mri_classifier)
    spect_cls = load_classifier(args.spect_classifier)
    mri_feat = build_feature_extractor(mri_cls, name="mri_feat")
    spect_feat = build_feature_extractor(spect_cls, name="spect_feat")
    spect_head = build_spect_head(spect_cls, name="spect_head")
    transfer = build_transfer_module()
    train_model = build_fst_train_model(mri_feat, spect_feat, transfer, spect_head)
    infer_model = build_fst_infer_model(mri_feat, transfer, spect_head)
    freeze_all(mri_feat)
    freeze_all(spect_feat)
    freeze_all(spect_head)
    transfer.trainable = True
    return train_model, infer_model, transfer


def smoke(args) -> None:
    refuse_paths(args)
    labels = load_labels(args.meta_csv)
    train_ids = load_id_list(args.train_list)[:2]
    man = build_manifest(train_ids, args, labels)
    seq = make_paired_sequence(man.head(4), batch_size=2, shuffle=False)
    train_model, _infer, transfer = assemble_models(args)
    (mri, spect), y, _ = seq[0]
    outs = train_model([mri, spect], training=False)
    assert outs["f_hat"].shape == (mri.shape[0], 8, 8, 1024), outs["f_hat"].shape
    assert outs["f_spect"].shape == (mri.shape[0], 8, 8, 1024)
    assert outs["predictions"].shape == (mri.shape[0], 2)
    n_train = count_trainable_weights(train_model)
    n_xfer = count_trainable_weights(transfer)
    assert n_train == n_xfer and n_train > 0, (n_train, n_xfer)
    print(f"SMOKE OK shapes f_hat={outs['f_hat'].shape} trainable={n_train}")


def main() -> None:
    args = parse_args()
    refuse_paths(args)
    tf.keras.utils.set_random_seed(args.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_only:
        smoke(args)
        return

    labels = load_labels(args.meta_csv)
    all_train = load_id_list(args.train_list)
    test_ids = load_id_list(args.test_list)

    fit_path = out_dir / "train_fit.txt"
    val_path = out_dir / "val_holdout.txt"
    if fit_path.exists() and val_path.exists():
        fit_ids = load_id_list(fit_path)
        val_ids = load_id_list(val_path)
    else:
        fit_ids, val_ids = stratified_holdout(
            all_train, labels, n_val=args.n_val_holdout, seed=args.seed
        )
        write_id_list(fit_path, fit_ids)
        write_id_list(val_path, val_ids)

    print(f"Subjects fit={len(fit_ids)} val_holdout={len(val_ids)} test={len(test_ids)}")

    train_man = build_manifest(fit_ids, args, labels)
    val_man = build_manifest(val_ids, args, labels)
    test_man = build_manifest(test_ids, args, labels)

    config = {
        "exp_id": EXP_ID,
        "mri_root": str(args.mri_root),
        "spect_root": str(args.spect_root),
        "mri_classifier": str(args.mri_classifier),
        "spect_classifier": str(args.spect_classifier),
        "cache_dir": str(args.cache_dir),
        "n_fit_subjects": len(fit_ids),
        "n_val_holdout": len(val_ids),
        "n_test_subjects": len(test_ids),
        "n_fit_slices": len(train_man),
        "n_val_slices": len(val_man),
        "mse_weight": args.mse_weight,
        "ce_weight": args.ce_weight,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "patience": args.patience,
        "seed": args.seed,
        "slice_range": list(DEFAULT_SLICE_RANGE),
        "early_stop_metric": "val_balanced_accuracy",
        "model": "ProjUNet_pw13_64ch",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    train_model, infer_model, transfer = assemble_models(args)
    best_path = out_dir / "best.keras"

    if args.eval_only:
        ckpt = args.ckpt or best_path
        loaded = tf.keras.models.load_model(str(ckpt), compile=False)
        transfer.set_weights(loaded.get_weights())
        val_seq = make_paired_sequence(val_man, batch_size=args.batch_size, shuffle=False)
        test_seq = make_paired_sequence(test_man, batch_size=args.batch_size, shuffle=False)
        val_metrics, val_pred = evaluate_split(train_model, val_seq)
        test_metrics, test_pred = evaluate_split(train_model, test_seq)
        val_pred.to_csv(out_dir / "val_predictions.csv", index=False)
        test_pred.to_csv(out_dir / "test_predictions.csv", index=False)
        (out_dir / "metrics_valid.json").write_text(json.dumps(val_metrics, indent=2))
        (out_dir / "metrics_test.json").write_text(json.dumps(test_metrics, indent=2))
        print(f"eval val_BA={val_metrics['balanced_accuracy']:.4f} test_BA={test_metrics['balanced_accuracy']:.4f}")
        return

    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    mse_w = tf.constant(args.mse_weight, dtype=tf.float32)
    ce_w = tf.constant(args.ce_weight, dtype=tf.float32)

    history = []
    best_ba = -1.0
    best_epoch = 0
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_seq = make_paired_sequence(
            train_man, batch_size=args.batch_size, shuffle=True, seed=args.seed + epoch
        )
        val_seq = make_paired_sequence(val_man, batch_size=args.batch_size, shuffle=False)
        tr_loss, tr_mse, tr_ce = run_epoch_train(train_model, opt, train_seq, mse_w, ce_w)
        val_metrics, _ = evaluate_split(train_model, val_seq)
        val_ba = float(val_metrics["balanced_accuracy"])
        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_mse": tr_mse,
            "train_ce": tr_ce,
            "val_balanced_accuracy": val_ba,
            "val_roc_auc": val_metrics.get("roc_auc"),
            "val_mse_feat": val_metrics["mse_feat"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs} loss={tr_loss:.4f} "
            f"val_BA={val_ba:.4f} val_mse={val_metrics['mse_feat']:.4f} ({row['seconds']:.1f}s)"
        )

        if val_ba > best_ba + 1e-6:
            best_ba = val_ba
            best_epoch = epoch
            patience_left = args.patience
            transfer.save(str(best_path))
            print(f"New best checkpoint epoch={epoch} val_BA={best_ba:.4f}")
        else:
            patience_left -= 1
            print(f"early-stop counter: {args.patience - patience_left}/{args.patience}")
            if patience_left <= 0:
                print(f"Early stop at epoch {epoch} (best={best_epoch}, val_BA={best_ba:.4f})")
                break

    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)

    loaded = tf.keras.models.load_model(str(best_path), compile=False)
    transfer.set_weights(loaded.get_weights())
    val_seq = make_paired_sequence(val_man, batch_size=args.batch_size, shuffle=False)
    test_seq = make_paired_sequence(test_man, batch_size=args.batch_size, shuffle=False)
    val_metrics, val_pred = evaluate_split(train_model, val_seq)
    test_metrics, test_pred = evaluate_split(train_model, test_seq)
    val_metrics = {"best_epoch": best_epoch, "best_val_balanced_accuracy": best_ba, **val_metrics}
    test_metrics = {"best_epoch": best_epoch, **test_metrics}
    val_pred.to_csv(out_dir / "val_predictions.csv", index=False)
    test_pred.to_csv(out_dir / "test_predictions.csv", index=False)
    (out_dir / "metrics_valid.json").write_text(json.dumps(val_metrics, indent=2))
    (out_dir / "metrics_test.json").write_text(json.dumps(test_metrics, indent=2))
    report = classification_report(
        [r["label"] for r in test_metrics["per_subject"]],
        [r["pred_class"] for r in test_metrics["per_subject"]],
        digits=4,
    )
    (out_dir / "classification_report_test.txt").write_text(report)
    cm = pd.DataFrame(test_metrics["confusion_matrix"])
    cm.to_csv(out_dir / "confusion_matrix_test.csv", index=False)
    print(
        f"Done best_epoch={best_epoch} val_BA={val_metrics['balanced_accuracy']:.4f} "
        f"test_BA={test_metrics['balanced_accuracy']:.4f} -> {out_dir}"
    )


if __name__ == "__main__":
    main()
