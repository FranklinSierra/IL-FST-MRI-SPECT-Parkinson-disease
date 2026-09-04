#!/usr/bin/env python3
"""Train one NHY-stratified ownKFold FST fold (canonical UNet v2, frozen SPECT head).

Inner val: 15 subjects held out from trainFold, stratified by NHY stratum, seed 42.
Early stopping: subject-level val AUC (patience 12). Threshold: max BA on that val.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
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

FST = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FST))

import tensorflow as tf
from fst.keras_fst_models import build_spect_head, load_classifier
from fst.keras_transfer_unet_v2 import build_transfer_module_v2

OWN = FST / "result" / "fst_clsfeat_mnet2d_v2_z78_112" / "ownKFold"
DEFAULT_SPECT = FST / "classifiers" / "classifiers_roi_4mm" / "spect" / "model.keras"
ENV_SPECT = os.environ.get("FST_CLSFEAT_SPECT_CKPT", "")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, default=None, help="1..5; omit with --aggregate")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--feat-cache", type=Path, default=FST / "data" / "feat_cache_z78_112")
    p.add_argument("--files-dir", type=Path, default=OWN / "files")
    p.add_argument("--models-dir", type=Path, default=OWN / "models")
    p.add_argument("--nhy-csv", type=Path, default=FST / "data" / "nhy_distribution.csv")
    spect_default = Path(ENV_SPECT) if ENV_SPECT and Path(ENV_SPECT).is_file() else DEFAULT_SPECT
    p.add_argument("--spect-classifier", type=Path, default=spect_default)
    p.add_argument("--n-val", type=int, default=15)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--mse-weight", type=float, default=0.3)
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument("--mid-channels", type=int, default=192)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--z-min", type=int, default=78)
    p.add_argument("--z-max", type=int, default=112)
    p.add_argument("--gpu-memory-mb", type=int, default=0, help="hard cap per process; 0 = growth")
    p.add_argument("--fit-list", type=Path, default=None, help="reuse train_fit IDs instead of recomputing holdout")
    p.add_argument("--val-list", type=Path, default=None, help="reuse val_holdout IDs instead of recomputing holdout")
    return p.parse_args()


def configure_gpu(memory_mb: int) -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("WARNING: no GPU visible", flush=True)
        return
    try:
        if memory_mb and memory_mb > 0:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=int(memory_mb))],
            )
            print(f"GPU memory_limit={memory_mb} MB on {gpus[0].name}", flush=True)
        else:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            print(f"GPU memory growth on {len(gpus)} device(s)", flush=True)
    except Exception as e:
        print(f"WARNING: GPU config failed: {e}", flush=True)


def load_ids(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().split() if l.strip()]


def load_strata(nhy_csv: Path) -> dict[str, str]:
    df = pd.read_csv(nhy_csv, dtype={"ID": str})
    df["stratum"] = df["stratum"].fillna("control").astype(str)
    return dict(zip(df["ID"], df["stratum"]))


def stratified_holdout_by_stratum(
    ids: list[str],
    strata: dict[str, str],
    n_val: int = 15,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    by: dict[str, list[str]] = {}
    for sid in ids:
        by.setdefault(strata[sid], []).append(sid)
    classes = sorted(by.keys())
    n = max(len(ids), 1)
    raw = [n_val * (len(by[c]) / n) for c in classes]
    takes = [max(1, int(round(x))) for x in raw]
    takes = [min(t, len(by[c])) for t, c in zip(takes, classes)]

    def room(j: int) -> int:
        return len(by[classes[j]]) - takes[j]

    while sum(takes) > n_val:
        cand = [j for j in range(len(takes)) if takes[j] > 1]
        if not cand:
            break
        j = max(cand, key=lambda i: (takes[i], raw[i]))
        takes[j] -= 1
    while sum(takes) < n_val:
        cand = [j for j in range(len(takes)) if room(j) > 0]
        if not cand:
            break
        j = max(cand, key=lambda i: (room(i), raw[i]))
        takes[j] += 1

    val: list[str] = []
    for c, take in zip(classes, takes):
        pool = sorted(by[c])
        take = min(int(take), len(pool))
        val.extend(str(x) for x in rng.choice(pool, size=take, replace=False).tolist())
    val_set = set(val)
    fit = [s for s in ids if s not in val_set]
    if len(val) < 2 or len(fit) < 2:
        raise ValueError(f"Bad NHY holdout: fit={len(fit)} val={len(val)} takes={dict(zip(classes, takes))}")
    return fit, sorted(val)


def load_cache(cache: Path, train_ids, test_ids, z_min, z_max):
    idx = pd.read_csv(cache / "index.csv", dtype={"subject_id": str})
    fm = np.load(cache / "f_mri.npy", mmap_mode="r")
    gs = np.load(cache / "gap_spect.npy")
    keep = np.ones(len(idx), bool)
    if z_min is not None:
        keep &= idx.slice_idx.values >= z_min
    if z_max is not None:
        keep &= idx.slice_idx.values <= z_max
    tr = idx.subject_id.isin(set(train_ids)).values & keep
    te = idx.subject_id.isin(set(test_ids)).values & keep
    missing_tr = set(train_ids) - set(idx.loc[keep, "subject_id"])
    missing_te = set(test_ids) - set(idx.loc[keep, "subject_id"])
    if missing_tr or missing_te:
        raise SystemExit(f"IDs missing from feature cache train={sorted(missing_tr)} test={sorted(missing_te)}")
    return idx, fm, gs, np.where(tr)[0], np.where(te)[0]


def rows_for(idx, pool_rows, ids):
    return pool_rows[np.isin(idx.subject_id.values[pool_rows], list(ids))]


def build(args, spect_head):
    xfer = build_transfer_module_v2(
        mid_channels=args.mid_channels, l2=args.l2, dropout=args.dropout
    )
    inp = tf.keras.Input(shape=(8, 8, 1024), name="f_mri")
    f_hat = xfer(inp)
    gap = tf.keras.layers.GlobalAveragePooling2D(name="gap_hat")(f_hat)
    logits = spect_head(f_hat)
    return tf.keras.Model(inp, {"gap": gap, "logits": logits}, name="fst_v2"), xfer


def make_train_step(model, opt, mse_w, ce_w, var):
    opt.build(model.trainable_weights)

    @tf.function
    def step(x, gt_gap, y, w):
        with tf.GradientTape() as t:
            o = model(x, training=True)
            mse = tf.reduce_mean(tf.square(o["gap"] - gt_gap)) / var
            ce = tf.reduce_mean(w * tf.keras.losses.categorical_crossentropy(y, o["logits"]))
            loss = mse_w * mse + ce_w * ce
            if model.losses:
                loss = loss + tf.add_n(model.losses)
        gr = t.gradient(loss, model.trainable_weights)
        opt.apply_gradients(zip(gr, model.trainable_weights))
        return loss, mse, ce

    return step


@tf.function
def _fwd(model, x):
    return model(x, training=False)


def predict_rows(model, fm, rows, bs=64):
    ps, gs = [], []
    for i in range(0, len(rows), bs):
        x = np.asarray(fm[rows[i : i + bs]], dtype=np.float32)
        o = _fwd(model, x)
        ps.append(o["logits"].numpy()[:, 1])
        gs.append(o["gap"].numpy())
    return np.concatenate(ps), np.concatenate(gs)


def subj_prob(idx, rows, p):
    d = pd.DataFrame({"s": idx.subject_id.values[rows], "y": idx.label.values[rows], "p": p})
    g = d.groupby("s").agg(y=("y", "first"), p=("p", "mean"))
    return g.index.to_numpy(), g.y.to_numpy(), g.p.to_numpy()


def best_threshold(y, p):
    cand = np.unique(np.concatenate([p, [0.5]]))
    return float(max(cand, key=lambda t: balanced_accuracy_score(y, (p > t).astype(int))))


def metrics(y, p, thr):
    pred = (p > thr).astype(int)
    return {
        "n_subjects": int(len(y)),
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def write_id_list(path: Path, ids: list[str]) -> None:
    path.write_text("\n".join(ids) + ("\n" if ids else ""))


def train_fold(a):
    k = a.fold
    tf.keras.utils.set_random_seed(a.seed)
    out = a.models_dir / f"fold{k}"
    out.mkdir(parents=True, exist_ok=True)

    train_list = a.files_dir / f"trainFold{k}.txt"
    test_list = a.files_dir / f"testFold{k}.txt"
    train_ids = load_ids(train_list)
    test_ids = load_ids(test_list)
    strata = load_strata(a.nhy_csv)
    missing = [s for s in train_ids + test_ids if s not in strata]
    if missing:
        raise SystemExit(f"fold {k}: IDs not in NHY csv: {missing[:12]}")

    if a.fit_list and a.val_list:
        fit_ids = load_ids(a.fit_list)
        val_ids = load_ids(a.val_list)
        extra = (set(fit_ids) | set(val_ids)) - set(train_ids)
        missing_split = set(train_ids) - (set(fit_ids) | set(val_ids))
        if extra or missing_split:
            raise SystemExit(
                f"fold {k}: fit/val must partition trainFold extra={sorted(extra)[:8]} "
                f"missing={sorted(missing_split)[:8]}"
            )
        if set(fit_ids) & set(val_ids):
            raise SystemExit(f"fold {k}: fit/val overlap")
        if set(val_ids) & set(test_ids) or set(fit_ids) & set(test_ids):
            raise SystemExit(f"fold {k}: inner split overlaps testFold")
        print(f"fold {k}: reusing fit/val lists {a.fit_list} {a.val_list}", flush=True)
    else:
        fit_ids, val_ids = stratified_holdout_by_stratum(train_ids, strata, a.n_val, a.seed)
    write_id_list(out / f"trainFold.txt", train_ids)
    write_id_list(out / f"testFold.txt", test_ids)
    write_id_list(out / "train_fit.txt", fit_ids)
    write_id_list(out / "val_holdout.txt", val_ids)
    shutil.copy2(train_list, out / f"trainFold{k}.txt")
    shutil.copy2(test_list, out / f"testFold{k}.txt")

    print(
        f"fold {k}: train={len(train_ids)} fit={len(fit_ids)} val={len(val_ids)} test={len(test_ids)}",
        flush=True,
    )
    print(f"  val strata={dict(Counter(strata[i] for i in val_ids))}", flush=True)
    print(f"  test strata={dict(Counter(strata[i] for i in test_ids))}", flush=True)

    idx, fm, gs, tr_rows, te_rows = load_cache(
        a.feat_cache, train_ids, test_ids, a.z_min, a.z_max
    )
    rf = rows_for(idx, tr_rows, fit_ids)
    rv = rows_for(idx, tr_rows, val_ids)
    print(f"  slices fit={len(rf)} val={len(rv)} test={len(te_rows)}", flush=True)

    y_tr = idx.iloc[tr_rows].groupby("subject_id")["label"].first().to_numpy()
    var = float(gs[tr_rows].var())
    cw = {0: len(y_tr) / (2 * np.sum(y_tr == 0)), 1: len(y_tr) / (2 * np.sum(y_tr == 1))}
    print(f"  var(GAP f_spect)={var:.4f} class_weights={cw}", flush=True)

    spect_cls = load_classifier(a.spect_classifier)
    head = build_spect_head(spect_cls)
    model, xfer = build(a, head)
    opt = tf.keras.optimizers.Adam(a.lr)
    step = make_train_step(
        model,
        opt,
        tf.constant(a.mse_weight, tf.float32),
        tf.constant(a.ce_weight, tf.float32),
        tf.constant(var, tf.float32),
    )
    yoh = tf.keras.utils.to_categorical(idx.label.values, 2).astype(np.float32)
    wvec = np.array([cw[int(l)] for l in idx.label.values], np.float32)

    best_auc = -1.0
    best_w = None
    best_ep = 0
    left = a.patience
    hist = []
    rng = np.random.default_rng(a.seed + k)
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        perm = rng.permutation(rf)
        tot = [0.0, 0.0, 0.0]
        n = 0
        for i in range(0, len(perm), a.batch_size):
            b = perm[i : i + a.batch_size]
            l, m, c = step(np.asarray(fm[b], dtype=np.float32), gs[b], yoh[b], wvec[b])
            tot[0] += float(l) * len(b)
            tot[1] += float(m) * len(b)
            tot[2] += float(c) * len(b)
            n += len(b)
        pv, gv = predict_rows(model, fm, rv)
        _, yv, pvs = subj_prob(idx, rv, pv)
        auc = roc_auc_score(yv, pvs) if len(np.unique(yv)) > 1 else float("nan")
        mrel = float(((gv - gs[rv]) ** 2).mean() / var)
        hist.append(
            {
                "fold": k,
                "epoch": ep,
                "loss": tot[0] / n,
                "mse_rel": tot[1] / n,
                "ce": tot[2] / n,
                "val_auc": auc,
                "val_mse_rel": mrel,
                "sec": time.time() - t0,
            }
        )
        print(
            f"fold {k} ep {ep:02d}: loss={tot[0]/n:.4f} mse_rel={tot[1]/n:.4f} "
            f"ce={tot[2]/n:.4f} val_AUC={auc:.4f} val_mse_rel={mrel:.4f} "
            f"({time.time()-t0:.1f}s)",
            flush=True,
        )
        if np.isfinite(auc) and auc > best_auc + 1e-6:
            best_auc = auc
            best_w = [w.numpy() for w in model.weights]
            best_ep = ep
            left = a.patience
        else:
            left -= 1
            if left <= 0:
                break

    model.set_weights(best_w)
    print(f"fold {k}: best_epoch={best_ep} val_AUC={best_auc:.4f}", flush=True)

    pv, _ = predict_rows(model, fm, rv)
    sv, yv, pvs = subj_prob(idx, rv, pv)
    thr = best_threshold(yv, pvs)
    val_m = metrics(yv, pvs, thr)

    pt, gt = predict_rows(model, fm, te_rows)
    st, yt, pts = subj_prob(idx, te_rows, pt)
    te_m = metrics(yt, pts, thr)
    te_m["mse_rel_test"] = float(((gt - gs[te_rows]) ** 2).mean() / var)
    pred = (pts > thr).astype(int)

    ckpt = out / f"transfer_fold{k}.keras"
    xfer.save(ckpt)

    payload = {
        "fold": k,
        "best_epoch": best_ep,
        "val_auc_early_stop": float(best_auc),
        "threshold_source": "inner_val_max_BA",
        "n_train": len(train_ids),
        "n_fit": len(fit_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "val_strata": dict(Counter(strata[i] for i in val_ids)),
        "test_strata": dict(Counter(strata[i] for i in st)),
        "var_gap_spect_trainfold": var,
        "class_weights_trainfold": {str(k_): float(v) for k_, v in cw.items()},
        "val": val_m,
        "test": te_m,
    }
    (out / f"metrics_fold{k}.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame(
        {
            "subject_id": st,
            "label": yt,
            "prob": pts,
            "pred": pred,
            "fold": k,
            "threshold": thr,
            "stratum": [strata.get(str(s), "") for s in st],
        }
    ).to_csv(out / f"predictions_fold{k}.csv", index=False)
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    (out / f"classification_report_fold{k}.txt").write_text(
        classification_report(yt, pred, digits=4, zero_division=0)
    )
    cfg = dict(vars(a))
    cfg["best_epoch"] = best_ep
    cfg["threshold"] = thr
    (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    print(f"fold {k} TEST: {json.dumps(te_m, indent=2)}", flush=True)
    print(f"saved {ckpt}", flush=True)


def aggregate(a):
    rows = []
    for k in range(1, 6):
        p = a.models_dir / f"fold{k}" / f"predictions_fold{k}.csv"
        m = a.models_dir / f"fold{k}" / f"metrics_fold{k}.json"
        if not p.is_file() or not m.is_file():
            raise SystemExit(f"missing fold {k} artifacts: {p} / {m}")
        df = pd.read_csv(p, dtype={"subject_id": str})
        df["fold"] = k
        rows.append(df)
    oof = pd.concat(rows, ignore_index=True)
    if oof.subject_id.duplicated().any():
        raise SystemExit("duplicate subject in OOF predictions")
    y = oof.label.to_numpy()
    p = oof.prob.to_numpy()
    pred = oof.pred.to_numpy()
    overall = {
        "n_subjects": int(len(oof)),
        "threshold_policy": "per_fold_inner_val_max_BA",
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "per_fold": [],
    }
    for k in range(1, 6):
        met = json.loads((a.models_dir / f"fold{k}" / f"metrics_fold{k}.json").read_text())
        overall["per_fold"].append(
            {
                "fold": k,
                "best_epoch": met["best_epoch"],
                "threshold": met["test"]["threshold"],
                "val_auc": met["val_auc_early_stop"],
                "val_ba": met["val"]["balanced_accuracy"],
                "test_auc": met["test"]["roc_auc"],
                "test_ba": met["test"]["balanced_accuracy"],
                "test_acc": met["test"]["accuracy"],
                "n_test": met["n_test"],
                "test_strata": met["test_strata"],
            }
        )
    overall["test_ba_mean"] = float(np.mean([f["test_ba"] for f in overall["per_fold"]]))
    overall["test_auc_mean"] = float(np.mean([f["test_auc"] for f in overall["per_fold"]]))
    a.models_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(a.models_dir / "oof_predictions.csv", index=False)
    (a.models_dir / "metrics_cv.json").write_text(json.dumps(overall, indent=2))
    print(json.dumps({k: v for k, v in overall.items() if k != "per_fold"}, indent=2), flush=True)


def main():
    a = parse_args()
    if a.aggregate:
        aggregate(a)
        return
    if a.fold is None or a.fold < 1:
        raise SystemExit("pass --fold K (1..5) or --aggregate")
    configure_gpu(a.gpu_memory_mb)
    train_fold(a)


if __name__ == "__main__":
    main()
