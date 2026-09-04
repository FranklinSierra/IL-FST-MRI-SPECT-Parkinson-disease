#!/usr/bin/env python3
"""Grad-CAM (predicted class) for FST transfer model on a subject test list.

Pipeline: MRI tile → MobileNet_MRI → UNet → SPECT head → softmax.
Target layer: MRI features ``conv_pw_13_relu`` (8×8×1024), which is ``f_mri``.

Selects up to 6 subjects stratified by clinical stage (Control / PD NHY1 / PD NHY2):
  - best correct prediction (highest confidence in the true class)
  - worst incorrect prediction (lowest confidence in the true class)

For each subject, Grad-CAM is computed on the axial slice whose PD probability
is closest to the subject mean (z ∈ [z_min, z_max]).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib import cm
from PIL import Image

FST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FST_ROOT))

DEFAULT_RESULT = FST_ROOT / "result" / "fst_clsfeat_mnet2d_v2_z78_112"
DEFAULT_NHY = FST_ROOT / "data" / "nhy_distribution.csv"
FEATURE_LAYER = "conv_pw_13_relu"
STAGES = ("Control", "NHY1", "NHY2")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, default=5)
    p.add_argument(
        "--feat-cache",
        type=Path,
        default=FST_ROOT / "data" / "feat_cache_z78_112",
    )
    p.add_argument("--transfer", type=Path, default=None)
    p.add_argument("--test-list", type=Path, default=None)
    p.add_argument(
        "--mri-classifier",
        type=Path,
        default=Path(
            os.environ.get(
                "FST_CLSFEAT_MRI_CKPT",
                str(FST_ROOT / "classifiers/classifiers_roi_4mm/mri/model.keras"),
            )
        ),
    )
    p.add_argument(
        "--spect-classifier",
        type=Path,
        default=Path(
            os.environ.get(
                "FST_CLSFEAT_SPECT_CKPT",
                str(FST_ROOT / "classifiers/classifiers_roi_4mm/spect/model.keras"),
            )
        ),
    )
    p.add_argument(
        "--mri-root",
        type=Path,
        default=Path(os.environ.get("ILCLDM_WHOLE_MRI", "/workspace/IL-CLDM/data_rois/whole_MRI")),
    )
    p.add_argument("--nhy-csv", type=Path, default=DEFAULT_NHY)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--metrics-json", type=Path, default=None, help="JSON with test.threshold")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--z-min", type=int, default=78)
    p.add_argument("--z-max", type=int, default=112)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cpu", action="store_true", help="Force TensorFlow on CPU")
    p.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay opacity")
    return p.parse_args()


def resolve_paths(args):
    own = DEFAULT_RESULT / "ownKFold"
    if args.transfer is None:
        if args.fold == 5 and (DEFAULT_RESULT / "transfer_fold5.keras").is_file():
            args.transfer = DEFAULT_RESULT / "transfer_fold5.keras"
        else:
            args.transfer = own / "models" / f"fold{args.fold}" / f"transfer_fold{args.fold}.keras"
    if args.test_list is None:
        if args.fold == 5 and not (own / "files" / "testFold5.txt").exists():
            args.test_list = Path(os.environ.get("FST_TEST_LIST", "/workspace/IL-CLDM/data_info/test.txt"))
        else:
            cand = own / "files" / f"testFold{args.fold}.txt"
            if cand.is_file():
                args.test_list = cand
            else:
                args.test_list = own / "models" / f"fold{args.fold}" / "testFold.txt"
    if args.out_dir is None:
        if args.fold == 5 and args.transfer.parent == DEFAULT_RESULT:
            args.out_dir = DEFAULT_RESULT / "gradcam_fold5"
        else:
            args.out_dir = own / "models" / f"fold{args.fold}" / "gradCam"
    if args.metrics_json is None and args.threshold is None:
        for cand in (
            own / "models" / f"fold{args.fold}" / f"metrics_fold{args.fold}.json",
            DEFAULT_RESULT / "metrics_test.json",
        ):
            if cand.is_file():
                args.metrics_json = cand
                break
    return args


def _ensure_tf_cuda() -> None:
    try:
        sys.path.insert(0, "/workspace/CoCoLIT/src")
        from cocolit.classification.cuda_env import ensure_nvidia_libs_for_tf

        ensure_nvidia_libs_for_tf(reexec=True)
    except Exception as exc:
        print(f"WARN: TF CUDA lib bootstrap skipped: {exc}", flush=True)


def resolve_threshold(args) -> float:
    if args.threshold is not None:
        return float(args.threshold)
    if args.metrics_json and args.metrics_json.is_file():
        data = json.loads(args.metrics_json.read_text())
        if "threshold" in data:
            return float(data["threshold"])
        if "test" in data and "threshold" in data["test"]:
            return float(data["test"]["threshold"])
    return 0.4228927195072174


def load_ids(path: Path) -> set[str]:
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip()}


def stage_of(label: int, nhy) -> str:
    if int(label) == 0:
        return "Control"
    if pd.isna(nhy):
        return "PD?"
    return f"NHY{int(nhy)}"


def load_mri_tile(mri_root: Path, subject_id: str, slice_idx: int) -> np.ndarray:
    from fst.slice_data import map_volume_to_unit_interval, slice_to_rgb

    path = mri_root / f"{subject_id}.nii.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    vol = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
    unit = map_volume_to_unit_interval(vol, "mri")
    sl = unit[:, :, int(slice_idx)]
    return slice_to_rgb(sl)


def compute_subject_probs(
    feat_cache: Path,
    transfer_path: Path,
    spect_ckpt: Path,
    test_ids: set[str],
    z_min: int,
    z_max: int,
    batch_size: int = 8,
):
    """Per-slice and subject-level P(PD) via cached f_mri + transfer + SPECT head."""
    import tensorflow as tf

    from fst.keras_fst_models import build_spect_head, load_classifier

    idx = pd.read_csv(feat_cache / "index.csv", dtype={"subject_id": str})
    te = idx[idx["subject_id"].isin(test_ids)].copy()
    if "slice_idx" in te.columns:
        te = te[(te["slice_idx"] >= z_min) & (te["slice_idx"] <= z_max)]
    if te.empty:
        raise SystemExit("No test rows in feature cache for the requested Z range / test list")

    Fm = np.load(feat_cache / "f_mri.npy", mmap_mode="r")
    rows = te.index.to_numpy()
    f_mri = np.asarray(Fm[rows], dtype=np.float32)

    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    xfer = tf.keras.models.load_model(str(transfer_path), compile=False)
    spect = load_classifier(spect_ckpt)
    head = build_spect_head(spect)

    probs = []
    for i in range(0, len(f_mri), batch_size):
        fh = xfer(f_mri[i : i + batch_size], training=False)
        p = head(fh, training=False).numpy()
        probs.append(p[:, 1])
    te = te.reset_index(drop=True)
    te["prob_pd"] = np.concatenate(probs)

    subj = (
        te.groupby("subject_id", as_index=False)
        .agg(prob=("prob_pd", "mean"), label=("label", "first"))
        .sort_values("subject_id")
    )
    return te, subj


def select_cases(subj: pd.DataFrame, nhy: pd.DataFrame, thr: float) -> pd.DataFrame:
    nhy = nhy.copy()
    nhy["ID"] = nhy["ID"].astype(str)
    m = subj.merge(nhy[["ID", "NHY"]], left_on="subject_id", right_on="ID", how="left")
    m["pred"] = (m["prob"] > thr).astype(int)
    m["stage"] = [stage_of(lab, n) for lab, n in zip(m["label"], m["NHY"])]
    m["correct"] = m["pred"] == m["label"]
    m["conf_true"] = np.where(m["label"] == 1, m["prob"], 1.0 - m["prob"])

    rows = []
    for stage in STAGES:
        for kind, correct in (("best_correct", True), ("worst_incorrect", False)):
            pool = m[(m["stage"] == stage) & (m["correct"] == correct)]
            if pool.empty:
                print(f"WARN: no subjects for {stage} / {kind}", flush=True)
                continue
            if correct:
                pick = pool.nlargest(1, "conf_true").iloc[0]
            else:
                pick = pool.nsmallest(1, "conf_true").iloc[0]
            rows.append(
                {
                    "panel_role": kind,
                    "stage": stage,
                    "subject_id": str(pick["subject_id"]),
                    "label": int(pick["label"]),
                    "NHY": None if pd.isna(pick["NHY"]) else int(pick["NHY"]),
                    "prob_pd_subject": float(pick["prob"]),
                    "pred": int(pick["pred"]),
                    "correct": bool(pick["correct"]),
                    "conf_true": float(pick["conf_true"]),
                    "pred_class_name": "PD" if int(pick["pred"]) == 1 else "Control",
                    "pred_class_idx": int(pick["pred"]),
                }
            )
    return pd.DataFrame(rows)


def attach_representative_slice(cases: pd.DataFrame, slice_df: pd.DataFrame) -> pd.DataFrame:
    out = cases.copy()
    zs, pzs, deltas = [], [], []
    for _, row in out.iterrows():
        sub = slice_df[slice_df["subject_id"] == row["subject_id"]]
        mean_p = float(row["prob_pd_subject"])
        diffs = np.abs(sub["prob_pd"].to_numpy(dtype=float) - mean_p)
        j = int(np.argmin(diffs))
        zs.append(int(sub.iloc[j]["slice_idx"]))
        pzs.append(float(sub.iloc[j]["prob_pd"]))
        deltas.append(float(diffs[j]))
    out["slice_idx"] = zs
    out["prob_pd_slice"] = pzs
    out["abs_diff_to_mean"] = deltas
    return out


def build_gradcam_model(mri_ckpt: Path, transfer, spect_head):
    import tensorflow as tf

    from fst.keras_fst_models import (
        FEATURE_LAYER as FL,
        build_feature_extractor,
        freeze_all,
        load_classifier,
    )

    mri_cls = load_classifier(mri_ckpt)
    mri_feat = build_feature_extractor(mri_cls, name="mri_feat")
    freeze_all(mri_feat)
    freeze_all(transfer)
    freeze_all(spect_head)

    mri_in = tf.keras.Input(shape=(256, 256, 3), name="mri_in")
    f_mri = mri_feat(mri_in)
    f_hat = transfer(f_mri)
    preds = spect_head(f_hat)
    model = tf.keras.Model(
        inputs=mri_in,
        outputs={"f_mri": f_mri, "predictions": preds},
        name="fst_gradcam",
    )
    return model, FL


def gradcam_map(model, tile: np.ndarray, class_idx: int) -> tuple[np.ndarray, np.ndarray]:
    import tensorflow as tf

    x = tf.convert_to_tensor(tile[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        outs = model(x, training=False)
        f_mri = outs["f_mri"]
        preds = outs["predictions"]
        score = preds[:, int(class_idx)]

    grads = tape.gradient(score, f_mri)
    if grads is None:
        raise RuntimeError("Gradients w.r.t. f_mri are None — graph may be disconnected")

    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(weights[:, None, None, :] * f_mri, axis=-1)
    cam = tf.nn.relu(cam)[0].numpy().astype(np.float32)
    cam_max = float(cam.max())
    if cam_max > 0:
        cam = cam / cam_max
    probs = preds[0].numpy().astype(np.float32)
    return cam, probs


def upsample_cam(cam: np.ndarray, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    img = Image.fromarray((np.clip(cam, 0, 1) * 255.0).astype(np.uint8), mode="L")
    return np.asarray(img.resize(size, Image.BILINEAR), dtype=np.float32) / 255.0


def overlay_cam(tile_rgb: np.ndarray, cam_up: np.ndarray, alpha: float) -> np.ndarray:
    base = np.clip(tile_rgb, 0, 1)
    heat = cm.jet(cam_up)[..., :3].astype(np.float32)
    mask = cam_up > 0.05
    out = base.copy()
    out[mask] = (1.0 - alpha) * base[mask] + alpha * heat[mask]
    return np.clip(out, 0, 1)


def save_case_figure(
    out_path: Path,
    tile: np.ndarray,
    overlay: np.ndarray,
    cam_up: np.ndarray,
    meta: dict,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
    axes[0].imshow(tile[..., 0], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("MRI slice")
    axes[1].imshow(cam_up, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    title = (
        f"{meta['panel_role']} | {meta['stage']} | ID {meta['subject_id']} | "
        f"z={meta['slice_idx']} | pred={meta['pred_class_name']} "
        f"(P(PD)={meta['prob_pd_slice']:.3f}, subj={meta['prob_pd_subject']:.3f})"
    )
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_panel(cases: pd.DataFrame, overlays: dict[str, np.ndarray], out_path: Path, fold: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2))
    role_row = {"best_correct": 0, "worst_incorrect": 1}
    row_labels = {0: "Best correct", 1: "Worst incorrect"}
    for _, row in cases.iterrows():
        r = role_row[row["panel_role"]]
        c = STAGES.index(row["stage"])
        ax = axes[r, c]
        key = f"{row['panel_role']}_{row['stage']}_{row['subject_id']}"
        ax.imshow(overlays[key])
        ax.set_title(
            f"{row['stage']}  ID {row['subject_id']}\n"
            f"pred={row['pred_class_name']}  P(PD)={row['prob_pd_subject']:.3f}  z={row['slice_idx']}",
            fontsize=8,
        )
        ax.axis("off")
        if c == 0:
            ax.set_ylabel(row_labels[r], fontsize=10)
    fig.suptitle(
        f"Grad-CAM (predicted class) — FST fold {fold}, MRI → UNet → SPECT head\n"
        "Target: conv_pw_13_relu (f_mri); slice closest to subject-mean P(PD)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    args = resolve_paths(parse_args())
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    _ensure_tf_cuda()

    import tensorflow as tf

    thr = resolve_threshold(args)
    test_ids = load_ids(args.test_list)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"GPU visible: {tf.config.list_physical_devices('GPU')}", flush=True)
    print(f"fold={args.fold} threshold={thr:.6f}  out={args.out_dir}", flush=True)
    print(f"transfer={args.transfer}", flush=True)
    print(f"test_list={args.test_list} (n={len(test_ids)})", flush=True)

    slice_df, subj = compute_subject_probs(
        args.feat_cache,
        args.transfer,
        args.spect_classifier,
        test_ids,
        args.z_min,
        args.z_max,
        args.batch_size,
    )
    tf.keras.backend.clear_session()

    nhy = pd.read_csv(args.nhy_csv, dtype={"ID": str})
    cases = select_cases(subj, nhy, thr)
    cases = attach_representative_slice(cases, slice_df)
    if cases.empty:
        raise SystemExit("No Grad-CAM cases selected")

    cases.to_csv(args.out_dir / "case_selection.csv", index=False)
    meta = {
        "fold": args.fold,
        "transfer": str(args.transfer),
        "test_list": str(args.test_list),
        "mri_classifier": str(args.mri_classifier),
        "spect_classifier": str(args.spect_classifier),
        "threshold": thr,
        "z_range": [args.z_min, args.z_max],
        "target_layer": FEATURE_LAYER,
        "gradcam_signal": "predicted_class",
        "slice_rule": "closest_to_subject_mean_prob_pd",
        "n_cases": int(len(cases)),
        "cases": cases.to_dict(orient="records"),
    }
    (args.out_dir / "case_selection.json").write_text(json.dumps(meta, indent=2))
    print(cases.to_string(index=False), flush=True)

    import tensorflow as tf
    from fst.keras_fst_models import build_spect_head, load_classifier

    xfer = tf.keras.models.load_model(str(args.transfer), compile=False)
    head = build_spect_head(load_classifier(args.spect_classifier))
    model, _ = build_gradcam_model(args.mri_classifier, xfer, head)
    overlays: dict[str, np.ndarray] = {}

    for _, row in cases.iterrows():
        sid = str(row["subject_id"])
        z = int(row["slice_idx"])
        class_idx = int(row["pred_class_idx"])
        tile = load_mri_tile(args.mri_root, sid, z)
        cam, probs = gradcam_map(model, tile, class_idx)
        cam_up = upsample_cam(cam)
        overlay = overlay_cam(tile, cam_up, args.alpha)

        stem = f"{row['panel_role']}_{row['stage']}_{sid}_z{z}"
        key = f"{row['panel_role']}_{row['stage']}_{sid}"
        overlays[key] = overlay

        np.save(args.out_dir / f"{stem}_cam8.npy", cam)
        np.save(args.out_dir / f"{stem}_tile.npy", tile)

        save_case_figure(
            args.out_dir / f"{stem}.png",
            tile,
            overlay,
            cam_up,
            {**row.to_dict(), "softmax": [float(probs[0]), float(probs[1])]},
        )
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        ax.imshow(overlay)
        ax.axis("off")
        fig.savefig(args.out_dir / f"{stem}_overlay.png", dpi=140, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(
            f"  {stem}: softmax={probs.tolist()} GradCAM→class {row['pred_class_name']}",
            flush=True,
        )

    save_panel(cases, overlays, args.out_dir / "panel_2x3.png", args.fold)
    print(f"Wrote panel → {args.out_dir / 'panel_2x3.png'}", flush=True)


if __name__ == "__main__":
    main()
