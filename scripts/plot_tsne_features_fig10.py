#!/usr/bin/env python3
"""t-SNE of high-level features on the official test set (Fig. 10 style).

Three views of the same subjects/slices:
  1) GAP(f_mri)          — MRI MobileNet features
  2) GAP(UNet_fold5(f_mri)) — translated / synthesized features
  3) GAP(f_spect)        — SPECT reference features

Saves slice-level and subject-level figures under --out-dir.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

FST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FST_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--feat-cache",
        type=Path,
        default=FST_ROOT / "data" / "feat_cache_z78_112",
    )
    p.add_argument(
        "--transfer",
        type=Path,
        default=FST_ROOT
        / "result"
        / "fst_clsfeat_mnet2d_v2_z78_112"
        / "transfer_fold5.keras",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=FST_ROOT / "result" / "fst_clsfeat_mnet2d_v2_z78_112" / "tsne_fig10",
    )
    p.add_argument("--z-min", type=int, default=78)
    p.add_argument("--z-max", type=int, default=112)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--perplexity-slice", type=float, default=30.0)
    p.add_argument("--perplexity-subject", type=float, default=8.0)
    return p.parse_args()


def load_transfer_and_predict(transfer_path: Path, f_mri: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    xfer = tf.keras.models.load_model(str(transfer_path), compile=False)
    outs = []
    bs = 64
    for i in range(0, len(f_mri), bs):
        x = np.asarray(f_mri[i : i + bs], dtype=np.float32)
        y = xfer(x, training=False).numpy()
        outs.append(y)
    return np.concatenate(outs, axis=0)


def gap(maps: np.ndarray) -> np.ndarray:
    """Global average pool over spatial dims → (N, C)."""
    return maps.mean(axis=(1, 2)).astype(np.float32)


def run_tsne(x: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    n = len(x)
    perp = min(perplexity, max(5.0, (n - 1) / 3.0))
    emb = TSNE(
        n_components=2,
        perplexity=perp,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        max_iter=1000,
    ).fit_transform(x)
    return emb


def plot_three_panels(
    emb: np.ndarray,
    labels: np.ndarray,
    n_each: int,
    title: str,
    out_path: Path,
    point_size: float = 18.0,
):
    """emb is stacked [MRI | hat | SPECT], each block length n_each."""
    names = ["MRI features", "Translated (UNet fold 5)", "SPECT reference"]
    colors = {0: "#1f77b4", 1: "#d62728"}
    label_names = {0: "Control", 1: "PD"}

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharex=True, sharey=True)
    xmin, xmax = emb[:, 0].min(), emb[:, 0].max()
    ymin, ymax = emb[:, 1].min(), emb[:, 1].max()
    pad_x = 0.05 * (xmax - xmin + 1e-6)
    pad_y = 0.05 * (ymax - ymin + 1e-6)

    for i, ax in enumerate(axes):
        sl = slice(i * n_each, (i + 1) * n_each)
        xy = emb[sl]
        y = labels
        for c in (0, 1):
            m = y == c
            ax.scatter(
                xy[m, 0],
                xy[m, 1],
                s=point_size,
                c=colors[c],
                alpha=0.75,
                edgecolors="none",
                label=label_names[c],
            )
        ax.set_title(names[i], fontsize=12)
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        if i == 0:
            ax.legend(loc="best", frameon=True, fontsize=9)

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    args = parse_args()
    idx = pd.read_csv(args.feat_cache / "index.csv", dtype={"subject_id": str})
    Fm = np.load(args.feat_cache / "f_mri.npy", mmap_mode="r")
    Gs = np.load(args.feat_cache / "gap_spect.npy")

    keep = (
        (idx["split"] == "test")
        & (idx["slice_idx"] >= args.z_min)
        & (idx["slice_idx"] <= args.z_max)
    )
    rows = np.where(keep.values)[0]
    if len(rows) == 0:
        raise SystemExit("No test rows in feature cache for the requested Z range")

    sub_idx = idx.iloc[rows].reset_index(drop=True)
    labels = sub_idx["label"].to_numpy().astype(int)
    print(
        f"test slices={len(rows)} subjects={sub_idx['subject_id'].nunique()} "
        f"z=[{args.z_min},{args.z_max}] transfer={args.transfer.name}"
    )

    f_mri = np.asarray(Fm[rows], dtype=np.float32)
    print("running UNet fold5 on MRI features…", flush=True)
    f_hat = load_transfer_and_predict(args.transfer, f_mri)

    gap_mri = gap(f_mri)
    gap_hat = gap(f_hat)
    gap_spect = np.asarray(Gs[rows], dtype=np.float32)

    # --- slice-level joint t-SNE ---
    print("t-SNE slice-level…", flush=True)
    x_slice = np.vstack([gap_mri, gap_hat, gap_spect])
    emb_slice = run_tsne(x_slice, args.perplexity_slice, args.seed)
    n = len(gap_mri)
    plot_three_panels(
        emb_slice,
        labels,
        n,
        title="t-SNE of high-level features (test set, slice-level) — fold 5",
        out_path=args.out_dir / "tsne_test_slices.png",
        point_size=10.0,
    )

    # --- subject-level: mean GAP per subject, then joint t-SNE ---
    print("t-SNE subject-level…", flush=True)
    sids = sub_idx["subject_id"].to_numpy()
    uniq = sorted(sub_idx["subject_id"].unique(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    y_subj = np.array([int(sub_idx.loc[sub_idx.subject_id == s, "label"].iloc[0]) for s in uniq])
    gm = np.stack([gap_mri[sids == s].mean(axis=0) for s in uniq])
    gh = np.stack([gap_hat[sids == s].mean(axis=0) for s in uniq])
    gs = np.stack([gap_spect[sids == s].mean(axis=0) for s in uniq])
    x_subj = np.vstack([gm, gh, gs])
    emb_subj = run_tsne(x_subj, args.perplexity_subject, args.seed)
    plot_three_panels(
        emb_subj,
        y_subj,
        len(uniq),
        title="t-SNE of high-level features (test set, subject-level) — fold 5",
        out_path=args.out_dir / "tsne_test_subjects.png",
        point_size=55.0,
    )

    meta = {
        "transfer": str(args.transfer),
        "feat_cache": str(args.feat_cache),
        "z_range": [args.z_min, args.z_max],
        "n_test_slices": int(len(rows)),
        "n_test_subjects": int(len(uniq)),
        "feature": "GAP(conv_pw_13_relu) dim=1024",
        "tsne": {
            "slice_perplexity": args.perplexity_slice,
            "subject_perplexity": args.perplexity_subject,
            "seed": args.seed,
            "joint_embedding": True,
        },
        "panels": ["MRI GAP(f_mri)", "Translated GAP(UNet_fold5(f_mri))", "SPECT GAP(f_spect)"],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "tsne_meta.json").write_text(json.dumps(meta, indent=2))

    # also dump 2D coords for reuse
    pd.DataFrame(
        {
            "source": np.repeat(["mri", "translated", "spect"], n),
            "subject_id": np.tile(sids, 3),
            "label": np.tile(labels, 3),
            "tsne_1": emb_slice[:, 0],
            "tsne_2": emb_slice[:, 1],
        }
    ).to_csv(args.out_dir / "tsne_coords_slices.csv", index=False)
    pd.DataFrame(
        {
            "source": np.repeat(["mri", "translated", "spect"], len(uniq)),
            "subject_id": np.tile(uniq, 3),
            "label": np.tile(y_subj, 3),
            "tsne_1": emb_subj[:, 0],
            "tsne_2": emb_subj[:, 1],
        }
    ).to_csv(args.out_dir / "tsne_coords_subjects.csv", index=False)
    print(f"done → {args.out_dir}")


if __name__ == "__main__":
    main()
