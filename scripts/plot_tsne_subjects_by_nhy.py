#!/usr/bin/env python3
"""Fig.10-style extras: t-SNE colored by NHY + distance to control centroid vs NHY.

Compares three GAP(conv_pw_13_relu) spaces on the official test set:
  1) GAP(f_mri)
  2) GAP(transfer_fold5(f_mri))
  3) GAP(f_spect)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

FST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FST_ROOT))

DEFAULT_OUT = FST_ROOT / "result" / "fst_clsfeat_mnet2d_v2_z78_112" / "tsne_fig10"
DEFAULT_NHY = Path("/workspace/IL-CLDM/data_info/splits/nhy_v1/nhy_distribution.csv")
DEFAULT_CACHE = FST_ROOT / "data" / "feat_cache_z78_112"
DEFAULT_TRANSFER = (
    FST_ROOT / "result" / "fst_clsfeat_mnet2d_v2_z78_112" / "transfer_fold5.keras"
)

SPACE_ORDER = ("mri", "translated", "spect")
SPACE_LABELS = {
    "mri": "MRI GAP(f_mri)",
    "translated": "Translated GAP(UNet fold 5)",
    "spect": "SPECT GAP(f_spect)",
}
SPACE_YLABEL = {
    "mri": "Distance to control centroid (MRI GAP)",
    "translated": "Distance to control centroid (translated GAP)",
    "spect": "Distance to control centroid (SPECT GAP)",
}

STAGE_STYLE = {
    "Control": {"color": "#1f77b4", "marker": "o", "zorder": 2},
    "PD NHY1": {"color": "#ff7f0e", "marker": "^", "zorder": 3},
    "PD NHY2": {"color": "#d62728", "marker": "s", "zorder": 3},
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--nhy-csv", type=Path, default=DEFAULT_NHY)
    p.add_argument("--feat-cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--transfer", type=Path, default=DEFAULT_TRANSFER)
    p.add_argument("--z-min", type=int, default=78)
    p.add_argument("--z-max", type=int, default=112)
    return p.parse_args()


def stage_label(row) -> str:
    if int(row["label"]) == 0:
        return "Control"
    nhy = row["NHY"]
    if pd.isna(nhy):
        return "PD (NHY?)"
    return f"PD NHY{int(nhy)}"


def gap(maps: np.ndarray) -> np.ndarray:
    return maps.mean(axis=(1, 2)).astype(np.float32)


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


def load_subject_gap_features(
    feat_cache: Path,
    transfer_path: Path,
    z_min: int,
    z_max: int,
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    idx = pd.read_csv(feat_cache / "index.csv", dtype={"subject_id": str})
    Fm = np.load(feat_cache / "f_mri.npy", mmap_mode="r")
    Gs = np.load(feat_cache / "gap_spect.npy")
    keep = (
        (idx["split"] == "test")
        & (idx["slice_idx"] >= z_min)
        & (idx["slice_idx"] <= z_max)
    )
    rows = np.where(keep.values)[0]
    if len(rows) == 0:
        raise SystemExit("No test rows in feature cache for the requested Z range")

    sub = idx.iloc[rows].reset_index(drop=True)
    f_mri = np.asarray(Fm[rows], dtype=np.float32)
    print("running UNet on MRI features for centroid analysis…", flush=True)
    f_hat = load_transfer_and_predict(transfer_path, f_mri)

    gap_mri = gap(f_mri)
    gap_hat = gap(f_hat)
    gap_spect = np.asarray(Gs[rows], dtype=np.float32)

    uniq = sorted(
        sub["subject_id"].unique(),
        key=lambda x: int(x) if str(x).isdigit() else str(x),
    )
    labels = np.array(
        [int(sub.loc[sub.subject_id == s, "label"].iloc[0]) for s in uniq]
    )
    feats = {
        "mri": np.stack([gap_mri[sub["subject_id"].to_numpy() == s].mean(axis=0) for s in uniq]),
        "translated": np.stack([gap_hat[sub["subject_id"].to_numpy() == s].mean(axis=0) for s in uniq]),
        "spect": np.stack([gap_spect[sub["subject_id"].to_numpy() == s].mean(axis=0) for s in uniq]),
    }
    return uniq, labels, feats


def compute_distances(feats: np.ndarray, labels: np.ndarray) -> np.ndarray:
    centroid = feats[labels == 0].mean(axis=0)
    return np.linalg.norm(feats - centroid[None, :], axis=1)


def summarize_space(
    space: str,
    uniq: list[str],
    labels: np.ndarray,
    dist: np.ndarray,
    nhy_map: dict[str, float],
) -> tuple[pd.DataFrame, dict]:
    rows_out = []
    for i, sid in enumerate(uniq):
        rows_out.append(
            {
                "subject_id": sid,
                "label": int(labels[i]),
                "NHY": nhy_map.get(sid, np.nan),
                "stage": "Control"
                if labels[i] == 0
                else (
                    f"PD NHY{int(nhy_map[sid])}"
                    if sid in nhy_map and pd.notna(nhy_map[sid])
                    else "PD"
                ),
                "dist_to_control_centroid": float(dist[i]),
            }
        )
    df = pd.DataFrame(rows_out)
    pd_df = df[df["label"] == 1].dropna(subset=["NHY"])
    x = pd_df["NHY"].to_numpy(dtype=float)
    y = pd_df["dist_to_control_centroid"].to_numpy(dtype=float)
    pearson = stats.pearsonr(x, y) if len(x) > 1 else (float("nan"), float("nan"))
    spearman = stats.spearmanr(x, y) if len(x) > 1 else (float("nan"), float("nan"))
    d1 = y[x == 1.0]
    d2 = y[x == 2.0]
    if len(d1) and len(d2):
        mw = stats.mannwhitneyu(d1, d2, alternative="two-sided")
        mw_stat, mw_p = float(mw.statistic), float(mw.pvalue)
    else:
        mw_stat, mw_p = float("nan"), float("nan")

    summary = {
        "space": f"GAP({space}) subject-mean, Euclidean distance to control centroid in-space",
        "n_control": int((labels == 0).sum()),
        "n_pd_with_nhy": int(len(pd_df)),
        "nhy_counts": {str(int(k)): int(v) for k, v in pd_df["NHY"].value_counts().items()},
        "mean_dist_control": float(df.loc[df.label == 0, "dist_to_control_centroid"].mean()),
        "mean_dist_pd_nhy1": float(d1.mean()) if len(d1) else None,
        "mean_dist_pd_nhy2": float(d2.mean()) if len(d2) else None,
        "pearson_r": float(pearson[0]),
        "pearson_p": float(pearson[1]),
        "spearman_rho": float(spearman[0]),
        "spearman_p": float(spearman[1]),
        "mannwhitney_nhy1_vs_nhy2_U": mw_stat,
        "mannwhitney_nhy1_vs_nhy2_p": mw_p,
    }
    return df, summary


def plot_strip_panel(ax, df: pd.DataFrame, pd_df: pd.DataFrame, title: str, ylabel: str) -> None:
    rng = np.random.default_rng(42)
    ctrl = df[df["label"] == 0]
    ax.scatter(
        np.zeros(len(ctrl)) + rng.normal(0, 0.03, len(ctrl)),
        ctrl["dist_to_control_centroid"],
        c=STAGE_STYLE["Control"]["color"],
        marker="o",
        s=45,
        alpha=0.85,
        label="Control",
        edgecolors="white",
        linewidths=0.3,
    )
    for stage, nhy_val, style_key in (
        ("PD NHY1", 1.0, "PD NHY1"),
        ("PD NHY2", 2.0, "PD NHY2"),
    ):
        sub = pd_df[pd_df["NHY"] == nhy_val]
        ax.scatter(
            np.full(len(sub), nhy_val) + rng.normal(0, 0.03, len(sub)),
            sub["dist_to_control_centroid"],
            c=STAGE_STYLE[style_key]["color"],
            marker=STAGE_STYLE[style_key]["marker"],
            s=60,
            alpha=0.9,
            label=stage,
            edgecolors="white",
            linewidths=0.3,
        )
    for xv, group in ((0.0, ctrl), (1.0, pd_df[pd_df.NHY == 1]), (2.0, pd_df[pd_df.NHY == 2])):
        if len(group) == 0:
            continue
        mu = group["dist_to_control_centroid"].mean()
        ax.hlines(mu, xv - 0.15, xv + 0.15, colors="black", linewidths=2, zorder=4)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Control", "NHY1", "NHY2"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, axis="y", alpha=0.3)


def plot_pd_only_panel(ax, pd_df: pd.DataFrame, summary: dict) -> None:
    rng = np.random.default_rng(42)
    x = pd_df["NHY"].to_numpy(dtype=float)
    y = pd_df["dist_to_control_centroid"].to_numpy(dtype=float)
    ax.scatter(
        x + rng.normal(0, 0.02, len(x)),
        y,
        c=[STAGE_STYLE["PD NHY1"]["color"] if v == 1 else STAGE_STYLE["PD NHY2"]["color"] for v in x],
        s=70,
        edgecolors="white",
        linewidths=0.3,
    )
    if len(x) > 1:
        coef = np.polyfit(x, y, 1)
        xs = np.array([1.0, 2.0])
        ax.plot(xs, np.polyval(coef, xs), "k--", lw=1.5, label="linear fit (PD only)")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["NHY1", "NHY2"])
    ax.set_xlabel("Hoehn–Yahr stage")
    ax.set_ylabel("Distance to control centroid")
    ax.set_title(
        f"PD only — Spearman ρ={summary['spearman_rho']:.3f} (p={summary['spearman_p']:.3f})"
    )
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)


def plot_all_spaces_distance_vs_nhy(
    space_dfs: dict[str, pd.DataFrame],
    summaries: dict[str, dict],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))
    for col, space in enumerate(SPACE_ORDER):
        df = space_dfs[space]
        pd_df = df[df["label"] == 1].dropna(subset=["NHY"])
        plot_strip_panel(
            axes[0, col],
            df,
            pd_df,
            SPACE_LABELS[space],
            SPACE_YLABEL[space],
        )
        plot_pd_only_panel(axes[1, col], pd_df, summaries[space])

    fig.suptitle(
        "Distance to control centroid vs clinical stage (NHY) — MRI / translated / SPECT",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    fig_path = out_dir / "dist_vs_nhy_all_spaces.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fig_path}")


def plot_tsne_by_nhy(coords: pd.DataFrame, out_path: Path) -> None:
    panel_order = ["mri", "translated", "spect"]
    panel_titles = {
        "mri": "MRI features",
        "translated": "Translated (UNet fold 5)",
        "spect": "SPECT reference",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharex=True, sharey=True)
    xmin, xmax = coords["tsne_1"].min(), coords["tsne_1"].max()
    ymin, ymax = coords["tsne_2"].min(), coords["tsne_2"].max()
    pad_x = 0.05 * (xmax - xmin + 1e-6)
    pad_y = 0.05 * (ymax - ymin + 1e-6)

    for ax, src in zip(axes, panel_order):
        sub = coords[coords["source"] == src]
        for stage, style in STAGE_STYLE.items():
            m = sub["stage"] == stage
            if not m.any():
                continue
            ax.scatter(
                sub.loc[m, "tsne_1"],
                sub.loc[m, "tsne_2"],
                s=70,
                c=style["color"],
                marker=style["marker"],
                alpha=0.9,
                edgecolors="white",
                linewidths=0.4,
                label=stage,
                zorder=style["zorder"],
            )
        ax.set_title(panel_titles[src], fontsize=12)
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        if src == "mri":
            ax.legend(loc="best", frameon=True, fontsize=9)

    fig.suptitle(
        "t-SNE subject-level — PD colored by Hoehn–Yahr (NHY)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def centroid_analysis_all_spaces(
    feat_cache: Path,
    transfer_path: Path,
    nhy: pd.DataFrame,
    z_min: int,
    z_max: int,
    out_dir: Path,
) -> dict[str, dict]:
    uniq, labels, feats = load_subject_gap_features(feat_cache, transfer_path, z_min, z_max)
    nhy_map = dict(zip(nhy["ID"].astype(str), nhy["NHY"]))

    space_dfs: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict] = {}
    dist_cols: dict[str, np.ndarray] = {}
    for space in SPACE_ORDER:
        dist = compute_distances(feats[space], labels)
        dist_cols[space] = dist
        df, summary = summarize_space(space, uniq, labels, dist, nhy_map)
        space_dfs[space] = df
        summaries[space] = summary

    combined = space_dfs["mri"][["subject_id", "label", "NHY", "stage"]].copy()
    for space in SPACE_ORDER:
        combined[f"dist_{space}"] = dist_cols[space]
    combined.to_csv(out_dir / "dist_to_control_centroid_all_spaces.csv", index=False)

    # Backward-compatible SPECT-only exports
    space_dfs["spect"].to_csv(out_dir / "spect_dist_to_control_centroid.csv", index=False)
    summaries["spect"]["interpretation_hint"] = (
        "If early PD (NHY1) sits closer to controls, expect smaller mean distance "
        "for NHY1 than NHY2 and positive correlation distance~NHY."
    )
    (out_dir / "spect_nhy_distance_stats.json").write_text(
        json.dumps(summaries["spect"], indent=2)
    )

    all_summary = {
        "spaces": summaries,
        "interpretation_hint": (
            "Compare whether translated GAP(UNet(f_mri)) preserves control-vs-PD "
            "separation and NHY ordering seen in SPECT GAP."
        ),
    }
    (out_dir / "nhy_distance_stats_all_spaces.json").write_text(
        json.dumps(all_summary, indent=2)
    )

    plot_all_spaces_distance_vs_nhy(space_dfs, summaries, out_dir)

    # Legacy single-panel SPECT figure (regenerated for consistency)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    pd_df = space_dfs["spect"][space_dfs["spect"]["label"] == 1].dropna(subset=["NHY"])
    plot_strip_panel(
        axes[0],
        space_dfs["spect"],
        pd_df,
        "SPECT feature space",
        "Distance to control centroid (SPECT GAP)",
    )
    plot_pd_only_panel(axes[1], pd_df, summaries["spect"])
    fig.suptitle(
        "Distance to SPECT control centroid vs clinical stage (NHY)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig_path = out_dir / "spect_dist_vs_nhy.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fig_path}")

    return summaries


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    coords = pd.read_csv(args.out_dir / "tsne_coords_subjects.csv", dtype={"subject_id": str})
    nhy = pd.read_csv(args.nhy_csv)
    nhy["ID"] = nhy["ID"].astype(str)

    coords = coords.merge(nhy[["ID", "NHY"]], left_on="subject_id", right_on="ID", how="left")
    coords["stage"] = coords.apply(stage_label, axis=1)
    coords.to_csv(args.out_dir / "tsne_coords_subjects_with_nhy.csv", index=False)

    plot_tsne_by_nhy(coords, args.out_dir / "tsne_test_subjects_by_nhy.png")

    summaries = centroid_analysis_all_spaces(
        args.feat_cache,
        args.transfer,
        nhy,
        args.z_min,
        args.z_max,
        args.out_dir,
    )
    print(json.dumps({"spaces": summaries}, indent=2))


if __name__ == "__main__":
    main()
