#!/usr/bin/env python3
"""Cache conv_pw_13 features for a nested fold using fold-specific frozen gates.

Does not use the canonical feat_cache_z78_112 or primary-split checkpoints.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

FST = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FST))

import tensorflow as tf
from fst.keras_fst_models import build_feature_extractor, load_classifier
from fst.slice_data import load_id_list, load_labels, map_volume_to_unit_interval, slice_to_rgb

for g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--mri-ckpt", type=Path, required=True)
    p.add_argument("--spect-ckpt", type=Path, required=True)
    p.add_argument("--train-list", type=Path, required=True)
    p.add_argument("--test-list", type=Path, required=True)
    p.add_argument("--meta-csv", type=Path, default=Path(os.environ.get("FST_META_CSV", "")))
    p.add_argument("--mri-root", type=Path, default=Path(os.environ.get("ILCLDM_WHOLE_MRI", "")))
    p.add_argument("--spect-root", type=Path, default=Path(os.environ.get("ILCLDM_WHOLE_ABETA", "")))
    p.add_argument("--z-min", type=int, default=78)
    p.add_argument("--z-max", type=int, default=112)
    p.add_argument("--z-step", type=int, default=1)
    return p.parse_args()


def main():
    a = parse_args()
    zs = list(range(a.z_min, a.z_max + 1, a.z_step))
    labels = load_labels(a.meta_csv)
    tr = load_id_list(a.train_list)
    te = load_id_list(a.test_list)
    overlap = set(tr) & set(te)
    if overlap:
        raise SystemExit(f"train/test overlap: {sorted(overlap)[:8]}")
    ids = tr + te
    split = ["train"] * len(tr) + ["test"] * len(te)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"z={zs[0]}..{zs[-1]} step={a.z_step} -> {len(zs)} slices x {len(ids)} subjects", flush=True)
    print(f"mri={a.mri_ckpt} spect={a.spect_ckpt}", flush=True)

    mri_cls = load_classifier(a.mri_ckpt)
    sp_cls = load_classifier(a.spect_ckpt)
    mf = build_feature_extractor(mri_cls, "mf")
    sf = build_feature_extractor(sp_cls, "sf")

    n = len(ids) * len(zs)
    fm = np.lib.format.open_memmap(a.out_dir / "f_mri.npy", mode="w+", dtype=np.float16, shape=(n, 8, 8, 1024))
    gs = np.zeros((n, 1024), np.float32)
    rows = []
    i = 0
    for n_sid, (sid, sp) in enumerate(zip(ids, split)):
        mv = map_volume_to_unit_interval(
            np.asarray(nib.load(str(a.mri_root / f"{sid}.nii.gz")).get_fdata(), np.float32), "mri"
        )
        sv = map_volume_to_unit_interval(
            np.asarray(nib.load(str(a.spect_root / f"{sid}.nii.gz")).get_fdata(), np.float32), "spect"
        )
        tm = np.stack([slice_to_rgb(mv[:, :, z]) for z in zs])
        ts = np.stack([slice_to_rgb(sv[:, :, z]) for z in zs])
        fm[i : i + len(zs)] = mf(tm, training=False).numpy().astype(np.float16)
        gs[i : i + len(zs)] = sf(ts, training=False).numpy().mean(axis=(1, 2))
        rows += [{"subject_id": sid, "slice_idx": z, "label": labels[sid], "split": sp} for z in zs]
        i += len(zs)
        if n_sid % 20 == 0:
            print(n_sid, "/", len(ids), flush=True)
    fm.flush()
    np.save(a.out_dir / "gap_spect.npy", gs)
    pd.DataFrame(rows).to_csv(a.out_dir / "index.csv", index=False)
    print("saved", fm.shape, "->", a.out_dir, flush=True)


if __name__ == "__main__":
    main()
