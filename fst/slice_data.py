"""Axial ROI slice extraction and paired MRI/SPECT Keras sequences (CoCoLIT-compatible)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image

DEFAULT_SLICE_AXIS = 2
DEFAULT_SLICE_RANGE = (-1, 15)
DEFAULT_TILE_SIZE = (256, 256)
MODALITIES = ("mri", "spect")
Modality = Literal["mri", "spect"]

PREPROCESS_SPECS: dict[str, str] = {
    "mri": "volume_zscore_clip3sigma_to_unit",
    "spect": "volume_p1_p99_to_unit_no_zscore",
}

COCOLIT_CACHE_ROOTS = (
    Path("/workspace/CoCoLIT/results/exp_cocolit_data_rois_4mm/classifiers/cache"),
    Path("/workspace/CoCoLIT/results/exp_cocolit_data_rois/classifiers/cache"),
)
TMP_TILE_ROOT = Path("/tmp/fst_clsfeat_tiles")


def slice_indices(z_min: int, z_max: int, volume_shape: tuple[int, ...], axis: int) -> list[int]:
    depth = volume_shape[axis]
    lo = max(0, z_min)
    hi = min(depth - 1, z_max)
    if lo > hi:
        return []
    return list(range(lo, hi + 1))


def _take_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    slc = [slice(None)] * volume.ndim
    slc[axis] = index
    return volume[tuple(slc)]


def _finite_voxels(volume: np.ndarray) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float32)
    return arr[np.isfinite(arr)]


def normalize_volume_mri(volume: np.ndarray) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float32).copy()
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    vals = arr[finite]
    mean = float(vals.mean())
    std = float(vals.std())
    if std <= 0:
        out = np.zeros_like(arr, dtype=np.float32)
        out[~finite] = 0.0
        return out
    out = (arr - mean) / std
    out[~finite] = 0.0
    return out.astype(np.float32)


def map_volume_to_unit_interval(volume: np.ndarray, modality: Modality) -> np.ndarray:
    if modality == "mri":
        z = normalize_volume_mri(volume)
        clipped = np.clip(z, -3.0, 3.0)
        return ((clipped + 3.0) / 6.0).astype(np.float32)

    if modality == "spect":
        arr = np.asarray(volume, dtype=np.float32).copy()
        vals = _finite_voxels(arr)
        if vals.size == 0:
            return np.zeros_like(arr, dtype=np.float32)
        lo, hi = np.percentile(vals, [1.0, 99.0])
        lo = float(lo)
        hi = float(hi)
        if hi <= lo:
            out = np.zeros_like(arr, dtype=np.float32)
            out[~np.isfinite(arr)] = 0.0
            return out
        out = (np.clip(arr, lo, hi) - lo) / (hi - lo)
        out[~np.isfinite(arr)] = 0.0
        return out.astype(np.float32)

    raise ValueError(f"Unsupported modality: {modality!r}; expected one of {MODALITIES}")


def slice_to_rgb(
    slice_2d: np.ndarray,
    tile_size: tuple[int, int] = DEFAULT_TILE_SIZE,
) -> np.ndarray:
    arr = np.asarray(slice_2d, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    gray = (arr * 255.0).astype(np.uint8)
    resized = Image.fromarray(gray).resize(tile_size, Image.BILINEAR)
    rgb = np.stack([np.asarray(resized, dtype=np.float32)] * 3, axis=-1) / 255.0
    return rgb.astype(np.float32)


def load_id_list(path: Path | str) -> list[str]:
    lines = Path(path).read_text().strip().splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def write_id_list(path: Path | str, ids: list[str]) -> None:
    Path(path).write_text("\n".join(ids) + ("\n" if ids else ""))


def load_labels(meta_csv: Path | str) -> dict[str, int]:
    df = pd.read_csv(meta_csv)
    if "ID" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{meta_csv} must have ID and label columns")
    return {str(r.ID): int(r.label) for r in df.itertuples(index=False)}


def stratified_holdout(
    subject_ids: list[str],
    labels: dict[str, int],
    n_val: int = 15,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Hold out ~n_val subjects stratified by label from train ids."""
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[str]] = {}
    for sid in subject_ids:
        y = int(labels[sid])
        by_label.setdefault(y, []).append(sid)

    val: list[str] = []
    classes = sorted(by_label.keys())
    remaining = n_val
    for i, c in enumerate(classes):
        pool = sorted(by_label[c])
        if i == len(classes) - 1:
            take = min(remaining, len(pool))
        else:
            frac = len(pool) / max(len(subject_ids), 1)
            take = max(1, int(round(n_val * frac)))
            take = min(take, len(pool), remaining - (len(classes) - i - 1))
        pick = rng.choice(pool, size=take, replace=False).tolist()
        val.extend(str(x) for x in pick)
        remaining = n_val - len(val)

    val_set = set(val)
    train = [s for s in subject_ids if s not in val_set]
    if len(val) < 2 or len(train) < 2:
        raise ValueError(f"Bad holdout: train={len(train)} val={len(val)}")
    return train, sorted(val)


def _find_cached_tile(modality: Modality, subject_id: str, z: int, cache_root: Path) -> Path | None:
    name = f"{subject_id}_slice_{z:03d}.npy"
    local = cache_root / modality / name
    if local.is_file() and local.stat().st_size > 0:
        return local
    for root in COCOLIT_CACHE_ROOTS:
        for split in ("train", "valid", "test"):
            cand = root / f"{modality}_slices_real" / split / name
            if cand.is_file() and cand.stat().st_size > 0:
                return cand
    return None


def _materialize_subject_tiles(
    sid: str,
    mri_vol: Path,
    spect_vol: Path,
    zs: list[int],
) -> list[dict]:
    mri_unit = map_volume_to_unit_interval(
        np.asarray(nib.load(str(mri_vol)).get_fdata(), dtype=np.float32), "mri"
    )
    spect_unit = map_volume_to_unit_interval(
        np.asarray(nib.load(str(spect_vol)).get_fdata(), dtype=np.float32), "spect"
    )
    out_mri = TMP_TILE_ROOT / "mri" / sid
    out_spect = TMP_TILE_ROOT / "spect" / sid
    out_mri.mkdir(parents=True, exist_ok=True)
    out_spect.mkdir(parents=True, exist_ok=True)
    rows = []
    for z in zs:
        mp = out_mri / f"{sid}_slice_{z:03d}.npy"
        sp = out_spect / f"{sid}_slice_{z:03d}.npy"
        if not mp.is_file():
            np.save(mp, slice_to_rgb(_take_slice(mri_unit, DEFAULT_SLICE_AXIS, z)))
        if not sp.is_file():
            np.save(sp, slice_to_rgb(_take_slice(spect_unit, DEFAULT_SLICE_AXIS, z)))
        rows.append(
            {
                "subject_id": sid,
                "slice_idx": z,
                "mri_path": str(mp),
                "spect_path": str(sp),
            }
        )
    return rows


def ensure_subject_slices(
    subject_ids: list[str],
    *,
    mri_root: Path,
    spect_root: Path,
    cache_root: Path,
    z_min: int = DEFAULT_SLICE_RANGE[0],
    z_max: int = DEFAULT_SLICE_RANGE[1],
    skip_existing: bool = True,
) -> pd.DataFrame:
    """Paired MRI/SPECT tiles: prefer CoCoLIT/local cache, else write under /tmp."""
    del skip_existing  # kept for API compatibility
    rows: list[dict] = []
    for sid in subject_ids:
        mri_vol = mri_root / f"{sid}.nii.gz"
        spect_vol = spect_root / f"{sid}.nii.gz"
        if not mri_vol.is_file():
            raise FileNotFoundError(f"Missing MRI: {mri_vol}")
        if not spect_vol.is_file():
            raise FileNotFoundError(f"Missing SPECT: {spect_vol}")

        vol = np.asarray(nib.load(str(mri_vol)).get_fdata(), dtype=np.float32)
        zs = slice_indices(z_min, z_max, vol.shape, DEFAULT_SLICE_AXIS)
        subject_rows: list[dict] = []
        complete = True
        for z in zs:
            mri_p = _find_cached_tile("mri", sid, z, cache_root)
            spect_p = _find_cached_tile("spect", sid, z, cache_root)
            if mri_p is None or spect_p is None:
                complete = False
                break
            subject_rows.append(
                {
                    "subject_id": sid,
                    "slice_idx": z,
                    "mri_path": str(mri_p),
                    "spect_path": str(spect_p),
                }
            )
        if complete:
            rows.extend(subject_rows)
        else:
            rows.extend(_materialize_subject_tiles(sid, mri_vol, spect_vol, zs))
    return pd.DataFrame(rows)


def attach_labels(manifest: pd.DataFrame, labels: dict[str, int]) -> pd.DataFrame:
    out = manifest.copy()
    out["label"] = out["subject_id"].map(lambda s: int(labels[str(s)]))
    if out["label"].isna().any():
        raise ValueError("Missing labels for some subjects")
    return out


def make_paired_sequence(dataframe: pd.DataFrame, batch_size: int = 16, shuffle: bool = True, seed: int = 42):
    """Lazy import TF Sequence. Yields ((mri, spect), y_onehot, subject_ids)."""
    from tensorflow.keras.utils import Sequence, to_categorical

    class PairedSliceSequence(Sequence):
        def __init__(self):
            self.df = dataframe.reset_index(drop=True)
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.seed = seed
            self.indices = np.arange(len(self.df))
            self.on_epoch_end()

        def __len__(self) -> int:
            return max(1, int(np.ceil(len(self.df) / self.batch_size)))

        def __getitem__(self, index: int):
            batch_idx = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
            batch = self.df.iloc[batch_idx]
            mri = np.stack([np.load(p) for p in batch["mri_path"]], axis=0).astype(np.float32)
            spect = np.stack([np.load(p) for p in batch["spect_path"]], axis=0).astype(np.float32)
            y = to_categorical(batch["label"].values, num_classes=2)
            sids = batch["subject_id"].astype(str).tolist()
            return (mri, spect), y, sids

        def on_epoch_end(self) -> None:
            if self.shuffle:
                rng = np.random.default_rng(self.seed)
                rng.shuffle(self.indices)

    return PairedSliceSequence()
