#!/usr/bin/env python3
"""Train PyTorch MobileNetV2 axial-slice classifiers (MRI or SPECT) on ROI 4 mm.

Aligned with the IL-CLDM / ARAD PyTorch stack:
  - 2-class head (logits → Softmax at eval)
  - fixed axial range (default z=[78, 112] → 35 slices)
  - train/test from official lists; stratified ~15-subject val holdout from train
  - subject-level majority vote; early-stop on val balanced accuracy

Outputs under --out-dir (intended: models/classifiers_roi_4mm/{mri,spect}/).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

FST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FST_ROOT))

from fst.slice_data import (  # noqa: E402
    load_id_list,
    load_labels,
    map_volume_to_unit_interval,
    slice_indices,
    stratified_holdout,
    write_id_list,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_SLICE_RANGE = (78, 112)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modality", choices=["mri", "spect"], required=True)
    p.add_argument(
        "--volume-root",
        type=Path,
        default=None,
        help="Directory of {id}.nii.gz (default from env / modality)",
    )
    p.add_argument("--train-list", type=Path, default=Path(os.environ.get("FST_TRAIN_LIST", "")))
    p.add_argument("--test-list", type=Path, default=Path(os.environ.get("FST_TEST_LIST", "")))
    p.add_argument("--meta-csv", type=Path, default=Path(os.environ.get("FST_META_CSV", "")))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--fit-list", type=Path, default=None, help="Reuse train-fit IDs if present")
    p.add_argument("--val-list", type=Path, default=None, help="Reuse val-holdout IDs if present")
    p.add_argument("--n-val-holdout", type=int, default=15)
    p.add_argument("--z-min", type=int, default=DEFAULT_SLICE_RANGE[0])
    p.add_argument("--z-max", type=int, default=DEFAULT_SLICE_RANGE[1])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--image-size", type=int, default=224)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_volume_root(modality: str) -> Path:
    if modality == "mri":
        return Path(os.environ.get("ILCLDM_WHOLE_MRI", "/workspace/IL-CLDM/data_rois/whole_MRI"))
    return Path(os.environ.get("ILCLDM_WHOLE_ABETA", "/workspace/IL-CLDM/data_rois/whole_Abeta_4mm"))


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class FixedRangeSliceDataset(Dataset):
    """On-the-fly axial tiles; volumes pre-mapped to unit interval in RAM."""

    def __init__(
        self,
        subject_ids: list[str],
        labels: dict[str, int],
        volumes: dict[str, np.ndarray],
        z_indices: list[int],
        transform,
    ):
        self.transform = transform
        self.volumes = volumes
        self.samples: list[tuple[str, int, int]] = []
        for sid in subject_ids:
            y = int(labels[sid])
            for z in z_indices:
                self.samples.append((sid, y, z))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sid, label, z = self.samples[idx]
        slc = self.volumes[sid][:, :, z]
        rgb = np.stack([slc, slc, slc], axis=-1).astype(np.float32)
        rgb = np.clip(rgb, 0.0, 1.0)
        img = Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")
        x = self.transform(img)
        return x, label, sid


class MobileNetV2Softmax2(nn.Module):
    """MobileNetV2 ImageNet backbone + 2-neuron Softmax head.

    Training uses CrossEntropyLoss on *logits* (pre-softmax). Softmax is applied
    for probabilities / majority-vote evaluation.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        backbone = models.mobilenet_v2(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_features, num_classes)
        self.backbone = backbone
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.forward(x))


@torch.no_grad()
def predict_slice_loader(model: MobileNetV2Softmax2, loader, device) -> pd.DataFrame:
    model.eval()
    rows = []
    for xb, yb, sids in loader:
        xb = xb.to(device, non_blocking=True)
        probs = model.predict_proba(xb).cpu().numpy()
        preds = probs.argmax(axis=1)
        y_np = yb.numpy()
        for i, sid in enumerate(sids):
            rows.append(
                {
                    "ID": str(sid),
                    "label": int(y_np[i]),
                    "pred_slice": int(preds[i]),
                    "prob_1": float(probs[i, 1]),
                }
            )
    return pd.DataFrame(rows)


def aggregate_subject(slice_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in slice_df.groupby("ID"):
        votes = g["pred_slice"].to_numpy()
        counts = np.bincount(votes, minlength=2)
        if counts[0] == counts[1]:
            pred = int(g["prob_1"].mean() >= 0.5)
        else:
            pred = int(np.argmax(counts))
        rows.append(
            {
                "ID": sid,
                "label": int(g["label"].iloc[0]),
                "pred": pred,
                "prob_1": float(g["prob_1"].mean()),
                "n_slices": int(len(g)),
                "vote_class0": int(counts[0]),
                "vote_class1": int(counts[1]),
            }
        )
    return pd.DataFrame(rows).sort_values("ID").reset_index(drop=True)


def subject_metrics(subj_df: pd.DataFrame) -> dict:
    y = subj_df["label"].to_numpy()
    pred = subj_df["pred"].to_numpy()
    prob = subj_df["prob_1"].to_numpy()
    out = {
        "n_subjects": int(len(subj_df)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y, prob))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def evaluate(model, loader, device):
    slice_df = predict_slice_loader(model, loader, device)
    subj_df = aggregate_subject(slice_df)
    return subject_metrics(subj_df), slice_df, subj_df


def load_volumes(ids: list[str], volume_root: Path, modality: str) -> dict[str, np.ndarray]:
    volumes: dict[str, np.ndarray] = {}
    missing = []
    for sid in ids:
        path = volume_root / f"{sid}.nii.gz"
        if not path.is_file():
            missing.append(sid)
            continue
        raw = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
        volumes[sid] = map_volume_to_unit_interval(raw, modality)  # type: ignore[arg-type]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} volumes under {volume_root}: {missing[:10]}")
    return volumes


def resolve_splits(args, labels: dict[str, int]) -> tuple[list[str], list[str], list[str]]:
    train_all = load_id_list(args.train_list)
    test_ids = load_id_list(args.test_list)
    train_all = [s for s in train_all if s in labels]
    test_ids = [s for s in test_ids if s in labels]

    fit_path = args.fit_list
    val_path = args.val_list
    if fit_path and val_path and fit_path.is_file() and val_path.is_file():
        fit_ids = load_id_list(fit_path)
        val_ids = load_id_list(val_path)
        print(f"Reusing shared holdout: fit={len(fit_ids)} val={len(val_ids)}")
        return fit_ids, val_ids, test_ids

    fit_ids, val_ids = stratified_holdout(
        train_all, labels, n_val=args.n_val_holdout, seed=args.seed
    )
    return fit_ids, val_ids, test_ids


def main():
    args = parse_args()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    volume_root = args.volume_root or default_volume_root(args.modality)
    if not volume_root.is_dir():
        raise SystemExit(f"ERROR: missing volume root: {volume_root}")
    for path, name in (
        (args.train_list, "train-list"),
        (args.test_list, "test-list"),
        (args.meta_csv, "meta-csv"),
    ):
        if not Path(path).is_file():
            raise SystemExit(f"ERROR: missing {name}: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"modality={args.modality} device={device} out_dir={args.out_dir}")
    print(f"volume_root={volume_root} slice_range=[{args.z_min},{args.z_max}]")

    labels = load_labels(args.meta_csv)
    fit_ids, val_ids, test_ids = resolve_splits(args, labels)

    # Persist shared lists next to out-dir parent classifiers root when possible
    write_id_list(args.out_dir / "train_fit.txt", fit_ids)
    write_id_list(args.out_dir / "val_holdout.txt", val_ids)
    write_id_list(args.out_dir / "test.txt", test_ids)

    all_ids = sorted(set(fit_ids) | set(val_ids) | set(test_ids))
    print(f"Loading {len(all_ids)} volumes…")
    volumes = load_volumes(all_ids, volume_root, args.modality)
    sample_shape = next(iter(volumes.values())).shape
    z_indices = slice_indices(args.z_min, args.z_max, sample_shape, axis=2)
    if not z_indices:
        raise SystemExit(f"Empty slice range for shape {sample_shape}")
    print(f"volume_shape={sample_shape} n_slices={len(z_indices)} z={z_indices[0]}…{z_indices[-1]}")

    transform = build_transform(args.image_size)

    def make_loader(ids: list[str], shuffle: bool) -> DataLoader:
        ds = FixedRangeSliceDataset(ids, labels, volumes, z_indices, transform)
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    train_loader = make_loader(fit_ids, shuffle=True)
    val_loader = make_loader(val_ids, shuffle=False)
    test_loader = make_loader(test_ids, shuffle=False)
    print(
        f"subjects fit/val/test={len(fit_ids)}/{len(val_ids)}/{len(test_ids)} | "
        f"slices {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}"
    )
    val_lab = [labels[s] for s in val_ids]
    print(f"val class counts: control={val_lab.count(0)} PD={val_lab.count(1)}")

    model = MobileNetV2Softmax2(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_val_ba = -1.0
    best_epoch = -1
    patience_left = args.patience
    best_path = args.out_dir / "best_mobilenet_v2.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        t0 = time.time()
        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(yb)
            n += len(yb)
        train_loss = running / max(n, 1)

        val_metrics, _, _ = evaluate(model, val_loader, device)
        val_ba = val_metrics["balanced_accuracy"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_balanced_accuracy": val_ba,
            "val_accuracy": val_metrics["accuracy"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_f1_macro": val_metrics["f1_macro"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs} loss={train_loss:.4f} "
            f"val_BA={val_ba:.4f} val_auc={val_metrics['roc_auc']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} ({row['seconds']:.1f}s)",
            flush=True,
        )

        improved = np.isfinite(val_ba) and val_ba > best_val_ba + 1e-6
        if improved:
            best_val_ba = float(val_ba)
            best_epoch = epoch
            patience_left = args.patience
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_balanced_accuracy": best_val_ba,
                    "modality": args.modality,
                    "slice_range": [args.z_min, args.z_max],
                    "z_indices": z_indices,
                    "image_size": args.image_size,
                    "num_classes": 2,
                    "head": "Linear(2)+Softmax(eval)",
                    "framework": "pytorch",
                    "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                },
                best_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(
                    f"Early stop at epoch {epoch} "
                    f"(best epoch={best_epoch}, best_val_BA={best_val_ba:.4f})",
                    flush=True,
                )
                break

    pd.DataFrame(history).to_csv(args.out_dir / "train_history.csv", index=False)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    val_metrics, _, val_subjects = evaluate(model, val_loader, device)
    test_metrics, test_slices, test_subjects = evaluate(model, test_loader, device)

    val_subjects.to_csv(args.out_dir / "val_subject_predictions.csv", index=False)
    test_slices.to_csv(args.out_dir / "test_slice_predictions.csv", index=False)
    test_subjects.to_csv(args.out_dir / "test_subject_predictions.csv", index=False)

    report = classification_report(
        test_subjects["label"], test_subjects["pred"], digits=4, zero_division=0
    )
    (args.out_dir / "classification_report_test.txt").write_text(report)
    cm = confusion_matrix(test_subjects["label"], test_subjects["pred"])
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        args.out_dir / "confusion_matrix_test.csv"
    )

    config = {
        "framework": "pytorch",
        "model": "mobilenet_v2_imagenet",
        "modality": args.modality,
        "num_classes": 2,
        "head": "Linear(2) + Softmax (eval); CrossEntropyLoss on logits",
        "slice_axis": 2,
        "slice_range": [args.z_min, args.z_max],
        "n_slices_per_subject": len(z_indices),
        "z_indices": z_indices,
        "image_size": args.image_size,
        "preprocess": "mri_zscore_clip3sigma_to_unit"
        if args.modality == "mri"
        else "spect_p1_p99_to_unit",
        "aggregation": "majority_vote_slices",
        "early_stop_monitor": "val_balanced_accuracy",
        "n_val_holdout": len(val_ids),
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy": best_val_ba,
        "checkpoint": str(best_path),
        "volume_root": str(volume_root),
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (args.out_dir / "metrics_valid.json").write_text(json.dumps(val_metrics, indent=2))
    (args.out_dir / "metrics_test.json").write_text(json.dumps(test_metrics, indent=2))
    summary = {
        "config": config,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "n_train_subjects": len(fit_ids),
        "n_val_subjects": len(val_ids),
        "n_test_subjects": len(test_ids),
    }
    (args.out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== TEST (subject-level, majority vote) ===", flush=True)
    print(report, flush=True)
    print(json.dumps(test_metrics, indent=2), flush=True)
    print(f"Saved under {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
