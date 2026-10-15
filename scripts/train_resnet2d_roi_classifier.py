#!/usr/bin/env python3
"""Train ResNet18_2D on the Keras MobileNet ROI 4 mm MRI protocol.

Same subjects, axial window, and tile preprocessing as
classifiers/classifiers_roi_4mm/mri (z=[78, 112], 256x256 RGB in [0, 1]).
Early stopping uses subject-level balanced accuracy. The checkpoint is the
best validation epoch, not the last one.
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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

FST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FST_ROOT))
sys.path.insert(0, str(FST_ROOT / "code"))

from fst.slice_data import (  # noqa: E402
    load_id_list,
    load_labels,
    map_volume_to_unit_interval,
    slice_indices,
    slice_to_rgb,
    write_id_list,
)
from resnet2D import MOBILENET_IMAGE_SIZE, MOBILENET_SLICE_RANGE, ResNet18_2D  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modality", choices=["mri", "spect"], default="mri")
    p.add_argument("--volume-root", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--train-list", type=Path, default=Path(os.environ.get("FST_TRAIN_LIST", "")))
    p.add_argument("--test-list", type=Path, default=Path(os.environ.get("FST_TEST_LIST", "")))
    p.add_argument("--meta-csv", type=Path, default=Path(os.environ.get("FST_META_CSV", "")))
    p.add_argument("--fit-list", type=Path, required=True)
    p.add_argument("--val-list", type=Path, required=True)
    p.add_argument("--z-min", type=int, default=MOBILENET_SLICE_RANGE[0])
    p.add_argument("--z-max", type=int, default=MOBILENET_SLICE_RANGE[1])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--image-size", type=int, default=MOBILENET_IMAGE_SIZE)
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


class FixedRangeSliceDataset(Dataset):
    """Axial tiles in the same layout as Keras MobileNet (NHWC [0, 1] -> NCHW)."""

    def __init__(
        self,
        subject_ids: list[str],
        labels: dict[str, int],
        volumes: dict[str, np.ndarray],
        z_indices: list[int],
        image_size: int,
    ):
        self.volumes = volumes
        self.image_size = image_size
        self.samples: list[tuple[str, int, int]] = []
        for sid in subject_ids:
            y = int(labels[sid])
            for z in z_indices:
                self.samples.append((sid, y, z))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sid, label, z = self.samples[idx]
        tile = slice_to_rgb(
            self.volumes[sid][:, :, z],
            tile_size=(self.image_size, self.image_size),
        )
        x = torch.from_numpy(np.ascontiguousarray(tile)).permute(2, 0, 1)
        return x, label, sid


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


@torch.no_grad()
def predict_slices(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    for xb, yb, sids in loader:
        xb = xb.to(device, non_blocking=True)
        probs = torch.softmax(model(xb), dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        y_np = yb.numpy()
        for i, sid in enumerate(sids):
            rows.append(
                {
                    "subject_id": str(sid),
                    "label": int(y_np[i]),
                    "pred_class": int(preds[i]),
                    "pred_prob_pd": float(probs[i, 1]),
                }
            )
    return pd.DataFrame(rows)


def aggregate_subjects(slice_df: pd.DataFrame) -> pd.DataFrame:
    """Majority vote, matching the Keras MobileNet evaluator (value_counts idxmax)."""
    rows = []
    for sid, grp in slice_df.groupby("subject_id", sort=False):
        votes = grp["pred_class"].value_counts()
        rows.append(
            {
                "subject_id": str(sid),
                "label": int(grp["label"].iloc[0]),
                "pred_class": int(votes.idxmax()),
                "n_slices": int(len(grp)),
                "votes_control": int((grp["pred_class"] == 0).sum()),
                "votes_pd": int((grp["pred_class"] == 1).sum()),
                "mean_prob_pd": float(grp["pred_prob_pd"].mean()),
            }
        )
    return pd.DataFrame(rows)


def subject_metrics(subj: pd.DataFrame) -> dict:
    yt = subj["label"].to_numpy()
    yp = subj["pred_class"].to_numpy()
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
        metrics["auc"] = float(roc_auc_score(yt, subj["mean_prob_pd"].to_numpy()))
    return metrics


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict, pd.DataFrame]:
    subj = aggregate_subjects(predict_slices(model, loader, device))
    return subject_metrics(subj), subj


def main() -> None:
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
        (args.fit_list, "fit-list"),
        (args.val_list, "val-list"),
    ):
        if not Path(path).is_file():
            raise SystemExit(f"ERROR: missing {name}: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"cuda_device={torch.cuda.get_device_name(0)}", flush=True)
    print(f"modality={args.modality} device={device} out_dir={args.out_dir}", flush=True)

    labels = load_labels(args.meta_csv)
    fit_ids = [s for s in load_id_list(args.fit_list) if s in labels]
    val_ids = [s for s in load_id_list(args.val_list) if s in labels]
    test_ids = [s for s in load_id_list(args.test_list) if s in labels]
    write_id_list(args.out_dir / "train_fit.txt", fit_ids)
    write_id_list(args.out_dir / "val_holdout.txt", val_ids)
    write_id_list(args.out_dir / "test.txt", test_ids)

    all_ids = sorted(set(fit_ids) | set(val_ids) | set(test_ids))
    print(f"Loading {len(all_ids)} volumes from {volume_root}…", flush=True)
    volumes = load_volumes(all_ids, volume_root, args.modality)
    sample_shape = next(iter(volumes.values())).shape
    z_indices = slice_indices(args.z_min, args.z_max, sample_shape, axis=2)
    if not z_indices:
        raise SystemExit(f"Empty slice range for shape {sample_shape}")
    print(
        f"volume_shape={sample_shape} n_slices={len(z_indices)} "
        f"z={z_indices[0]}…{z_indices[-1]} image_size={args.image_size}",
        flush=True,
    )

    def make_loader(ids: list[str], shuffle: bool) -> DataLoader:
        ds = FixedRangeSliceDataset(ids, labels, volumes, z_indices, args.image_size)
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

    train_loader = make_loader(fit_ids, shuffle=True)
    val_loader = make_loader(val_ids, shuffle=False)
    test_loader = make_loader(test_ids, shuffle=False)
    print(
        f"subjects fit/val/test={len(fit_ids)}/{len(val_ids)}/{len(test_ids)} | "
        f"slices {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}",
        flush=True,
    )

    model = ResNet18_2D(
        {
            "in_channels": 3,
            "num_classes": 2,
            "pretrained": True,
            "image_size": args.image_size,
        }
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-7)

    history = []
    best_val_ba = -1.0
    best_epoch = 0
    patience_left = args.patience
    best_path = args.out_dir / "best_resnet18_2d.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        correct = 0
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
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            n += len(yb)
        train_loss = running / max(n, 1)
        train_acc = correct / max(n, 1)

        val_metrics, _ = evaluate(model, val_loader, device)
        val_ba = float(val_metrics["balanced_accuracy"])
        row = {
            "epoch": epoch,
            "loss": train_loss,
            "accuracy": train_acc,
            "val_balanced_accuracy": val_ba,
            "val_accuracy": val_metrics["accuracy"],
            "val_auc": val_metrics.get("auc", float("nan")),
            "val_f1_macro": val_metrics["f1_macro"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs} loss={train_loss:.4f} acc={train_acc:.4f} "
            f"val_subject_BA={val_ba:.4f} auc={row['val_auc']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} ({row['seconds']:.1f}s)",
            flush=True,
        )

        if np.isfinite(val_ba) and val_ba > best_val_ba:
            best_val_ba = val_ba
            best_epoch = epoch
            patience_left = args.patience
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_balanced_accuracy": best_val_ba,
                    "modality": args.modality,
                    "architecture": "ResNet18_2D",
                    "slice_range": [args.z_min, args.z_max],
                    "z_indices": z_indices,
                    "image_size": args.image_size,
                    "in_channels": 3,
                    "num_classes": 2,
                    "head": "Dropout-Linear(512,256)-ReLU-Dropout-Linear(256,2)",
                    "framework": "pytorch",
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
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2))

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    val_metrics, val_subjects = evaluate(model, val_loader, device)
    test_metrics, test_subjects = evaluate(model, test_loader, device)
    val_subjects.to_csv(args.out_dir / "val_subject_predictions.csv", index=False)
    test_subjects.to_csv(args.out_dir / "test_subject_predictions.csv", index=False)
    text = classification_report(
        test_subjects["label"],
        test_subjects["pred_class"],
        digits=4,
        zero_division=0,
    )
    (args.out_dir / "classification_report_test.txt").write_text(text)
    cm = confusion_matrix(test_subjects["label"], test_subjects["pred_class"])
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        args.out_dir / "confusion_matrix_test.csv"
    )

    config = {
        "framework": "pytorch",
        "architecture": "ResNet18_2D",
        "modality": args.modality,
        "pretrained": "imagenet",
        "in_channels": 3,
        "num_classes": 2,
        "head": "Dropout-Linear(512,256)-ReLU-Dropout-Linear(256,2)",
        "activation_out": "softmax_at_eval",
        "loss": "cross_entropy_on_logits",
        "optimizer": "adam",
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "height": args.image_size,
        "width": args.image_size,
        "slice_axis": 2,
        "slice_range": [args.z_min, args.z_max],
        "n_slices_per_subject": len(z_indices),
        "z_indices": z_indices,
        "preprocess": "mri_zscore_clip3sigma_to_unit"
        if args.modality == "mri"
        else "spect_p1_p99_to_unit",
        "aggregation": "majority_vote_slices",
        "early_stop_monitor": "val_balanced_accuracy",
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy": best_val_ba,
        "n_train_subjects": len(fit_ids),
        "n_val_subjects": len(val_ids),
        "n_test_subjects": len(test_ids),
        "seed": args.seed,
        "checkpoint": str(best_path),
        "volume_root": str(volume_root),
        "fit_list": str(args.fit_list),
        "val_list": str(args.val_list),
        "test_list": str(args.test_list),
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (args.out_dir / "metrics_valid.json").write_text(json.dumps(val_metrics, indent=2))
    (args.out_dir / "metrics_test.json").write_text(json.dumps(test_metrics, indent=2))

    print("\n=== TEST (subject-level, majority vote) ===", flush=True)
    print(text, flush=True)
    public = {
        k: test_metrics[k]
        for k in ("accuracy", "balanced_accuracy", "auc", "f1_macro")
        if k in test_metrics
    }
    print(json.dumps(public, indent=2), flush=True)
    print(f"Saved {best_path}", flush=True)


if __name__ == "__main__":
    main()
