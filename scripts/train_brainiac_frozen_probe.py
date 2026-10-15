#!/usr/bin/env python3
"""Frozen BrainIAC ViT-B linear probe: MRI whole brain vs ROI 4 mm.

The ViT stays frozen. Only a Linear(768, 2) softmax head is trained.
Splits are the persisted MobileNet holdout
classifiers/classifiers_roi_4mm/_shared_split_z78_112 (seed 42).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from monai.networks.nets import ViT
from monai.transforms import Compose, EnsureChannelFirst, NormalizeIntensity, Resize
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

FST_ROOT = Path(__file__).resolve().parents[1]
ILCLDM_ROOT = Path("/workspace/IL-CLDM")
SPLIT_DIR = FST_ROOT / "classifiers" / "classifiers_roi_4mm" / "_shared_split_z78_112"
META_CSV = ILCLDM_ROOT / "data_info" / "data_info.csv"
BASELINE_METRICS = FST_ROOT / "classifiers" / "classifiers_roi_4mm" / "mri" / "metrics_test.json"
VOLUME_ROOTS = {
    "whole": ILCLDM_ROOT / "data" / "whole_MRI",
    "roi": ILCLDM_ROOT / "data_rois" / "whole_MRI",
}
OUT_NAMES = {"whole": "mri_whole", "roi": "mri_roi_4mm"}
EXPERIMENTS = {
    "brainiac_vitb_frozen": {
        "cache": "feat_cache_brainiac_vitb_frozen",
        "out": "brainiac_vitb_frozen",
        "head": "linear",
        "backbone_name": "BrainIAC ViT-B SimCLR (MONAI)",
        "head_name": "Linear(768, 2)",
    },
    "brainiac_vit_mci_pdhead": {
        "cache": "feat_cache_brainiac_vit_mci_pdhead",
        "out": "brainiac_vit_mci_pdhead",
        "head": "mlp",
        "backbone_name": "BrainIAC ViT-B fine-tuned on MCI (vit_mci.ckpt), frozen",
        "head_name": "768-300-150-75-2",
    },
}
WHOLE_NATIVE = (182, 218, 182)
WHOLE_CROP = (slice(10, 170), slice(18, 210), slice(10, 170))
WHOLE_CROPPED = (160, 192, 160)
ROI_SHAPE = (96, 112, 114)
IMG_SIZE = (96, 96, 96)
FEAT_DIM = 768


def load_id_list(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def load_labels(path: Path) -> dict[str, int]:
    df = pd.read_csv(path)
    return {str(r.ID): int(r.label) for r in df.itertuples(index=False)}


def resolve_nifti(root: Path, sid: str) -> Path:
    for suffix in (".nii.gz", ".nii"):
        path = root / f"{sid}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"no NIfTI for {sid} under {root}")


def _affine_list(affine: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in np.asarray(affine)]


def audit_tree(field: str, root: Path) -> dict:
    files = sorted(p for p in root.iterdir() if p.suffix in {".gz", ".nii"} or p.name.endswith(".nii.gz"))
    files = [p for p in files if p.name.endswith(".nii") or p.name.endswith(".nii.gz")]
    shapes: Counter[str] = Counter()
    zooms: Counter[str] = Counter()
    axcodes: Counter[str] = Counter()
    dtypes: Counter[str] = Counter()
    subjects = []
    outside_fracs = []
    ref_affine = None
    max_affine_abs = 0.0
    for path in files:
        img = nib.load(str(path))
        shape = tuple(int(v) for v in img.shape[:3])
        zoom = tuple(float(v) for v in img.header.get_zooms()[:3])
        code = "".join(nib.aff2axcodes(img.affine))
        shapes[str(shape)] += 1
        zooms[str(zoom)] += 1
        axcodes[code] += 1
        dtypes[str(img.get_data_dtype())] += 1
        if ref_affine is None:
            ref_affine = np.asarray(img.affine, dtype=np.float64)
        else:
            max_affine_abs = max(max_affine_abs, float(np.max(np.abs(np.asarray(img.affine) - ref_affine))))
        rec = {
            "id": path.name.split(".")[0],
            "shape": list(shape),
            "zooms_mm": list(zoom),
            "axcodes": code,
        }
        if field == "whole" and shape == WHOLE_NATIVE:
            data = np.asanyarray(img.dataobj)
            finite = np.isfinite(data)
            nz = finite & (data != 0)
            inside = np.zeros(shape, dtype=bool)
            inside[WHOLE_CROP] = True
            n_nz = int(nz.sum())
            n_out = int((nz & ~inside).sum())
            frac = float(n_out / n_nz) if n_nz else 0.0
            rec.update(
                {
                    "n_nonzero": n_nz,
                    "n_nonzero_outside_crop": n_out,
                    "frac_nonzero_outside_crop": frac,
                    "n_nonfinite": int((~finite).sum()),
                }
            )
            outside_fracs.append(frac)
        subjects.append(rec)
        del img
    summary = {
        "field": field,
        "volume_root": str(root),
        "n_files": len(files),
        "shapes": dict(shapes),
        "zooms_mm": dict(zooms),
        "axcodes": dict(axcodes),
        "dtypes": dict(dtypes),
        "reference_affine": _affine_list(ref_affine) if ref_affine is not None else None,
        "max_abs_affine_diff_vs_first": max_affine_abs,
    }
    if field == "whole":
        summary["crop_index"] = "data[10:170, 18:210, 10:170]"
        summary["crop_result_shape"] = list(WHOLE_CROPPED)
        summary["native_shape_expected"] = list(WHOLE_NATIVE)
        if outside_fracs:
            arr = np.asarray(outside_fracs, dtype=np.float64)
            summary["nonzero_outside_crop"] = {
                "n_subjects": int(arr.size),
                "mean_frac": float(arr.mean()),
                "min_frac": float(arr.min()),
                "max_frac": float(arr.max()),
                "n_with_any_outside": int((arr > 0).sum()),
            }
    else:
        summary["crop_index"] = None
        summary["note"] = "ROI patches are already cropped to the striatal bbox; no IL-CLDM [10:170,...] crop is applied."
        summary["expected_shape"] = list(ROI_SHAPE)
    summary["subjects"] = subjects
    return summary


def write_audit(path: Path) -> dict:
    payload = {
        "description": (
            "Header and nonzero-outside-crop audit of the MRI NIfTIs used by "
            "brainiac_vitb_frozen. The whole-brain crop is the IL-CLDM border "
            "trim, not the striatal ROI."
        ),
        "fields": {
            "whole": audit_tree("whole", VOLUME_ROOTS["whole"]),
            "roi": audit_tree("roi", VOLUME_ROOTS["roi"]),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path}", flush=True)
    for name, block in payload["fields"].items():
        print(
            f"  {name}: n={block['n_files']} shapes={block['shapes']} zooms={block['zooms_mm']} axcodes={block['axcodes']}",
            flush=True,
        )
        extra = block.get("nonzero_outside_crop")
        if extra:
            print(f"  nonzero outside crop: {extra}", flush=True)
    return payload


def preprocess(volume: np.ndarray, field: str) -> torch.Tensor:
    """Crop when needed, resize to 96^3, z-score nonzero voxels. Returns (1,96,96,96)."""
    vol = np.asarray(volume, dtype=np.float32)
    if field == "whole":
        if vol.shape == WHOLE_NATIVE:
            vol = vol[WHOLE_CROP]
        elif vol.shape != WHOLE_CROPPED:
            raise ValueError(f"whole volume shape {vol.shape}, expected {WHOLE_NATIVE} or {WHOLE_CROPPED}")
    elif field == "roi":
        if vol.shape != ROI_SHAPE:
            raise ValueError(f"roi volume shape {vol.shape}, expected {ROI_SHAPE}")
    else:
        raise ValueError(field)
    tfm = Compose(
        [
            EnsureChannelFirst(channel_dim="no_channel"),
            Resize(spatial_size=IMG_SIZE, mode="trilinear"),
            NormalizeIntensity(nonzero=True, channel_wise=True),
        ]
    )
    out = tfm(vol)
    if not isinstance(out, torch.Tensor):
        out = torch.as_tensor(out)
    return out.float()


def _checkpoint_tensors(ckpt_path: Path) -> list[tuple[str, torch.Tensor]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        raise RuntimeError(f"{ckpt_path} has no state_dict")
    tensors = [(key, value) for key, value in state.items() if torch.is_tensor(value)]
    if not tensors:
        raise RuntimeError(f"{ckpt_path} state_dict has no tensors")
    return tensors


def load_backbone(ckpt_path: Path, device: torch.device) -> tuple[nn.Module, list[tuple[str, tuple[int, ...]]]]:
    """Load the MONAI ViT body. Discard classifier / optimizer tensors.

    Lightning checkpoints prefix keys (``model.backbone.``). Each ViT parameter
    is matched by exact name or by a dotted suffix, so those prefixes fall away.
    """
    backbone = ViT(
        in_channels=1,
        img_size=IMG_SIZE,
        patch_size=(16, 16, 16),
        hidden_size=FEAT_DIM,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        save_attn=True,
    )
    tensors = _checkpoint_tensors(ckpt_path)
    cleaned: dict[str, torch.Tensor] = {}
    used: set[str] = set()
    for model_key, model_value in backbone.state_dict().items():
        hits = []
        for key, value in tensors:
            if tuple(value.shape) != tuple(model_value.shape):
                continue
            if key == model_key or key.endswith("." + model_key):
                if "classifier" in key or key.startswith("head.") or ".head." in key:
                    continue
                hits.append((key, value))
        if len(hits) != 1:
            names = [key for key, _ in hits]
            raise RuntimeError(f"ViT key {model_key} matched {names or 'nothing'}")
        cleaned[model_key] = hits[0][1]
        used.add(hits[0][0])
    missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch missing={missing[:8]} unexpected={unexpected[:8]}")
    discarded = [
        (key, tuple(value.shape))
        for key, value in tensors
        if key not in used and any(token in key for token in ("head", "classifier", "fc"))
    ]
    print(f"loaded ViT from {ckpt_path.name}; discarded head tensors: {discarded}", flush=True)
    if not discarded and any("head" in key or "classifier" in key or ".fc." in key for key, _ in tensors):
        raise RuntimeError("classifier tensors were present but not discarded")
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad_(False)
    return backbone.to(device), discarded


def make_head(kind: str) -> nn.Module:
    if kind == "mlp":
        return nn.Sequential(
            nn.Linear(FEAT_DIM, 300),
            nn.ReLU(),
            nn.Linear(300, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 2),
        )
    if kind == "linear":
        return nn.Linear(FEAT_DIM, 2)
    raise ValueError(f"unknown head kind {kind}")


@torch.no_grad()
def embed_volume(backbone: nn.Module, volume: torch.Tensor, device: torch.device) -> np.ndarray:
    batch = volume.unsqueeze(0).to(device)
    tokens, _hidden = backbone(batch)
    # MONAI ViT (classification=False) has no CLS token. BrainIAC reads token 0.
    feat = tokens[:, 0]
    return feat.squeeze(0).detach().cpu().numpy().astype(np.float32)


def extract_split(
    field: str,
    ids: list[str],
    labels: dict[str, int],
    cache_dir: Path,
    backbone: nn.Module,
    device: torch.device,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    root = VOLUME_ROOTS[field]
    pending = []
    for sid in ids:
        dest = cache_dir / f"{sid}.npy"
        if dest.is_file() and np.load(dest).shape == (FEAT_DIM,):
            continue
        pending.append(sid)
    print(f"extract {field}: {len(ids) - len(pending)} cached, {len(pending)} to run", flush=True)
    for i, sid in enumerate(pending, start=1):
        path = resolve_nifti(root, sid)
        img = nib.load(str(path))
        feat = embed_volume(backbone, preprocess(np.asanyarray(img.dataobj), field), device)
        if feat.shape != (FEAT_DIM,):
            raise RuntimeError(f"{sid} feature shape {feat.shape}")
        np.save(cache_dir / f"{sid}.npy", feat)
        if i == 1 or i % 20 == 0 or i == len(pending):
            print(f"  extracted {i}/{len(pending)} {sid}", flush=True)
    rows = []
    for sid in ids:
        rows.append({"id": sid, "label": int(labels[sid]), "feature": f"{sid}.npy"})
    pd.DataFrame(rows).to_csv(cache_dir / "index.csv", index=False)


def stack_features(cache_dir: Path, ids: list[str], labels: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for sid in ids:
        xs.append(np.load(cache_dir / f"{sid}.npy"))
        ys.append(int(labels[sid]))
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def subject_metrics(ids: list[str], y_true: np.ndarray, logits: np.ndarray) -> dict:
    prob = torch.softmax(torch.as_tensor(logits), dim=1).numpy()
    pred = prob.argmax(axis=1)
    rows = [
        {
            "subject_id": str(sid),
            "label": int(y),
            "pred_class": int(p),
            "pred_prob_pd": float(pr[1]),
        }
        for sid, y, p, pr in zip(ids, y_true, pred, prob)
    ]
    metrics = {
        "n_subjects": int(len(rows)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "classification_report": classification_report(y_true, pred, output_dict=True, zero_division=0),
        "per_subject": rows,
        "decision": "argmax",
        "score_for_auc": "softmax[:,1]",
    }
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, prob[:, 1]))
    return metrics


def train_head(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    val_ids: list[str],
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    batch_size: int,
    seed: int,
    head_kind: str,
) -> tuple[nn.Module, dict, list[dict]]:
    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)
    head = make_head(head_kind).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    x_fit_t = torch.as_tensor(x_fit, device=device)
    y_fit_t = torch.as_tensor(y_fit, device=device)
    x_val_t = torch.as_tensor(x_val, device=device)
    best_ba = -1.0
    best_state = None
    best_epoch = 0
    wait = 0
    best_val = None
    history = []
    n = len(y_fit)
    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(n, generator=g).tolist()
        total = 0.0
        seen = 0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb = x_fit_t[idx]
            yb = y_fit_t[idx]
            opt.zero_grad(set_to_none=True)
            logits = head(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(idx)
            seen += len(idx)
        head.eval()
        with torch.no_grad():
            val_logits = head(x_val_t).cpu().numpy()
        val_metrics = subject_metrics(val_ids, y_val, val_logits)
        ba = val_metrics["balanced_accuracy"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / max(seen, 1),
                "val_balanced_accuracy": ba,
                "val_auc": val_metrics.get("auc"),
            }
        )
        print(
            f"  epoch {epoch:03d} loss={history[-1]['train_loss']:.4f} val_BA={ba:.4f} val_AUC={val_metrics.get('auc', float('nan')):.4f}",
            flush=True,
        )
        if ba > best_ba:
            best_ba = ba
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            best_val = val_metrics
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  early stop at epoch {epoch}; best epoch {best_epoch} BA={best_ba:.4f}", flush=True)
                break
    if best_state is None or best_val is None:
        raise RuntimeError("head training produced no checkpoint")
    head.load_state_dict(best_state)
    head.eval()
    return head, {"best_epoch": best_epoch, "best_val_balanced_accuracy": best_ba, "val_metrics": best_val}, history


def _checkpoint_source(path: Path) -> str | None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload.get("source")
    return None


def crop_note(field: str) -> str | None:
    if field == "whole":
        return "data[10:170, 18:210, 10:170] applied when native shape is 182x218x182; skipped when already 160x192x160"
    return None


def run_field(args: argparse.Namespace, labels: dict[str, int]) -> None:
    field = args.field
    fit_ids = load_id_list(SPLIT_DIR / "train_fit.txt")
    val_ids = load_id_list(SPLIT_DIR / "val_holdout.txt")
    test_ids = load_id_list(SPLIT_DIR / "test.txt")
    for sid in fit_ids + val_ids + test_ids:
        if sid not in labels:
            raise KeyError(f"{sid} missing from {META_CSV}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = EXPERIMENTS[args.experiment]
    print(f"experiment={args.experiment} field={field} device={device} cuda_count={torch.cuda.device_count()}", flush=True)
    cache_dir = FST_ROOT / "data" / spec["cache"] / OUT_NAMES[field]
    out_dir = FST_ROOT / "classifiers" / spec["out"] / OUT_NAMES[field]
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    backbone, discarded_head = load_backbone(args.checkpoint, device)
    extract_split(field, fit_ids + val_ids + test_ids, labels, cache_dir, backbone, device)
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    x_fit, y_fit = stack_features(cache_dir, fit_ids, labels)
    x_val, y_val = stack_features(cache_dir, val_ids, labels)
    x_te, y_te = stack_features(cache_dir, test_ids, labels)
    head, info, history = train_head(
        x_fit,
        y_fit,
        x_val,
        y_val,
        val_ids,
        device,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        head_kind=spec["head"],
    )
    with torch.no_grad():
        te_logits = head(torch.as_tensor(x_te, device=device)).cpu().numpy()
    test_metrics = subject_metrics(test_ids, y_te, te_logits)
    val_metrics = info["val_metrics"]
    torch.save(head.state_dict(), out_dir / "head.pt")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    (out_dir / "metrics_valid.json").write_text(json.dumps(val_metrics, indent=2))
    (out_dir / "metrics_test.json").write_text(json.dumps(test_metrics, indent=2))
    pd.DataFrame(test_metrics["per_subject"]).to_csv(out_dir / "predictions_test.csv", index=False)
    pd.DataFrame(val_metrics["per_subject"]).to_csv(out_dir / "predictions_valid.csv", index=False)
    config = {
        "experiment": args.experiment,
        "field": field,
        "framework": "pytorch",
        "backbone": spec["backbone_name"],
        "backbone_frozen": True,
        "feature": "token_0 of ViT output, dim 768 (no learned CLS; matches BrainIAC ViTBackboneNet)",
        "head": spec["head_name"],
        "head_hidden_activation": "ReLU" if spec["head"] == "mlp" else None,
        "discarded_mci_head": discarded_head if args.experiment == "brainiac_vit_mci_pdhead" else None,
        "activation_out": "softmax",
        "loss": "cross_entropy",
        "class_weight": None,
        "dropout": 0.0,
        "optimizer": "Adam",
        "lr": args.lr,
        "seed": args.seed,
        "epochs_max": args.epochs,
        "patience": args.patience,
        "early_stop_metric": "val_subject_balanced_accuracy",
        "best_epoch": info["best_epoch"],
        "best_val_balanced_accuracy": info["best_val_balanced_accuracy"],
        "batch_size_head": args.batch_size,
        "n_train_subjects": len(fit_ids),
        "n_val_subjects": len(val_ids),
        "n_test_subjects": len(test_ids),
        "split_dir": str(SPLIT_DIR),
        "baseline_classifier": "classifiers/classifiers_roi_4mm/mri",
        "baseline_metrics": str(BASELINE_METRICS),
        "volume_root": str(VOLUME_ROOTS[field]),
        "crop": crop_note(field),
        "roi_patch_note": (
            "Full 3D ROI patch 96x112x114 is resized to 96^3. "
            "The MobileNet baseline only sees axial slices z in [78, 112]."
            if field == "roi"
            else None
        ),
        "input_size": list(IMG_SIZE),
        "resize": "monai.Resize trilinear",
        "normalize": "monai.NormalizeIntensity nonzero=True channel_wise=True",
        "checkpoint": str(args.checkpoint),
        "checkpoint_source": _checkpoint_source(args.checkpoint),
        "feature_cache": str(cache_dir),
        "head_weights": str(out_dir / "head.pt"),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(
        json.dumps(
            {k: test_metrics[k] for k in ("accuracy", "balanced_accuracy", "auc", "f1_macro")},
            indent=2,
        ),
        flush=True,
    )


def _pick_metrics(path: Path) -> dict:
    metrics = json.loads(path.read_text())
    return {
        "path": str(path),
        "n_subjects": metrics.get("n_subjects"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "auc": metrics.get("auc"),
        "accuracy": metrics.get("accuracy"),
        "f1_macro": metrics.get("f1_macro"),
    }


def write_comparison(experiment: str) -> None:
    baseline = json.loads(BASELINE_METRICS.read_text())
    frozen = FST_ROOT / "classifiers" / "brainiac_vitb_frozen"
    if experiment == "brainiac_vitb_frozen":
        root = frozen
        payload = {
            "baseline_mri_mobilenet_roi_slices_z78_112": _pick_metrics(BASELINE_METRICS)
            | {
                "note": "Existing Keras MobileNet on ROI axial slices z=[78,112]. Not retrained.",
                "balanced_accuracy_check": baseline.get("balanced_accuracy"),
            },
            "brainiac_frozen_mri_whole": _pick_metrics(root / "mri_whole" / "metrics_test.json"),
            "brainiac_frozen_mri_roi_4mm": _pick_metrics(root / "mri_roi_4mm" / "metrics_test.json"),
            "roi_comparison_note": (
                "brainiac_frozen_mri_roi_4mm sees the full 3D ROI patch. "
                "The MobileNet baseline sees only 35 axial slices."
            ),
        }
    elif experiment == "brainiac_vit_mci_pdhead":
        root = FST_ROOT / "classifiers" / "brainiac_vit_mci_pdhead"
        payload = {
            "baseline_mri_mobilenet_roi_slices_z78_112": _pick_metrics(BASELINE_METRICS)
            | {
                "note": "Existing Keras MobileNet on ROI axial slices z=[78,112]. Not retrained.",
                "balanced_accuracy_check": baseline.get("balanced_accuracy"),
            },
            "brainiac_vitb_frozen_mri_whole": _pick_metrics(frozen / "mri_whole" / "metrics_test.json"),
            "brainiac_vitb_frozen_mri_roi_4mm": _pick_metrics(frozen / "mri_roi_4mm" / "metrics_test.json"),
            "brainiac_vit_mci_pdhead_mri_whole": _pick_metrics(root / "mri_whole" / "metrics_test.json"),
            "brainiac_vit_mci_pdhead_mri_roi_4mm": _pick_metrics(root / "mri_roi_4mm" / "metrics_test.json"),
            "head": "768-300-150-75-2 with ReLU between hidden layers; ViT frozen from vit_mci.ckpt",
            "roi_comparison_note": (
                "ROI runs see the full 3D ROI patch. "
                "The MobileNet baseline sees only 35 axial slices."
            ),
        }
    else:
        raise ValueError(experiment)
    dest = root / "metrics_vs_mobilenet.json"
    dest.write_text(json.dumps(payload, indent=2))
    print(f"wrote {dest}", flush=True)
    print(json.dumps(payload, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--field", choices=["whole", "roi"], default=None)
    p.add_argument("--experiment", choices=tuple(EXPERIMENTS), default="brainiac_vitb_frozen")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.audit_only:
        write_audit(FST_ROOT / "classifiers" / "brainiac_vitb_frozen" / "nifti_audit.json")
        return
    if args.checkpoint is None:
        default_name = "vit_mci.ckpt" if args.experiment == "brainiac_vit_mci_pdhead" else "BrainIAC.ckpt"
        args.checkpoint = FST_ROOT / "checkpoints" / default_name
    if args.compare:
        write_comparison(args.experiment)
        return
    if args.field is None:
        raise SystemExit("pass --field whole|roi, --audit-only, or --compare")
    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint {args.checkpoint}")
    labels = load_labels(META_CSV)
    run_field(args, labels)


if __name__ == "__main__":
    main()
