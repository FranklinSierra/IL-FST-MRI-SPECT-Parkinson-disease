#!/usr/bin/env python3
"""Train ARAD-style FST UNet: MRI h64 → SPECT h64 + decode + MobileNet3D CE."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

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
from torch.utils.data import DataLoader
from tqdm import tqdm

FST_ROOT = Path(__file__).resolve().parents[1]
ILCLDM_ROOT = Path(os.environ.get("ILCLDM_ROOT", "/workspace/IL-CLDM")).resolve()


def _bootstrap_paths() -> None:
    # IL-CLDM owns `code.*`; FST owns `fst.*`.
    for p in (str(FST_ROOT), str(ILCLDM_ROOT)):
        if p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(ILCLDM_ROOT))
    sys.path.insert(0, str(FST_ROOT))
    mod = sys.modules.get("code")
    if mod is not None and (
        not hasattr(mod, "__path__")
        or "IL-CLDM" not in str(getattr(mod, "__file__", "") or "")
    ):
        to_del = [k for k in sys.modules if k == "code" or k.startswith("code.")]
        for k in to_del:
            del sys.modules[k]


def refuse_paths(spect_root: str, spect_aae: str, mri_ckpt: str, mnet_ckpt: str) -> None:
    sr = spect_root.replace("\\", "/")
    if "whole_Abeta_4mm" not in sr:
        raise SystemExit(f"ERROR: expected whole_Abeta_4mm, got {spect_root}")
    if "spect_rois_4mm" not in spect_aae.replace("\\", "/"):
        raise SystemExit(f"ERROR: expected spect_rois_4mm AAE, got {spect_aae}")
    if "mri_rois_4mm_xmodal_h64" not in mri_ckpt.replace("\\", "/"):
        raise SystemExit(f"ERROR: expected mri_rois_4mm_xmodal_h64, got {mri_ckpt}")
    if "cls_spect_aae_mnet3d" not in mnet_ckpt.replace("\\", "/"):
        raise SystemExit(f"ERROR: expected cls_spect_aae_mnet3d, got {mnet_ckpt}")
    for p, label in (
        (spect_root, "SPECT root"),
        (spect_aae, "SPECT AAE"),
        (mri_ckpt, "MRI xmodal"),
        (mnet_ckpt, "MobileNet3D"),
    ):
        path = Path(p)
        if label == "SPECT root":
            if not path.is_dir():
                raise SystemExit(f"ERROR: missing {label}: {p}")
        elif not path.is_file():
            raise SystemExit(f"ERROR: missing {label}: {p}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spect-root", default=os.environ.get("ILCLDM_WHOLE_ABETA", ""))
    p.add_argument("--mri-root", default=os.environ.get("ILCLDM_WHOLE_MRI", ""))
    p.add_argument("--spect-aae", default=os.environ.get("FST_SPECT_AAE", ""))
    p.add_argument("--mri-xmodal", default=os.environ.get("FST_MRI_XMODAL", ""))
    p.add_argument("--mobilenet3d", default=os.environ.get("FST_MOBILENET3D", ""))
    p.add_argument("--out-dir", type=Path, default=Path(os.environ.get("FST_OUT_DIR", "result/fst_h64_arad")))
    p.add_argument("--train-list", type=Path, default=Path(os.environ.get("FST_TRAIN_LIST", "")))
    p.add_argument("--val-list", type=Path, default=Path(os.environ.get("FST_VAL_LIST", "")))
    p.add_argument("--test-list", type=Path, default=Path(os.environ.get("FST_TEST_LIST", "")))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--mse-weight", type=float, default=0.2)
    p.add_argument("--ce-weight", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Load best.pt and evaluate val/test (official lists); no training.",
    )
    p.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Checkpoint for --eval-only (default: <out-dir>/best.pt).",
    )
    return p.parse_args()


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def load_aae(checkpoint: str, device: torch.device):
    import config
    from model import AAE
    from utils import load_checkpoint

    model = AAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
    load_checkpoint(checkpoint, model, opt, config.learning_rate)
    freeze_module(model)
    return model


def load_mobilenet3d(checkpoint: str, device: torch.device):
    from code.mobilenet3d import build_mobilenet_v2_3d

    model = build_mobilenet_v2_3d(num_classes=2, in_channels=1, pretrained=False).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    freeze_module(model)
    return model


@torch.no_grad()
def encode_h64(aae: nn.Module, volume: torch.Tensor) -> torch.Tensor:
    _z, h64 = aae.encoder(volume, return_hidden=True)
    return h64


def subject_metrics(y_true, y_pred, y_prob) -> dict:
    out = {
        "n_subjects": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def run_smoke(args, device) -> None:
    from code.diffusion_features import decode_h64_via_bottleneck
    from fst.transfer_unet import build_transfer_unet
    from code.volume_shapes import get_latent_shape, get_working_volume_shape
    from dataset import TwoDataset

    spect_aae = load_aae(args.spect_aae, device)
    mri_aae = load_aae(args.mri_xmodal, device)
    mnet = load_mobilenet3d(args.mobilenet3d, device)
    unet = build_transfer_unet().to(device)

    ds = TwoDataset(
        root_MRI=args.mri_root,
        root_Abeta=args.spect_root,
        task=str(args.train_list),
        stage="validation",
    )
    mri, spect, name, label = ds[0]
    mri_t = torch.as_tensor(mri, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    spect_t = torch.as_tensor(spect, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    h64_m = encode_h64(mri_aae, mri_t)
    h64_s = encode_h64(spect_aae, spect_t)
    expected = (1, 64, *get_latent_shape())
    if tuple(h64_m.shape) != expected or tuple(h64_s.shape) != expected:
        raise RuntimeError(f"h64 shape mismatch: mri={tuple(h64_m.shape)} spect={tuple(h64_s.shape)} expected={expected}")

    h64_hat = unet(h64_m)
    recon = decode_h64_via_bottleneck(spect_aae, h64_hat)
    logits = mnet(recon)
    working = (1, 1, *get_working_volume_shape())
    if tuple(recon.shape) != working:
        raise RuntimeError(f"recon shape {tuple(recon.shape)} != {working}")
    if tuple(logits.shape) != (1, 2):
        raise RuntimeError(f"logits shape {tuple(logits.shape)}")

    for mod, tag in ((spect_aae, "spect"), (mri_aae, "mri"), (mnet, "mnet")):
        if any(p.requires_grad for p in mod.parameters()):
            raise RuntimeError(f"{tag} has trainable params")

    print(
        f"SMOKE OK working={get_working_volume_shape()} latent={get_latent_shape()} "
        f"h64={tuple(h64_m.shape)} recon={tuple(recon.shape)} logits={tuple(logits.shape)} "
        f"id={name} label={int(label[0])}"
    )


@torch.no_grad()
def evaluate_split(
    *,
    loader,
    mri_aae,
    spect_aae,
    unet,
    mnet,
    device,
    mse_weight: float,
    ce_weight: float,
):
    from code.diffusion_features import decode_h64_via_bottleneck
    from main import batch_metric_scores

    unet.eval()
    criterion = nn.CrossEntropyLoss()
    rows = []
    mse_total = 0.0
    psnr_total = 0.0
    ssim_total = 0.0
    loss_total = 0.0
    n = 0
    for mri, spect, names, labels in loader:
        mri = torch.as_tensor(np.asarray(mri), dtype=torch.float32, device=device).unsqueeze(1)
        spect = torch.as_tensor(np.asarray(spect), dtype=torch.float32, device=device).unsqueeze(1)
        y = labels.reshape(-1).long().to(device)

        h64_m = encode_h64(mri_aae, mri)
        h64_s = encode_h64(spect_aae, spect)
        h64_hat = unet(h64_m)
        mse = torch.nn.functional.mse_loss(h64_hat, h64_s)
        recon = decode_h64_via_bottleneck(spect_aae, h64_hat)
        logits = mnet(recon)
        ce = criterion(logits, y)
        loss = mse_weight * mse + ce_weight * ce

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        y_np = y.cpu().numpy()
        bpsnr, bssim = batch_metric_scores(spect, recon)
        bsz = mri.shape[0]
        mse_total += float(mse.item()) * bsz
        psnr_total += float(bpsnr)
        ssim_total += float(bssim)
        loss_total += float(loss.item()) * bsz
        n += bsz
        for i, sid in enumerate(names):
            rows.append(
                {
                    "ID": sid if isinstance(sid, str) else str(sid),
                    "label": int(y_np[i]),
                    "pred": int(preds[i]),
                    "prob_1": float(probs[i, 1]),
                }
            )

    df = pd.DataFrame(rows).sort_values("ID").reset_index(drop=True)
    metrics = subject_metrics(df["label"].to_numpy(), df["pred"].to_numpy(), df["prob_1"].to_numpy())
    metrics.update(
        {
            "mse_h64": mse_total / max(n, 1),
            "psnr": psnr_total / max(n, 1),
            "ssim": ssim_total / max(n, 1),
            "loss": loss_total / max(n, 1),
        }
    )
    return metrics, df


def main():
    _bootstrap_paths()
    args = parse_args()
    refuse_paths(args.spect_root, args.spect_aae, args.mri_xmodal, args.mobilenet3d)

    from utils import seed_torch

    seed_torch(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke_only:
        run_smoke(args, device)
        return

    from code.diffusion_features import decode_h64_via_bottleneck
    from fst.transfer_unet import build_transfer_unet
    from code.volume_shapes import get_latent_shape, get_working_volume_shape
    from dataset import TwoDataset

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spect_aae = load_aae(args.spect_aae, device)
    mri_aae = load_aae(args.mri_xmodal, device)
    mnet = load_mobilenet3d(args.mobilenet3d, device)
    unet = build_transfer_unet().to(device)

    def make_loader(list_path: Path, shuffle: bool) -> DataLoader:
        ds = TwoDataset(
            root_MRI=args.mri_root,
            root_Abeta=args.spect_root,
            task=str(list_path),
            stage="train" if shuffle else "validation",
        )
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=shuffle,
        )

    val_loader = make_loader(args.val_list, shuffle=False)
    test_loader = make_loader(args.test_list, shuffle=False)

    if args.eval_only:
        ckpt_path = args.ckpt or (out_dir / "best.pt")
        if not ckpt_path.is_file():
            raise SystemExit(f"ERROR: missing checkpoint for eval: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        unet.load_state_dict(ckpt["unet_state_dict"])
        best_epoch = int(ckpt.get("epoch", -1))
        val_metrics, val_df = evaluate_split(
            loader=val_loader,
            mri_aae=mri_aae,
            spect_aae=spect_aae,
            unet=unet,
            mnet=mnet,
            device=device,
            mse_weight=args.mse_weight,
            ce_weight=args.ce_weight,
        )
        test_metrics, test_df = evaluate_split(
            loader=test_loader,
            mri_aae=mri_aae,
            spect_aae=spect_aae,
            unet=unet,
            mnet=mnet,
            device=device,
            mse_weight=args.mse_weight,
            ce_weight=args.ce_weight,
        )
        val_df.to_csv(out_dir / "val_predictions.csv", index=False)
        test_df.to_csv(out_dir / "test_predictions.csv", index=False)
        report = classification_report(test_df["label"], test_df["pred"], digits=4, zero_division=0)
        (out_dir / "classification_report_test.txt").write_text(report)
        cm = confusion_matrix(test_df["label"], test_df["pred"])
        pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
            out_dir / "confusion_matrix_test.csv"
        )
        metrics_valid = {"best_epoch": best_epoch, **val_metrics}
        metrics_test = {"best_epoch": best_epoch, **test_metrics}
        (out_dir / "metrics_valid.json").write_text(json.dumps(metrics_valid, indent=2))
        (out_dir / "metrics_test.json").write_text(json.dumps(metrics_test, indent=2))
        print(report)
        print(json.dumps(metrics_test, indent=2))
        print(f"Eval done: {ckpt_path}")
        return

    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_loader = make_loader(args.train_list, shuffle=True)

    config_payload = {
        "spect_root": args.spect_root,
        "mri_root": args.mri_root,
        "spect_aae": args.spect_aae,
        "mri_xmodal": args.mri_xmodal,
        "mobilenet3d": args.mobilenet3d,
        "out_dir": str(out_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "mse_weight": args.mse_weight,
        "ce_weight": args.ce_weight,
        "patience": args.patience,
        "seed": args.seed,
        "working_volume_shape": list(get_working_volume_shape()),
        "latent_h64_shape": [64, *get_latent_shape()],
        "early_stop_metric": "val_balanced_accuracy",
        "model": "FeatureTransferUNet_64to64",
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    (out_dir / "config.json").write_text(json.dumps(config_payload, indent=2))
    print(json.dumps(config_payload, indent=2))

    history = []
    best_val_ba = -1.0
    best_epoch = -1
    patience_left = args.patience
    best_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        unet.train()
        running = 0.0
        running_mse = 0.0
        running_ce = 0.0
        n = 0
        t0 = time.time()
        loop = tqdm(train_loader, desc=f"fst:epoch_{epoch}", leave=True)
        for mri, spect, _names, labels in loop:
            mri = torch.as_tensor(np.asarray(mri), dtype=torch.float32, device=device).unsqueeze(1)
            spect = torch.as_tensor(np.asarray(spect), dtype=torch.float32, device=device).unsqueeze(1)
            y = labels.reshape(-1).long().to(device)

            with torch.no_grad():
                h64_m = encode_h64(mri_aae, mri)
                h64_s = encode_h64(spect_aae, spect)

            h64_hat = unet(h64_m)
            mse = torch.nn.functional.mse_loss(h64_hat, h64_s)
            recon = decode_h64_via_bottleneck(spect_aae, h64_hat)
            logits = mnet(recon)
            ce = criterion(logits, y)
            loss = args.mse_weight * mse + args.ce_weight * ce

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bsz = mri.shape[0]
            running += float(loss.item()) * bsz
            running_mse += float(mse.item()) * bsz
            running_ce += float(ce.item()) * bsz
            n += bsz
            loop.set_postfix(loss=running / max(n, 1), mse=running_mse / max(n, 1), ce=running_ce / max(n, 1))

        train_loss = running / max(n, 1)
        val_metrics, _ = evaluate_split(
            loader=val_loader,
            mri_aae=mri_aae,
            spect_aae=spect_aae,
            unet=unet,
            mnet=mnet,
            device=device,
            mse_weight=args.mse_weight,
            ce_weight=args.ce_weight,
        )
        val_ba = val_metrics["balanced_accuracy"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mse": running_mse / max(n, 1),
            "train_ce": running_ce / max(n, 1),
            "val_balanced_accuracy": val_ba,
            "val_roc_auc": val_metrics["roc_auc"],
            "val_mse_h64": val_metrics["mse_h64"],
            "val_psnr": val_metrics["psnr"],
            "val_ssim": val_metrics["ssim"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs} loss={train_loss:.4f} "
            f"val_BA={val_ba:.4f} val_auc={val_metrics['roc_auc']:.4f} "
            f"val_mse={val_metrics['mse_h64']:.4f} val_psnr={val_metrics['psnr']:.2f} "
            f"({row['seconds']:.1f}s)"
        )

        improved = np.isfinite(val_ba) and val_ba > best_val_ba + 1e-6
        if improved:
            best_val_ba = float(val_ba)
            best_epoch = epoch
            patience_left = args.patience
            torch.save(
                {
                    "epoch": epoch,
                    "unet_state_dict": unet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_balanced_accuracy": best_val_ba,
                    "config": config_payload,
                },
                best_path,
            )
            print(f"New best checkpoint at epoch {epoch} (val_BA={best_val_ba:.4f})")
        else:
            patience_left -= 1
            print(f"early-stop counter: {args.patience - patience_left}/{args.patience}")
            if patience_left <= 0:
                print(f"Early stop at epoch {epoch} (best={best_epoch}, val_BA={best_val_ba:.4f})")
                break

    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    unet.load_state_dict(ckpt["unet_state_dict"])

    val_metrics, val_df = evaluate_split(
        loader=val_loader,
        mri_aae=mri_aae,
        spect_aae=spect_aae,
        unet=unet,
        mnet=mnet,
        device=device,
        mse_weight=args.mse_weight,
        ce_weight=args.ce_weight,
    )
    test_metrics, test_df = evaluate_split(
        loader=test_loader,
        mri_aae=mri_aae,
        spect_aae=spect_aae,
        unet=unet,
        mnet=mnet,
        device=device,
        mse_weight=args.mse_weight,
        ce_weight=args.ce_weight,
    )
    val_df.to_csv(out_dir / "val_predictions.csv", index=False)
    test_df.to_csv(out_dir / "test_predictions.csv", index=False)

    report = classification_report(test_df["label"], test_df["pred"], digits=4, zero_division=0)
    (out_dir / "classification_report_test.txt").write_text(report)
    cm = confusion_matrix(test_df["label"], test_df["pred"])
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        out_dir / "confusion_matrix_test.csv"
    )

    metrics_valid = {"best_epoch": best_epoch, "best_val_balanced_accuracy": best_val_ba, **val_metrics}
    metrics_test = {"best_epoch": best_epoch, **test_metrics}
    (out_dir / "metrics_valid.json").write_text(json.dumps(metrics_valid, indent=2))
    (out_dir / "metrics_test.json").write_text(json.dumps(metrics_test, indent=2))

    print("\n=== VAL ===")
    print(json.dumps(metrics_valid, indent=2))
    print("\n=== TEST ===")
    print(report)
    print(json.dumps(metrics_test, indent=2))
    if test_metrics["balanced_accuracy"] <= 0.5 + 1e-6:
        print("GATE: test BA not clearly above chance on translated recons.")
    else:
        print(
            f"GATE: test BA={test_metrics['balanced_accuracy']:.4f} > 0.5 "
            "(compare vs AAE-recon classifier ~0.975)."
        )
    print(f"Done: {best_path}")


if __name__ == "__main__":
    main()
