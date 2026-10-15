#!/usr/bin/env python3
"""Fetch BrainIAC ViT-B weights into checkpoints/BrainIAC.ckpt.

The official Dropbox folder is one 7.4 GB zip. Its members are stored with
data descriptors and the host ignores HTTP Range, so a single checkpoint
cannot be pulled out without keeping the whole archive. This machine does not
have that much free disk. The same MONAI ViT-B weights are published as
backbone.safetensors (token-0 readout, 768-d) and are wrapped here in the
state_dict layout BrainIAC's loader expects (keys prefixed with ``backbone.``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import Request, urlopen

import torch

HF_URL = "https://huggingface.co/eugenehp/brainiac/resolve/main/backbone.safetensors"
CHUNK = 1024 * 1024


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    req = Request(url, headers={"User-Agent": "IL-FST-brainiac/1.0"})
    with urlopen(req, timeout=300) as resp, tmp.open("wb") as out:
        total = resp.headers.get("Content-Length")
        total_n = int(total) if total and total.isdigit() else None
        got = 0
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            out.write(chunk)
            got += len(chunk)
            if total_n and got % (20 * CHUNK) < CHUNK:
                print(f"  {got / 1e6:.0f} / {total_n / 1e6:.0f} MB", flush=True)
    tmp.replace(dest)
    print(f"downloaded {dest} ({dest.stat().st_size} bytes)", flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ckpt_path = root / "checkpoints" / "BrainIAC.ckpt"
    if ckpt_path.is_file() and ckpt_path.stat().st_size > 1_000_000:
        print(f"checkpoint already present: {ckpt_path}", flush=True)
        return
    try:
        from safetensors.torch import load_file
    except ImportError:
        print("ERROR: pip install safetensors", file=sys.stderr)
        raise
    raw_path = root / "checkpoints" / "backbone.safetensors"
    if not raw_path.is_file():
        print(f"downloading {HF_URL}", flush=True)
        _download(HF_URL, raw_path)
    weights = load_file(str(raw_path))
    state = {f"backbone.{key}": value for key, value in weights.items()}
    payload = {
        "state_dict": state,
        "source": HF_URL,
        "note": (
            "MONAI ViT-B weights converted from backbone.safetensors. "
            "Official Dropbox zip is 7.4GB and cannot be partially fetched here."
        ),
    }
    torch.save(payload, ckpt_path)
    n_params = sum(v.numel() for v in weights.values())
    print(f"wrote {ckpt_path} tensors={len(weights)} params={n_params}", flush=True)


if __name__ == "__main__":
    main()
