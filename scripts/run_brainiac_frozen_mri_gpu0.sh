#!/usr/bin/env bash
# Frozen BrainIAC MRI probes on physical GPU 0.
# Does not source env.sh (that would pin FST_GPU, default 6).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
mkdir -p logs checkpoints classifiers/brainiac_vitb_frozen

export CUDA_VISIBLE_DEVICES=0

python3 -u scripts/download_brainiac_ckpt.py
python3 -u scripts/train_brainiac_frozen_probe.py --audit-only
python3 -u scripts/train_brainiac_frozen_probe.py --field whole
python3 -u scripts/train_brainiac_frozen_probe.py --field roi
python3 -u scripts/train_brainiac_frozen_probe.py --compare
