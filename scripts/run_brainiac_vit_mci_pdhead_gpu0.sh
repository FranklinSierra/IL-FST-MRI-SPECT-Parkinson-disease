#!/usr/bin/env bash
# PD MLP head on the frozen MCI-finetuned ViT. Physical GPU 0.
# Does not source env.sh (that would pin FST_GPU, default 6).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
mkdir -p logs checkpoints classifiers/brainiac_vit_mci_pdhead

export CUDA_VISIBLE_DEVICES=0

CKPT="${ROOT}/checkpoints/vit_mci.ckpt"
if [[ ! -f "${CKPT}" ]]; then
  echo "missing ${CKPT}" >&2
  exit 1
fi

python3 -u scripts/train_brainiac_frozen_probe.py \
  --experiment brainiac_vit_mci_pdhead \
  --checkpoint "${CKPT}" \
  --field whole
python3 -u scripts/train_brainiac_frozen_probe.py \
  --experiment brainiac_vit_mci_pdhead \
  --checkpoint "${CKPT}" \
  --field roi
python3 -u scripts/train_brainiac_frozen_probe.py \
  --experiment brainiac_vit_mci_pdhead \
  --compare
