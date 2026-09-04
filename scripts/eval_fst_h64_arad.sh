#!/usr/bin/env bash
# Re-run official val/test eval from result/fst_h64_arad/best.pt (GPU 6).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=env.sh
source "$ROOT/env.sh"

CKPT="${1:-${FST_OUT_DIR}/best.pt}"
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: missing checkpoint: $CKPT" >&2
  exit 1
fi

echo "=== FST eval GPU=${FST_GPU} ckpt=${CKPT} ==="
python3 scripts/train_fst_h64_arad.py --eval-only --ckpt "$CKPT"
echo "Done: ${FST_OUT_DIR}/metrics_test.json"
