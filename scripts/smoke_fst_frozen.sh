#!/usr/bin/env bash
# Smoke frozen encoders + decode + MobileNet3D + TransferUNet shapes (GPU 6 default).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=env.sh
source "$ROOT/env.sh"

echo "=== FST smoke GPU=${FST_GPU} ==="
python3 scripts/train_fst_h64_arad.py --smoke-only
echo "Smoke done."
