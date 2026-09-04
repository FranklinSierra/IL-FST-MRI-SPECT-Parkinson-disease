#!/usr/bin/env bash
# Re-eval fst_clsfeat_mnet2d_arad on val holdout + official test (GPU).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=env.sh
source "$ROOT/env.sh"

export FST_CLSFEAT_OUT_DIR="${FST_CLSFEAT_OUT_DIR:-${FST_ROOT}/result/fst_clsfeat_mnet2d_arad}"
export FST_CLSFEAT_CACHE="${FST_CLSFEAT_CACHE:-${FST_ROOT}/data/cache_slices_roi4mm}"
export FST_TF_FORCE_GPU=1

python3 -u scripts/train_fst_clsfeat_mnet2d_arad.py --eval-only "$@"
