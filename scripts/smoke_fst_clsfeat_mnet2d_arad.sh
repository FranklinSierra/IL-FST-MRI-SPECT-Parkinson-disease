#!/usr/bin/env bash
# Smoke: freeze + shapes for fst_clsfeat_mnet2d_arad (GPU by default).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=env.sh
source "$ROOT/env.sh"

export FST_CLSFEAT_OUT_DIR="${FST_CLSFEAT_OUT_DIR:-${FST_ROOT}/result/fst_clsfeat_mnet2d_arad}"
export FST_CLSFEAT_CACHE="${FST_CLSFEAT_CACHE:-${FST_ROOT}/data/cache_slices_roi4mm}"
export FST_CLSFEAT_MRI_CKPT="${FST_CLSFEAT_MRI_CKPT:-${FST_ROOT}/models/classifiers_roi_4mm/mri/model.keras}"
export FST_CLSFEAT_SPECT_CKPT="${FST_CLSFEAT_SPECT_CKPT:-${FST_ROOT}/models/classifiers_roi_4mm/spect/model.keras}"
export FST_TF_FORCE_GPU=1

python3 scripts/train_fst_clsfeat_mnet2d_arad.py --smoke-only
