#!/usr/bin/env bash
# Cache conv_pw_13 features z=78..112 then train FST clsfeat v2 (GPU 6).
set -euo pipefail
FST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${FST_ROOT}/env.sh"

export FST_GPU="${FST_GPU:-6}"
export CUDA_VISIBLE_DEVICES="${FST_GPU}"
export PYTHONUNBUFFERED=1

export FST_ZMIN=78
export FST_ZMAX=112
export FST_ZSTEP=1
export FST_FEAT_CACHE_Z="${FST_ROOT}/data/feat_cache_z78_112"
OUT="${FST_ROOT}/result/fst_clsfeat_mnet2d_v2_z78_112"
LOG_DIR="${FST_ROOT}/logs"
mkdir -p "${FST_FEAT_CACHE_Z}" "${OUT}" "${LOG_DIR}"

LOG_CACHE="${LOG_DIR}/cache_clsfeat_z78_112.log"
LOG_TRAIN="${LOG_DIR}/train_fst_clsfeat_v2_z78_112.log"

echo "=== 1/2 cache features → ${FST_FEAT_CACHE_Z} ==="
python3 -u "${FST_ROOT}/scripts/cache_clsfeat_z.py" >"${LOG_CACHE}" 2>&1
echo "cache done"

echo "=== 2/2 train FST clsfeat v2 → ${OUT} ==="
python3 -u "${FST_ROOT}/scripts/train_fst_clsfeat_v2.py" \
  --feat-cache "${FST_FEAT_CACHE_Z}" \
  --out-dir "${OUT}" \
  --spect-classifier "${FST_CLSFEAT_SPECT_CKPT}" \
  --z-min 78 --z-max 112 \
  --epochs 60 --patience 12 \
  >"${LOG_TRAIN}" 2>&1
echo "train done → ${OUT}"
