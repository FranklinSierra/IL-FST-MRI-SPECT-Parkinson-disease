#!/usr/bin/env bash
# NHY-stratified 5-fold FST on GPU 7. Trains up to 3 folds at a time (~7 GB each).
# Designed to run under nohup so an SSH drop does not kill training.
set -euo pipefail
FST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
export FST_GPU="${FST_GPU:-7}"
source "${FST_ROOT}/env.sh"

export FST_GPU=7
export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export FST_CLSFEAT_SPECT_CKPT="${FST_ROOT}/classifiers/classifiers_roi_4mm/spect/model.keras"

OUT="${FST_ROOT}/result/fst_clsfeat_mnet2d_v2_z78_112/ownKFold"
LOG_DIR="${FST_ROOT}/logs/ownKFold"
mkdir -p "${OUT}/files" "${OUT}/models" "${LOG_DIR}"

echo "=== $(date -Is) ownKFold GPU=${CUDA_VISIBLE_DEVICES} ==="
nvidia-smi -i 7 --query-gpu=name,memory.total,memory.used --format=csv,noheader || true

echo "=== splits ==="
python3 -u "${FST_ROOT}/scripts/make_ownkfold_nhy_splits.py" \
  --nhy-csv "${FST_ROOT}/data/nhy_distribution.csv" \
  --out-dir "${OUT}/files" \
  --folds 5 --seed 42

MEM_MB="${FST_GPU_MEMORY_MB:-7000}"
STAGGER_SEC="${FST_FOLD_STAGGER_SEC:-8}"

run_fold() {
  local k="$1"
  local log="${LOG_DIR}/fold${k}.log"
  echo "=== $(date -Is) start fold ${k} log=${log} ==="
  python3 -u "${FST_ROOT}/scripts/train_fst_ownkfold.py" \
    --fold "${k}" \
    --feat-cache "${FST_ROOT}/data/feat_cache_z78_112" \
    --files-dir "${OUT}/files" \
    --models-dir "${OUT}/models" \
    --nhy-csv "${FST_ROOT}/data/nhy_distribution.csv" \
    --spect-classifier "${FST_CLSFEAT_SPECT_CKPT}" \
    --z-min 78 --z-max 112 \
    --epochs 60 --patience 12 --batch-size 32 --lr 3e-4 \
    --mse-weight 0.3 --ce-weight 1.0 \
    --mid-channels 192 --l2 1e-4 --dropout 0.2 --seed 42 \
    --n-val 15 \
    --gpu-memory-mb "${MEM_MB}" \
    >"${log}" 2>&1
  local rc=$?
  echo "=== $(date -Is) fold ${k} exit=${rc} ==="
  return "${rc}"
}

run_batch() {
  local pids=()
  local ks=("$@")
  for k in "${ks[@]}"; do
    run_fold "${k}" &
    pids+=("$!")
    echo "launched fold ${k} pid=${pids[-1]}"
    sleep "${STAGGER_SEC}"
  done
  local fail=0
  local i=0
  for pid in "${pids[@]}"; do
    local k="${ks[$i]}"
    if ! wait "${pid}"; then
      echo "ERROR: fold ${k} pid=${pid} failed" >&2
      fail=1
    else
      echo "fold ${k} pid=${pid} ok"
    fi
    i=$((i + 1))
  done
  return "${fail}"
}

FAIL=0
echo "=== batch 1: folds 1 2 3 ==="
run_batch 1 2 3 || FAIL=1
echo "=== batch 2: folds 4 5 ==="
run_batch 4 5 || FAIL=1

if [[ "${FAIL}" -ne 0 ]]; then
  echo "ERROR: one or more folds failed; skip aggregate. See ${LOG_DIR}/fold*.log" >&2
  exit 1
fi

echo "=== aggregate ==="
python3 -u "${FST_ROOT}/scripts/train_fst_ownkfold.py" \
  --aggregate \
  --models-dir "${OUT}/models" \
  | tee "${LOG_DIR}/aggregate.log"

echo "=== $(date -Is) ownKFold done → ${OUT}/models ==="
ls -lh "${OUT}/models"/fold*/transfer_fold*.keras "${OUT}/models"/metrics_cv.json 2>/dev/null || true
