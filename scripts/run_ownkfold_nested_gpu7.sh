#!/usr/bin/env bash
# Nested 5-fold FST: retrain MRI + SPECT gates per fold, recache, then UNet.
# Same subject indices as ownKFold/files and ownKFold/models/fold{k}/{train_fit,val_holdout}.txt
# Sequential folds on GPU 7. Does not write into ownKFold/models or feat_cache_z78_112.
set -euo pipefail
FST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FST_GPU="${FST_GPU:-7}"
# shellcheck disable=SC1091
source "${FST_ROOT}/env.sh"

export FST_GPU=7
export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1
export TF_FORCE_GPU_ALLOW_GROWTH=true

OWN="${FST_ROOT}/result/fst_clsfeat_mnet2d_v2_z78_112/ownKFold"
NESTED="${FST_ROOT}/result/fst_clsfeat_mnet2d_v2_z78_112/ownKFold_nested"
CACHE_ROOT="${FST_ROOT}/data/feat_cache_ownkfold_nested"
LOG_DIR="${FST_ROOT}/logs/ownKFold_nested"
mkdir -p "${NESTED}/models" "${CACHE_ROOT}" "${LOG_DIR}"

python3 - <<PY
import json
from pathlib import Path
p = Path("${NESTED}") / "experiment.json"
p.write_text(json.dumps({
    "experiment": "ownKFold_nested",
    "gpu": 7,
    "splits_source": str(Path("${OWN}") / "files"),
    "inner_val_source": str(Path("${OWN}") / "models" / "fold{k}"),
    "does_not_modify": [
        "result/fst_clsfeat_mnet2d_v2_z78_112/ownKFold/",
        "data/feat_cache_z78_112/",
        "classifiers/classifiers_roi_4mm/",
    ],
    "protocol": "Per fold: train MRI+SPECT on train_fit, early-stop on val_holdout, cache features, train UNet, eval testFold.",
}, indent=2) + "\n")
print("wrote", p)
PY

echo "=== $(date -Is) ownKFold_nested GPU=${CUDA_VISIBLE_DEVICES} ==="
nvidia-smi -i 7 --query-gpu=name,memory.total,memory.used --format=csv,noheader || true

run_fold() {
  local k="$1"
  local fold_log="${LOG_DIR}/fold${k}.log"
  local files="${OWN}/files"
  local split="${OWN}/models/fold${k}"
  local out="${NESTED}/models/fold${k}"
  local cache="${CACHE_ROOT}/fold${k}"
  local mri_ckpt="${out}/mri/model.keras"
  local spect_ckpt="${out}/spect/model.keras"
  local xfer="${out}/transfer_fold${k}.keras"
  local metrics="${out}/metrics_fold${k}.json"

  mkdir -p "${out}/mri" "${out}/spect" "${cache}"
  echo "=== $(date -Is) NESTED fold ${k} ===" | tee -a "${fold_log}"

  if [[ -f "${xfer}" && -f "${metrics}" ]]; then
    echo "fold ${k}: transfer already done, skip" | tee -a "${fold_log}"
    return 0
  fi

  if [[ ! -f "${split}/train_fit.txt" || ! -f "${split}/val_holdout.txt" ]]; then
    echo "ERROR: missing ${split}/train_fit.txt or val_holdout.txt" | tee -a "${fold_log}"
    return 1
  fi

  if [[ ! -f "${mri_ckpt}" ]]; then
    echo "=== $(date -Is) fold ${k} train MRI ===" | tee -a "${fold_log}"
    python3 -u "${FST_ROOT}/scripts/train_keras_mobilenet_roi_classifier.py" \
      --modality mri \
      --volume-root "${ILCLDM_WHOLE_MRI}" \
      --out-dir "${out}/mri" \
      --train-list "${files}/trainFold${k}.txt" \
      --test-list "${files}/testFold${k}.txt" \
      --fit-list "${split}/train_fit.txt" \
      --val-list "${split}/val_holdout.txt" \
      --meta-csv "${FST_META_CSV}" \
      --z-min 78 --z-max 112 \
      --epochs "${EPOCHS:-50}" --batch-size "${BATCH:-12}" --patience "${PATIENCE:-10}" \
      --seed 42 --lr 1e-4 \
      >>"${fold_log}" 2>&1
  else
    echo "fold ${k}: MRI checkpoint exists, skip train" | tee -a "${fold_log}"
  fi

  if [[ ! -f "${spect_ckpt}" ]]; then
    echo "=== $(date -Is) fold ${k} train SPECT ===" | tee -a "${fold_log}"
    python3 -u "${FST_ROOT}/scripts/train_keras_mobilenet_roi_classifier.py" \
      --modality spect \
      --volume-root "${ILCLDM_WHOLE_ABETA}" \
      --out-dir "${out}/spect" \
      --train-list "${files}/trainFold${k}.txt" \
      --test-list "${files}/testFold${k}.txt" \
      --fit-list "${split}/train_fit.txt" \
      --val-list "${split}/val_holdout.txt" \
      --meta-csv "${FST_META_CSV}" \
      --z-min 78 --z-max 112 \
      --epochs "${EPOCHS:-50}" --batch-size "${BATCH:-12}" --patience "${PATIENCE:-10}" \
      --seed 42 --lr 1e-4 \
      >>"${fold_log}" 2>&1
  else
    echo "fold ${k}: SPECT checkpoint exists, skip train" | tee -a "${fold_log}"
  fi

  if [[ ! -f "${cache}/f_mri.npy" || ! -f "${cache}/gap_spect.npy" || ! -f "${cache}/index.csv" ]]; then
    echo "=== $(date -Is) fold ${k} cache features ===" | tee -a "${fold_log}"
    python3 -u "${FST_ROOT}/scripts/cache_clsfeat_fold.py" \
      --out-dir "${cache}" \
      --mri-ckpt "${mri_ckpt}" \
      --spect-ckpt "${spect_ckpt}" \
      --train-list "${files}/trainFold${k}.txt" \
      --test-list "${files}/testFold${k}.txt" \
      --meta-csv "${FST_META_CSV}" \
      --mri-root "${ILCLDM_WHOLE_MRI}" \
      --spect-root "${ILCLDM_WHOLE_ABETA}" \
      --z-min 78 --z-max 112 --z-step 1 \
      >>"${fold_log}" 2>&1
  else
    echo "fold ${k}: feature cache exists, skip" | tee -a "${fold_log}"
  fi

  echo "=== $(date -Is) fold ${k} train UNet ===" | tee -a "${fold_log}"
  python3 -u "${FST_ROOT}/scripts/train_fst_ownkfold.py" \
    --fold "${k}" \
    --feat-cache "${cache}" \
    --files-dir "${files}" \
    --models-dir "${NESTED}/models" \
    --nhy-csv "${FST_ROOT}/data/nhy_distribution.csv" \
    --spect-classifier "${spect_ckpt}" \
    --fit-list "${split}/train_fit.txt" \
    --val-list "${split}/val_holdout.txt" \
    --z-min 78 --z-max 112 \
    --epochs 60 --patience 12 --batch-size 32 --lr 3e-4 \
    --mse-weight 0.3 --ce-weight 1.0 \
    --mid-channels 192 --l2 1e-4 --dropout 0.2 --seed 42 \
    --n-val 15 \
    >>"${fold_log}" 2>&1

  echo "=== $(date -Is) fold ${k} done ===" | tee -a "${fold_log}"
}

for k in 1 2 3 4 5; do
  run_fold "${k}"
done

echo "=== $(date -Is) aggregate ==="
python3 -u "${FST_ROOT}/scripts/train_fst_ownkfold.py" \
  --aggregate \
  --models-dir "${NESTED}/models" \
  | tee "${LOG_DIR}/aggregate.log"

echo "=== $(date -Is) ownKFold_nested done → ${NESTED}/models ==="
ls -lh "${NESTED}/models"/fold*/transfer_fold*.keras "${NESTED}/models"/metrics_cv.json 2>/dev/null || true
