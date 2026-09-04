#!/usr/bin/env bash
# Train MRI + SPECT PyTorch MobileNetV2 classifiers in parallel on GPU 6.
# Slice range [78, 112] (35 axial tiles). Shared 15-subject stratified val holdout.
set -euo pipefail

FST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${FST_ROOT}/env.sh"

export FST_GPU="${FST_GPU:-6}"
export CUDA_VISIBLE_DEVICES="${FST_GPU}"

OUT_MRI="${FST_ROOT}/models/classifiers_roi_4mm/mri"
OUT_SPECT="${FST_ROOT}/models/classifiers_roi_4mm/spect"
SPLIT_DIR="${FST_ROOT}/models/classifiers_roi_4mm/_shared_split_z78_112"
LOG_DIR="${FST_ROOT}/logs"
mkdir -p "${OUT_MRI}" "${OUT_SPECT}" "${SPLIT_DIR}" "${LOG_DIR}"

Z_MIN=78
Z_MAX=112
N_VAL=15
SEED=42
EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:-12}"
PATIENCE="${PATIENCE:-15}"

# Shared stratified holdout (same subjects for MRI and SPECT)
python3 - <<PY
import sys
from pathlib import Path
sys.path.insert(0, "${FST_ROOT}")
from fst.slice_data import load_id_list, load_labels, stratified_holdout, write_id_list

train = load_id_list("${FST_TRAIN_LIST}")
labels = load_labels("${FST_META_CSV}")
fit, val = stratified_holdout(train, labels, n_val=${N_VAL}, seed=${SEED})
split = Path("${SPLIT_DIR}")
write_id_list(split / "train_fit.txt", fit)
write_id_list(split / "val_holdout.txt", val)
write_id_list(split / "test.txt", load_id_list("${FST_TEST_LIST}"))
vc = sum(1 for s in val if labels[s] == 0)
vp = sum(1 for s in val if labels[s] == 1)
print(f"shared split: fit={len(fit)} val={len(val)} (control={vc} PD={vp})")
PY

COMMON=(
  --train-list "${FST_TRAIN_LIST}"
  --test-list "${FST_TEST_LIST}"
  --meta-csv "${FST_META_CSV}"
  --fit-list "${SPLIT_DIR}/train_fit.txt"
  --val-list "${SPLIT_DIR}/val_holdout.txt"
  --z-min "${Z_MIN}"
  --z-max "${Z_MAX}"
  --n-val-holdout "${N_VAL}"
  --seed "${SEED}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH}"
  --patience "${PATIENCE}"
  --num-workers 2
)

LOG_MRI="${LOG_DIR}/train_mobilenet2d_mri_z78_112.log"
LOG_SPECT="${LOG_DIR}/train_mobilenet2d_spect_z78_112.log"

echo "Launching MRI + SPECT on GPU ${FST_GPU} (batch=${BATCH})"
echo "  MRI log:    ${LOG_MRI}"
echo "  SPECT log:  ${LOG_SPECT}"

nohup env CUDA_VISIBLE_DEVICES="${FST_GPU}" PYTHONUNBUFFERED=1 \
  python3 "${FST_ROOT}/scripts/train_mobilenet2d_roi_classifier.py" \
  --modality mri \
  --volume-root "${ILCLDM_WHOLE_MRI}" \
  --out-dir "${OUT_MRI}" \
  "${COMMON[@]}" \
  >"${LOG_MRI}" 2>&1 &
PID_MRI=$!

nohup env CUDA_VISIBLE_DEVICES="${FST_GPU}" PYTHONUNBUFFERED=1 \
  python3 "${FST_ROOT}/scripts/train_mobilenet2d_roi_classifier.py" \
  --modality spect \
  --volume-root "${ILCLDM_WHOLE_ABETA}" \
  --out-dir "${OUT_SPECT}" \
  "${COMMON[@]}" \
  >"${LOG_SPECT}" 2>&1 &
PID_SPECT=$!

echo "${PID_MRI}" > "${LOG_DIR}/train_mobilenet2d_mri_z78_112.pid"
echo "${PID_SPECT}" > "${LOG_DIR}/train_mobilenet2d_spect_z78_112.pid"
echo "PIDs: MRI=${PID_MRI} SPECT=${PID_SPECT}"
echo "Monitor: tail -f ${LOG_MRI} ${LOG_SPECT}"
