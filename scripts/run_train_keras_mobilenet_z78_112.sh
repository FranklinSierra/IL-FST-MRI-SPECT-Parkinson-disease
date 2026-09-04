#!/usr/bin/env bash
# Train Keras MobileNet v1 MRI + SPECT (z=78..112) into classifiers_roi_4mm/{mri,spect}/
# Parallel on GPU 6 with memory growth. Then ready for FST clsfeat.
set -euo pipefail

FST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${FST_ROOT}/env.sh"

export FST_GPU="${FST_GPU:-6}"
export CUDA_VISIBLE_DEVICES="${FST_GPU}"
export PYTHONUNBUFFERED=1

OUT_MRI="${FST_ROOT}/models/classifiers_roi_4mm/mri"
OUT_SPECT="${FST_ROOT}/models/classifiers_roi_4mm/spect"
SPLIT="${FST_ROOT}/models/classifiers_roi_4mm/_shared_split_z78_112"
LOG_DIR="${FST_ROOT}/logs"
mkdir -p "${OUT_MRI}" "${OUT_SPECT}" "${SPLIT}" "${LOG_DIR}"

# Ensure shared holdout exists
if [[ ! -f "${SPLIT}/train_fit.txt" ]]; then
  python3 - <<PY
import sys
from pathlib import Path
sys.path.insert(0, "${FST_ROOT}")
from fst.slice_data import load_id_list, load_labels, stratified_holdout, write_id_list
labels = load_labels("${FST_META_CSV}")
fit, val = stratified_holdout(load_id_list("${FST_TRAIN_LIST}"), labels, n_val=15, seed=42)
d = Path("${SPLIT}")
write_id_list(d/"train_fit.txt", fit)
write_id_list(d/"val_holdout.txt", val)
write_id_list(d/"test.txt", load_id_list("${FST_TEST_LIST}"))
print("wrote", d, "fit", len(fit), "val", len(val))
PY
fi

COMMON=(
  --train-list "${FST_TRAIN_LIST}"
  --test-list "${FST_TEST_LIST}"
  --meta-csv "${FST_META_CSV}"
  --fit-list "${SPLIT}/train_fit.txt"
  --val-list "${SPLIT}/val_holdout.txt"
  --z-min 78 --z-max 112
  --epochs "${EPOCHS:-50}"
  --batch-size "${BATCH:-12}"
  --patience "${PATIENCE:-10}"
  --seed 42
)

LOG_MRI="${LOG_DIR}/train_keras_mobilenet_mri_z78_112.log"
LOG_SPECT="${LOG_DIR}/train_keras_mobilenet_spect_z78_112.log"

echo "Keras MobileNet v1 → ${OUT_MRI} + ${OUT_SPECT} on GPU ${FST_GPU}"

nohup env CUDA_VISIBLE_DEVICES="${FST_GPU}" \
  python3 "${FST_ROOT}/scripts/train_keras_mobilenet_roi_classifier.py" \
  --modality mri --volume-root "${ILCLDM_WHOLE_MRI}" --out-dir "${OUT_MRI}" \
  "${COMMON[@]}" >"${LOG_MRI}" 2>&1 &
echo $! > "${LOG_DIR}/train_keras_mobilenet_mri_z78_112.pid"

nohup env CUDA_VISIBLE_DEVICES="${FST_GPU}" \
  python3 "${FST_ROOT}/scripts/train_keras_mobilenet_roi_classifier.py" \
  --modality spect --volume-root "${ILCLDM_WHOLE_ABETA}" --out-dir "${OUT_SPECT}" \
  "${COMMON[@]}" >"${LOG_SPECT}" 2>&1 &
echo $! > "${LOG_DIR}/train_keras_mobilenet_spect_z78_112.pid"

echo "PIDs MRI=$(cat ${LOG_DIR}/train_keras_mobilenet_mri_z78_112.pid) SPECT=$(cat ${LOG_DIR}/train_keras_mobilenet_spect_z78_112.pid)"
echo "tail -f ${LOG_MRI} ${LOG_SPECT}"
