#!/usr/bin/env bash
# FST clsfeat MobileNet2D ARAD train launcher — GPU (FST_GPU, default 6).
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

if [[ "${ILCLDM_WHOLE_ABETA}" != *"_4mm"* ]]; then
  echo "ERROR: refusing non-4mm SPECT path ${ILCLDM_WHOLE_ABETA}" >&2
  exit 1
fi
for f in "$FST_CLSFEAT_MRI_CKPT" "$FST_CLSFEAT_SPECT_CKPT"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing checkpoint: $f" >&2
    exit 1
  fi
done

BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-15}"

echo "=== FST clsfeat mnet2d ARAD -> ${FST_CLSFEAT_OUT_DIR} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ==="
python3 -u scripts/train_fst_clsfeat_mnet2d_arad.py \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE"

echo "Done: ${FST_CLSFEAT_OUT_DIR}/best.keras"
