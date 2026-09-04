#!/usr/bin/env bash
# FST ARAD train launcher — default GPU 6. Does not touch IL-CLDM E2/registry.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=env.sh
source "$ROOT/env.sh"

if [[ "${ILCLDM_WHOLE_ABETA}" != *"_4mm"* ]]; then
  echo "ERROR: refusing non-4mm SPECT path ${ILCLDM_WHOLE_ABETA}" >&2
  exit 1
fi
for f in "$FST_SPECT_AAE" "$FST_MRI_XMODAL" "$FST_MOBILENET3D"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing checkpoint: $f" >&2
    exit 1
  fi
done

BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-15}"

echo "=== FST h64 ARAD -> ${FST_OUT_DIR} GPU=${FST_GPU} SPECT=${ILCLDM_WHOLE_ABETA} ==="
python3 scripts/train_fst_h64_arad.py \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE"

echo "Done: ${FST_OUT_DIR}/best.pt"
