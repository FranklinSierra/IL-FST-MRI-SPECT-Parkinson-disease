#!/usr/bin/env bash
# IL-FST environment: ROI SPECT 4 mm + gates from IL-CLDM (read-only).
# Source before any python that imports IL-CLDM config.
set -euo pipefail

export FST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ILCLDM_ROOT="${ILCLDM_ROOT:-/workspace/IL-CLDM}"

export PYTHONPATH="${FST_ROOT}:${ILCLDM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export ILCLDM_SKIP_CROP=1
export ILCLDM_ROI_VOLUME_SHAPE="96,112,114"
export ILCLDM_IMAGE_VOLUME_NORM="legacy"
export ILCLDM_WHOLE_MRI="${ILCLDM_ROOT}/data_rois/whole_MRI"
export ILCLDM_WHOLE_ABETA="${ILCLDM_ROOT}/data_rois/whole_Abeta_4mm"
export ILCLDM_REFERENCE_NIFTI="${ILCLDM_WHOLE_MRI}/3102.nii.gz"

# Frozen gate checkpoints (absolute paths under IL-CLDM)
export FST_SPECT_AAE="${FST_SPECT_AAE:-${ILCLDM_ROOT}/result/exp_nm_s1_spect_rois_4mm/AAE.pth.tar}"
export FST_MRI_XMODAL="${FST_MRI_XMODAL:-${ILCLDM_ROOT}/result/exp_nm_s1_mri_rois_4mm_xmodal_h64/phase1_mri/checkpoints/MRI_AAE.pth.tar}"
export FST_MOBILENET3D="${FST_MOBILENET3D:-${ILCLDM_ROOT}/result/exp_nm_cls_spect_aae_mnet3d/best.pt}"

export FST_DATA_INFO="${FST_DATA_INFO:-${ILCLDM_ROOT}/data_info}"
export FST_TRAIN_LIST="${FST_TRAIN_LIST:-${FST_DATA_INFO}/train.txt}"
export FST_VAL_LIST="${FST_VAL_LIST:-${FST_DATA_INFO}/validation.txt}"
export FST_TEST_LIST="${FST_TEST_LIST:-${FST_DATA_INFO}/test.txt}"
export FST_META_CSV="${FST_META_CSV:-${FST_DATA_INFO}/data_info.csv}"

export FST_OUT_DIR="${FST_OUT_DIR:-${FST_ROOT}/result/fst_h64_arad}"

# Classifier-feature FST (Keras MobileNet2D) — does not override FST_OUT_DIR / h64.
export FST_CLSFEAT_OUT_DIR="${FST_CLSFEAT_OUT_DIR:-${FST_ROOT}/result/fst_clsfeat_mnet2d_arad}"
export FST_CLSFEAT_CACHE="${FST_CLSFEAT_CACHE:-${FST_ROOT}/data/cache_slices_roi4mm}"
# Keras MobileNet v1 (conv_pw_13_relu) — z=[78,112]
export FST_CLSFEAT_MRI_CKPT="${FST_CLSFEAT_MRI_CKPT:-${FST_ROOT}/models/classifiers_roi_4mm/mri/model.keras}"
export FST_CLSFEAT_SPECT_CKPT="${FST_CLSFEAT_SPECT_CKPT:-${FST_ROOT}/models/classifiers_roi_4mm/spect/model.keras}"

# Pin GPU 6 unless FST_GPU is set (ignore ambient CUDA_VISIBLE_DEVICES).
export FST_GPU="${FST_GPU:-6}"
export CUDA_VISIBLE_DEVICES="${FST_GPU}"

# Prefijo libs NVIDIA pip (cuDNN/cuBLAS) para TensorFlow antes de arrancar Python.
_FST_NV_LIBS=""
for _d in /opt/conda/lib/python3.11/site-packages/nvidia/*/lib; do
  case "$_d" in
    */cu13/*) continue ;;
  esac
  if [[ -d "$_d" ]]; then
    _FST_NV_LIBS="${_FST_NV_LIBS:+$_FST_NV_LIBS:}$_d"
  fi
done
if [[ -n "$_FST_NV_LIBS" ]]; then
  export LD_LIBRARY_PATH="${_FST_NV_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
unset _d _FST_NV_LIBS
