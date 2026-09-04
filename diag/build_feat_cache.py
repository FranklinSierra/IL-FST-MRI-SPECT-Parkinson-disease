"""Precomputa f_mri y f_spect (conv_pw_13_relu) para TODOS los sujetos. Extractores congelados."""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
FST=Path('/workspace/IL-FST'); sys.path.insert(0,str(FST))
import tensorflow as tf
for g in tf.config.list_physical_devices('GPU'): tf.config.experimental.set_memory_growth(g,True)
from fst.slice_data import (ensure_subject_slices, attach_labels, load_id_list,
                            load_labels, DEFAULT_SLICE_RANGE)
from fst.keras_fst_models import load_classifier, build_feature_extractor

OUT=Path('/dev/shm/fst_cache'); OUT.mkdir(parents=True, exist_ok=True)
labels=load_labels(os.environ['FST_META_CSV'])
ids=load_id_list(os.environ['FST_TRAIN_LIST'])+load_id_list(os.environ['FST_TEST_LIST'])
man=attach_labels(ensure_subject_slices(ids, mri_root=Path(os.environ['ILCLDM_WHOLE_MRI']),
    spect_root=Path(os.environ['ILCLDM_WHOLE_ABETA']), cache_root=Path(os.environ['FST_CLSFEAT_CACHE']),
    z_min=DEFAULT_SLICE_RANGE[0], z_max=DEFAULT_SLICE_RANGE[1]), labels)
man=man.sort_values(['subject_id','slice_idx']).reset_index(drop=True)
print('slices', len(man), 'subjects', man.subject_id.nunique())

mri_feat=build_feature_extractor(load_classifier(os.environ['FST_CLSFEAT_MRI_CKPT']),'mri_feat')
spect_feat=build_feature_extractor(load_classifier(os.environ['FST_CLSFEAT_SPECT_CKPT']),'spect_feat')

def extract(col, ex):
    out=np.empty((len(man),8,8,1024), dtype=np.float16)
    for i in range(0,len(man),32):
        b=man.iloc[i:i+32]
        x=np.stack([np.load(p) for p in b[col]],0).astype(np.float32)
        out[i:i+len(b)]=ex(x,training=False).numpy().astype(np.float16)
    return out

fm=extract('mri_path',mri_feat);  np.save(OUT/'f_mri.npy', fm);  print('f_mri', fm.shape, fm.nbytes/1e9,'GB')
fs=extract('spect_path',spect_feat); np.save(OUT/'f_spect.npy', fs); print('f_spect', fs.shape)
man[['subject_id','slice_idx','label']].to_csv(OUT/'manifest.csv', index=False)
print('OK ->', OUT)
