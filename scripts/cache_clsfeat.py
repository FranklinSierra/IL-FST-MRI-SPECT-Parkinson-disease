#!/usr/bin/env python3
"""Precomputa f_mri (8,8,1024) y GAP(f_spect) (1024,) de los MobileNet congelados.

La head SPECT es GAP->fc1->Drop->fc2->softmax, luego SOLO consume GAP(f).
Verificado: |head(mapa) - head(GAP replicado)| < 4e-7.
"""
import os, sys
from pathlib import Path
import numpy as np
FST=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(FST))
import tensorflow as tf
from fst.slice_data import (ensure_subject_slices, attach_labels, load_id_list,
                            load_labels, DEFAULT_SLICE_RANGE)
from fst.keras_fst_models import load_classifier, build_feature_extractor
for g in tf.config.list_physical_devices('GPU'): tf.config.experimental.set_memory_growth(g,True)

OUT=Path(os.environ.get('FST_FEAT_CACHE', FST/'data/feat_cache_pw13')); OUT.mkdir(parents=True,exist_ok=True)
labels=load_labels(os.environ['FST_META_CSV'])
ids=sorted(set(load_id_list(os.environ['FST_TRAIN_LIST'])+load_id_list(os.environ['FST_TEST_LIST'])))
man=attach_labels(ensure_subject_slices(ids,
    mri_root=Path(os.environ['ILCLDM_WHOLE_MRI']),spect_root=Path(os.environ['ILCLDM_WHOLE_ABETA']),
    cache_root=Path(os.environ['FST_CLSFEAT_CACHE']),
    z_min=DEFAULT_SLICE_RANGE[0],z_max=DEFAULT_SLICE_RANGE[1]),labels)
print('sujetos',len(ids),'slices',len(man),flush=True)
mri_cls=load_classifier(os.environ['FST_CLSFEAT_MRI_CKPT']); sp_cls=load_classifier(os.environ['FST_CLSFEAT_SPECT_CKPT'])
mf=build_feature_extractor(mri_cls,'mf'); sf=build_feature_extractor(sp_cls,'sf')

N=len(man); BS=32
Fm=np.lib.format.open_memmap(OUT/'f_mri.npy',mode='w+',dtype=np.float16,shape=(N,8,8,1024))
Gs=np.zeros((N,1024),np.float32); Gm=np.zeros((N,1024),np.float32)
for i in range(0,N,BS):
    xm=np.stack([np.load(p) for p in man['mri_path'].iloc[i:i+BS]],0).astype(np.float32)
    xs=np.stack([np.load(p) for p in man['spect_path'].iloc[i:i+BS]],0).astype(np.float32)
    fm=mf(xm,training=False).numpy(); fs=sf(xs,training=False).numpy()
    Fm[i:i+BS]=fm.astype(np.float16); Gm[i:i+BS]=fm.mean(axis=(1,2)); Gs[i:i+BS]=fs.mean(axis=(1,2))
    if i%640==0: print(i,'/',N,flush=True)
Fm.flush()
np.save(OUT/'gap_spect.npy',Gs); np.save(OUT/'gap_mri.npy',Gm)
man[['subject_id','slice_idx','label']].to_csv(OUT/'index.csv',index=False)
print('guardado f_mri',Fm.shape,'gap_spect',Gs.shape,'->',OUT,flush=True)
