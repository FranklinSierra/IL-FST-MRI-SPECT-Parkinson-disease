#!/usr/bin/env python3
"""Caché de features con rango axial CORRECTO, directo desde NIfTI (sin tiles en disco).

El rango [0,15] heredado de CoCoLIT no contiene cerebro en MRI (fraccion de senal
0.000-0.030; el cerebro empieza en z=18). Ver diag/d6_slice_range.py.
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
FST=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(FST))
import tensorflow as tf
from fst.slice_data import map_volume_to_unit_interval, slice_to_rgb, load_id_list, load_labels
from fst.keras_fst_models import load_classifier, build_feature_extractor
for g in tf.config.list_physical_devices('GPU'): tf.config.experimental.set_memory_growth(g,True)

ZS=list(range(int(os.environ.get("FST_ZMIN",18)),int(os.environ.get("FST_ZMAX",112))+1,
               int(os.environ.get("FST_ZSTEP",3))))
OUT=Path(os.environ.get("FST_FEAT_CACHE_Z",FST/"data/feat_cache_z")); OUT.mkdir(parents=True,exist_ok=True)
labels=load_labels(os.environ['FST_META_CSV'])
tr=load_id_list(os.environ['FST_TRAIN_LIST']); te=load_id_list(os.environ['FST_TEST_LIST'])
ids=tr+te; split=["train"]*len(tr)+["test"]*len(te)
print(f"z={ZS[0]}..{ZS[-1]} step={ZS[1]-ZS[0]} -> {len(ZS)} slices x {len(ids)} sujetos",flush=True)

mri_cls=load_classifier(os.environ['FST_CLSFEAT_MRI_CKPT']); sp_cls=load_classifier(os.environ['FST_CLSFEAT_SPECT_CKPT'])
mf=build_feature_extractor(mri_cls,'mf'); sf=build_feature_extractor(sp_cls,'sf')
MR=Path(os.environ['ILCLDM_WHOLE_MRI']); SR=Path(os.environ['ILCLDM_WHOLE_ABETA'])

N=len(ids)*len(ZS)
Fm=np.lib.format.open_memmap(OUT/"f_mri.npy",mode="w+",dtype=np.float16,shape=(N,8,8,1024))
Gs=np.zeros((N,1024),np.float32); rows=[]
i=0
for n,(sid,sp) in enumerate(zip(ids,split)):
    mv=map_volume_to_unit_interval(np.asarray(nib.load(str(MR/f'{sid}.nii.gz')).get_fdata(),np.float32),'mri')
    sv=map_volume_to_unit_interval(np.asarray(nib.load(str(SR/f'{sid}.nii.gz')).get_fdata(),np.float32),'spect')
    tm=np.stack([slice_to_rgb(mv[:,:,z]) for z in ZS]); ts=np.stack([slice_to_rgb(sv[:,:,z]) for z in ZS])
    Fm[i:i+len(ZS)]=mf(tm,training=False).numpy().astype(np.float16)
    Gs[i:i+len(ZS)]=sf(ts,training=False).numpy().mean(axis=(1,2))
    rows+= [{"subject_id":sid,"slice_idx":z,"label":labels[sid],"split":sp} for z in ZS]
    i+=len(ZS)
    if n%40==0: print(n,'/',len(ids),flush=True)
Fm.flush(); np.save(OUT/"gap_spect.npy",Gs); pd.DataFrame(rows).to_csv(OUT/"index.csv",index=False)
print("guardado",Fm.shape,"->",OUT,flush=True)
