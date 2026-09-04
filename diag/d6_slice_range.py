"""Barrido de rango axial: donde tiene cerebro el MRI y donde funcionan los clasificadores."""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
sys.path.insert(0,'/workspace/IL-FST')
import tensorflow as tf
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from fst.slice_data import map_volume_to_unit_interval, slice_to_rgb, load_id_list, load_labels
from fst.keras_fst_models import load_classifier, build_feature_extractor
for g in tf.config.list_physical_devices('GPU'): tf.config.experimental.set_memory_growth(g,True)

ZS=list(range(0,113,6))
labels=load_labels(os.environ['FST_META_CSV'])
te=load_id_list(os.environ['FST_TEST_LIST']); tr=load_id_list(os.environ['FST_TRAIN_LIST'])
mri_cls=load_classifier(os.environ['FST_CLSFEAT_MRI_CKPT']); sp_cls=load_classifier(os.environ['FST_CLSFEAT_SPECT_CKPT'])
mf=build_feature_extractor(mri_cls,'mf'); sf=build_feature_extractor(sp_cls,'sf')
MR=Path(os.environ['ILCLDM_WHOLE_MRI']); SR=Path(os.environ['ILCLDM_WHOLE_ABETA'])

def harvest(ids):
    Pm=np.zeros((len(ids),len(ZS)));Ps=np.zeros((len(ids),len(ZS)))
    Gm=np.zeros((len(ids),len(ZS),1024),np.float32);Gs=np.zeros((len(ids),len(ZS),1024),np.float32)
    Y=np.array([labels[i] for i in ids])
    for n,sid in enumerate(ids):
        mv=map_volume_to_unit_interval(np.asarray(nib.load(str(MR/f'{sid}.nii.gz')).get_fdata(),np.float32),'mri')
        sv=map_volume_to_unit_interval(np.asarray(nib.load(str(SR/f'{sid}.nii.gz')).get_fdata(),np.float32),'spect')
        tm=np.stack([slice_to_rgb(mv[:,:,z]) for z in ZS]); ts=np.stack([slice_to_rgb(sv[:,:,z]) for z in ZS])
        Pm[n]=mri_cls(tm,training=False).numpy()[:,1]; Ps[n]=sp_cls(ts,training=False).numpy()[:,1]
        Gm[n]=mf(tm,training=False).numpy().mean(axis=(1,2)); Gs[n]=sf(ts,training=False).numpy().mean(axis=(1,2))
        if n%40==0: print(n,'/',len(ids),flush=True)
    return Pm,Ps,Gm,Gs,Y
print('TEST...'); Pm,Ps,Gm,Gs,Y=harvest(te)
print('TRAIN...'); Pmt,Pst,Gmt,Gst,Yt=harvest(tr)
np.savez('/workspace/IL-FST/diag/slice_sweep.npz',ZS=ZS,Pm=Pm,Ps=Ps,Gm=Gm,Gs=Gs,Y=Y,
         Pmt=Pmt,Pst=Pst,Gmt=Gmt,Gst=Gst,Yt=Yt,te=te,tr=tr)

print('\n  z | AUC clasif MRI | AUC clasif SPECT   (TEST, por slice individual)')
for j,z in enumerate(ZS):
    print(f'{z:3d} |     {roc_auc_score(Y,Pm[:,j]):.3f}      |     {roc_auc_score(Y,Ps[:,j]):.3f}')

print('\n=== AUC promediando probabilidad sobre ventanas de slices (TEST) ===')
print('ventana            MRI     SPECT')
wins={'z=0..15 (ACTUAL)':[j for j,z in enumerate(ZS) if z<=15],
      'z=18..47':[j for j,z in enumerate(ZS) if 18<=z<=47],
      'z=48..77':[j for j,z in enumerate(ZS) if 48<=z<=77],
      'z=78..112':[j for j,z in enumerate(ZS) if 78<=z<=112],
      'z=36..96':[j for j,z in enumerate(ZS) if 36<=z<=96],
      'z=18..112 (todo)':[j for j,z in enumerate(ZS) if z>=18]}
for k,js in wins.items():
    if not js: continue
    print(f'{k:20s} {roc_auc_score(Y,Pm[:,js].mean(1)):.3f}   {roc_auc_score(Y,Ps[:,js].mean(1)):.3f}')

print('\n=== sonda logistica sobre GAP(features) por ventana: cuanto INFORMA cada espacio ===')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
print('ventana            MRI feat   SPECT feat   (train->TEST AUC)')
for k,js in wins.items():
    if not js: continue
    out=[]
    for G,Gt in ((Gm,Gmt),(Gs,Gst)):
        Xtr=Gt[:,js].mean(1); Xte=G[:,js].mean(1)
        sc=StandardScaler().fit(Xtr)
        c=LogisticRegression(max_iter=4000,C=0.01,class_weight='balanced').fit(sc.transform(Xtr),Yt)
        out.append(roc_auc_score(Y,c.predict_proba(sc.transform(Xte))[:,1]))
    print(f'{k:20s}   {out[0]:.3f}       {out[1]:.3f}')
