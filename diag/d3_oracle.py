"""Oraculo: mejor traductor lineal posible f_mri -> f_spect, y que da la head."""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
FST=Path('/workspace/IL-FST'); sys.path.insert(0,str(FST))
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.linear_model import Ridge
from fst.slice_data import (ensure_subject_slices, attach_labels, load_id_list,
                            load_labels, DEFAULT_SLICE_RANGE)
from fst.keras_fst_models import load_classifier, build_feature_extractor, build_spect_head
for g in tf.config.list_physical_devices('GPU'): tf.config.experimental.set_memory_growth(g,True)

labels=load_labels(os.environ['FST_META_CSV'])
tr=load_id_list(os.environ['FST_TRAIN_LIST']); te=load_id_list(os.environ['FST_TEST_LIST'])
def man(ids): return attach_labels(ensure_subject_slices(ids,
    mri_root=Path(os.environ['ILCLDM_WHOLE_MRI']),spect_root=Path(os.environ['ILCLDM_WHOLE_ABETA']),
    cache_root=Path(os.environ['FST_CLSFEAT_CACHE']),z_min=DEFAULT_SLICE_RANGE[0],z_max=DEFAULT_SLICE_RANGE[1]),labels)
mtr,mte=man(tr),man(te)
mri_cls=load_classifier(os.environ['FST_CLSFEAT_MRI_CKPT']); sp_cls=load_classifier(os.environ['FST_CLSFEAT_SPECT_CKPT'])
mf=build_feature_extractor(mri_cls,'mf'); sf=build_feature_extractor(sp_cls,'sf'); head=build_spect_head(sp_cls)

def gap(df,col,ex,bs=32):
    out=[]
    for i in range(0,len(df),bs):
        x=np.stack([np.load(p) for p in df[col].iloc[i:i+bs]],0).astype(np.float32)
        out.append(ex(x,training=False).numpy().mean(axis=(1,2)))
    return np.concatenate(out)
print('extrayendo GAP features...')
Gm_tr,Gs_tr=gap(mtr,'mri_path',mf),gap(mtr,'spect_path',sf)
Gm_te,Gs_te=gap(mte,'mri_path',mf),gap(mte,'spect_path',sf)
ytr=mtr.label.values; yte=mte.label.values
print('GAP',Gm_tr.shape,Gs_tr.shape)

# CLAVE: la head es GAP->fc1->fc2, asi que solo depende de GAP(f). Verificamos.
def head_from_gap(G):
    # head espera (N,8,8,1024); replicamos el GAP en el mapa -> mismo GAP
    m=np.repeat(np.repeat(G[:,None,None,:],8,1),8,2).astype(np.float32)
    ps=[]
    for i in range(0,len(m),64): ps.append(head(m[i:i+64],training=False).numpy()[:,1])
    return np.concatenate(ps)
def subj(df,p,thr=0.5):
    d=pd.DataFrame({'s':df.subject_id.values,'y':df.label.values,'p':p})
    g=d.groupby('s').agg(y=('y','first'),p=('p','mean'))
    return balanced_accuracy_score(g.y,(g.p>thr).astype(int)), roc_auc_score(g.y,g.p)

print('\n--- verificacion: head solo depende de GAP ---')
p_direct=[]
for i in range(0,len(mte),32):
    x=np.stack([np.load(p) for p in mte['spect_path'].iloc[i:i+32]],0).astype(np.float32)
    p_direct.append(head(sf(x,training=False),training=False).numpy()[:,1])
p_direct=np.concatenate(p_direct)
p_gapv=head_from_gap(Gs_te)
print('max|diff| head(mapa) vs head(GAP replicado):',np.abs(p_direct-p_gapv).max())
print('head @ f_spect real  BA=%.4f AUC=%.4f'%subj(mte,p_direct))

print('\n--- ORACULO: Ridge GAP(f_mri) -> GAP(f_spect) ---')
for a in [1e2,1e3,1e4,1e5]:
    r=Ridge(alpha=a).fit(Gm_tr,Gs_tr)
    Gh_te=r.predict(Gm_te); Gh_tr=r.predict(Gm_tr)
    r2=1-((Gs_te-Gh_te)**2).mean()/((Gs_te-Gs_tr.mean(0))**2).mean()
    ba,auc=subj(mte,head_from_gap(np.maximum(Gh_te,0).astype(np.float32)))
    batr,auctr=subj(mtr,head_from_gap(np.maximum(Gh_tr,0).astype(np.float32)))
    print(f'alpha={a:>8.0f} R2_test={r2:+.4f} | TEST BA={ba:.4f} AUC={auc:.4f} | train BA={batr:.4f} AUC={auctr:.4f}')

print('\n--- referencia: predictor media (constante) ---')
Gc=np.tile(Gs_tr.mean(0),(len(Gs_te),1)).astype(np.float32)
print('BA=%.4f AUC=%.4f (debe ser 0.5/nan-ish)'%subj(mte,head_from_gap(Gc)))
print('\n--- referencia: identidad GAP(f_mri) -> head ---')
print('BA=%.4f AUC=%.4f'%subj(mte,head_from_gap(Gm_te)))
np.savez('/workspace/IL-FST/diag/gap_feats.npz',Gm_tr=Gm_tr,Gs_tr=Gs_tr,Gm_te=Gm_te,Gs_te=Gs_te,
         str_=mtr.subject_id.values,ste=mte.subject_id.values,ytr=ytr,yte=yte)
print('OK')
