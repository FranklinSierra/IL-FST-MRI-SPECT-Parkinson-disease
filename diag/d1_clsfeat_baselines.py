"""Baselines duros para el enfoque B (clsfeat MobileNet2D)."""
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd
FST=Path('/workspace/IL-FST'); sys.path.insert(0,str(FST))
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score
from fst.slice_data import (ensure_subject_slices, attach_labels, load_id_list,
                            load_labels, DEFAULT_SLICE_RANGE)
from fst.keras_fst_models import load_classifier, build_feature_extractor, build_spect_head

for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)

MRI_ROOT=Path(os.environ['ILCLDM_WHOLE_MRI']); SPECT_ROOT=Path(os.environ['ILCLDM_WHOLE_ABETA'])
CACHE=Path(os.environ['FST_CLSFEAT_CACHE'])
labels=load_labels(os.environ['FST_META_CSV'])
test_ids=load_id_list(os.environ['FST_TEST_LIST'])
train_ids=load_id_list(os.environ['FST_TRAIN_LIST'])

def manifest(ids):
    return attach_labels(ensure_subject_slices(ids, mri_root=MRI_ROOT, spect_root=SPECT_ROOT,
        cache_root=CACHE, z_min=DEFAULT_SLICE_RANGE[0], z_max=DEFAULT_SLICE_RANGE[1]), labels)

man_te=manifest(test_ids); man_tr=manifest(train_ids)
print('slices test', len(man_te), 'train', len(man_tr))

mri_cls=load_classifier(os.environ['FST_CLSFEAT_MRI_CKPT'])
spect_cls=load_classifier(os.environ['FST_CLSFEAT_SPECT_CKPT'])
mri_feat=build_feature_extractor(mri_cls,'mri_feat')
spect_feat=build_feature_extractor(spect_cls,'spect_feat')
head=build_spect_head(spect_cls)

def batches(df, col, bs=32):
    for i in range(0,len(df),bs):
        b=df.iloc[i:i+bs]
        yield np.stack([np.load(p) for p in b[col]],0).astype(np.float32), b

def subj_vote(df, probs):
    d=df.copy(); d['p']=probs; d['pc']=(probs>0.5).astype(int)
    g=d.groupby('subject_id').agg(label=('label','first'), p=('p','mean'),
                                  vote=('pc', lambda s: int(s.mean()>0.5)))
    return g

def report(name, g):
    ba=balanced_accuracy_score(g.label,g.vote); auc=roc_auc_score(g.label,g.p)
    ba_p=balanced_accuracy_score(g.label,(g.p>0.5).astype(int))
    print(f'{name:42s} BA(vote)={ba:.4f} BA(meanp)={ba_p:.4f} AUC={auc:.4f} acc={accuracy_score(g.label,g.vote):.4f}')
    return dict(ba=ba,auc=auc)

res={}
# 1. clasificador MRI end-to-end sobre MRI real
p=np.concatenate([mri_cls(x,training=False).numpy()[:,1] for x,_ in batches(man_te,'mri_path')])
res['mri_cls_on_mri_test']=report('MRI classifier @ MRI real (TEST)', subj_vote(man_te,p))
# 2. clasificador SPECT end-to-end sobre SPECT real
p=np.concatenate([spect_cls(x,training=False).numpy()[:,1] for x,_ in batches(man_te,'spect_path')])
res['spect_cls_on_spect_test']=report('SPECT classifier @ SPECT real (TEST)', subj_vote(man_te,p))
# 3. head reconstruida sobre f_spect real  (debe igualar 2)
p=np.concatenate([head(spect_feat(x,training=False),training=False).numpy()[:,1] for x,_ in batches(man_te,'spect_path')])
res['head_on_fspect_test']=report('rebuilt head @ f_spect real (TEST)', subj_vote(man_te,p))
# 4. identidad: f_mri -> head SPECT
p=np.concatenate([head(mri_feat(x,training=False),training=False).numpy()[:,1] for x,_ in batches(man_te,'mri_path')])
res['head_on_fmri_test']=report('rebuilt head @ f_mri (identidad)', subj_vote(man_te,p))
# 5. MRI classifier en train (para ver overfit del fold3)
p=np.concatenate([mri_cls(x,training=False).numpy()[:,1] for x,_ in batches(man_tr,'mri_path')])
res['mri_cls_on_mri_train']=report('MRI classifier @ MRI real (TRAIN)', subj_vote(man_tr,p))
p=np.concatenate([spect_cls(x,training=False).numpy()[:,1] for x,_ in batches(man_tr,'spect_path')])
res['spect_cls_on_spect_train']=report('SPECT classifier @ SPECT real (TRAIN)', subj_vote(man_tr,p))

# 6. estadisticas de f_spect / f_mri
def feat_stats(df, col, extractor, n=400):
    F=[]
    for x,_ in batches(df.iloc[:n], col):
        F.append(extractor(x,training=False).numpy())
    F=np.concatenate(F,0)
    return F
Fs=feat_stats(man_te,'spect_path',spect_feat)
Fm=feat_stats(man_te,'mri_path',mri_feat)
print('\n--- f_spect (test) ---')
print('shape',Fs.shape,'min',Fs.min(),'max',Fs.max(),'mean',Fs.mean(),'std',Fs.std())
print('var total (=MSE de predecir la media global):', Fs.var())
mu=Fs.mean(axis=0,keepdims=True)
print('MSE si predices la media por-posicion:', ((Fs-mu)**2).mean())
print('fraccion de ceros:', (Fs==0).mean())
print('--- f_mri (test) ---')
print('shape',Fm.shape,'min',Fm.min(),'max',Fm.max(),'mean',Fm.mean(),'std',Fm.std())
json.dump({k:{kk:float(vv) for kk,vv in v.items()} for k,v in res.items()},
          open('/workspace/IL-FST/diag/d1_baselines.json','w'), indent=2)
print('\nOK')
