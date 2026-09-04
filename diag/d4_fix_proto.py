"""Prototipo del arreglo: nivel-sujeto + regularizacion fuerte + umbral calibrado + CV honesta."""
import numpy as np, pandas as pd, sys
from pathlib import Path
sys.path.insert(0,'/workspace/IL-FST')
import tensorflow as tf, os
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from fst.keras_fst_models import load_classifier, build_spect_head
for g in tf.config.list_physical_devices('GPU'): tf.config.experimental.set_memory_growth(g,True)

d=np.load('/workspace/IL-FST/diag/gap_feats.npz',allow_pickle=True)
Gm_tr,Gs_tr,Gm_te,Gs_te=d['Gm_tr'],d['Gs_tr'],d['Gm_te'],d['Gs_te']
str_,ste,ytr,yte=d['str_'],d['ste'],d['ytr'],d['yte']

def agg(G,s,y):
    df=pd.DataFrame(G); df['s']=s; df['y']=y
    g=df.groupby('s')
    return (g[list(range(G.shape[1]))].mean().values.astype(np.float32),
            g['y'].first().values, np.array(sorted(set(s))))
Am,sy_tr,sid_tr=agg(Gm_tr,str_,ytr); As,_,_=agg(Gs_tr,str_,ytr)
Bm,sy_te,sid_te=agg(Gm_te,ste,yte); Bs,_,_=agg(Gs_te,ste,yte)
print('sujetos train',Am.shape,'test',Bm.shape)

sp_cls=load_classifier(os.environ['FST_CLSFEAT_SPECT_CKPT']); head=build_spect_head(sp_cls)
def hp(G):
    m=np.repeat(np.repeat(np.maximum(G,0)[:,None,None,:].astype(np.float32),8,1),8,2)
    return np.concatenate([head(m[i:i+32],training=False).numpy()[:,1] for i in range(0,len(m),32)])

def ba_at(y,p,thr): return balanced_accuracy_score(y,(p>thr).astype(int))
def best_thr(y,p):
    ts=np.unique(p); return max(ts,key=lambda t:ba_at(y,p,t))

print('\n=== NIVEL SUJETO: Ridge GAP(f_mri)->GAP(f_spect) -> head SPECT congelada ===')
print('alpha    R2_te   | TEST AUC  BA@0.5  BA@thr(train)  BA@thr(oracle)')
rows=[]
for a in [1e3,1e4,3e4,1e5,3e5,1e6,3e6]:
    r=Ridge(alpha=a).fit(Am,As)
    Ph_te=r.predict(Bm); Ph_tr=r.predict(Am)
    r2=1-((Bs-Ph_te)**2).mean()/((Bs-As.mean(0))**2).mean()
    pte,ptr=hp(Ph_te),hp(Ph_tr)
    t=best_thr(sy_tr,ptr)
    auc=roc_auc_score(sy_te,pte)
    rows.append((a,r2,auc,ba_at(sy_te,pte,.5),ba_at(sy_te,pte,t),ba_at(sy_te,pte,best_thr(sy_te,pte))))
    print(f'{a:>8.0f} {r2:+.4f} |  {auc:.4f}   {rows[-1][3]:.4f}    {rows[-1][4]:.4f}        {rows[-1][5]:.4f}')

print('\n=== CV 5-fold HONESTA sobre los 147 de train (seleccion de alpha) ===')
skf=StratifiedKFold(5,shuffle=True,random_state=42)
for a in [1e3,1e4,1e5,1e6]:
    aucs=[]
    for i,(itr,iva) in enumerate(skf.split(Am,sy_tr)):
        r=Ridge(alpha=a).fit(Am[itr],As[itr])
        aucs.append(roc_auc_score(sy_tr[iva],hp(r.predict(Am[iva]))))
    print(f'alpha={a:>8.0f}  CV AUC = {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}')

print('\n=== referencias nivel-sujeto (TEST) ===')
print('head @ GAP(f_spect) real   AUC=%.4f BA=%.4f'%(roc_auc_score(sy_te,hp(Bs)),ba_at(sy_te,hp(Bs),.5)))
print('head @ GAP(f_mri) identidad AUC=%.4f BA@0.5=%.4f BA@thr=%.4f'%(
    roc_auc_score(sy_te,hp(Bm)),ba_at(sy_te,hp(Bm),.5),ba_at(sy_te,hp(Bm),best_thr(sy_tr,hp(Am)))))
lr=LogisticRegression(max_iter=5000,C=0.01,class_weight='balanced').fit(Am,sy_tr)
p=lr.predict_proba(Bm)[:,1]
print('sonda directa GAP(f_mri)->label  AUC=%.4f BA=%.4f  (sin pasar por head)'%(
    roc_auc_score(sy_te,p),balanced_accuracy_score(sy_te,(p>.5).astype(int))))
