"""Oraculo enfoque A: traductor de bajo rango h64_mri -> h64_spect -> decoder -> MobileNet3D."""
import os, sys
from pathlib import Path
import numpy as np, torch
FST=Path('/workspace/IL-FST'); ILC=Path('/workspace/IL-CLDM')
sys.path.insert(0,str(ILC)); sys.path.insert(0,str(FST))
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
import importlib.util
spec=importlib.util.spec_from_file_location('trA', FST/'scripts/train_fst_h64_arad.py')
trA=importlib.util.module_from_spec(spec); spec.loader.exec_module(trA); trA._bootstrap_paths()
from dataset import TwoDataset
from code.diffusion_features import decode_h64_via_bottleneck

dev=torch.device('cuda')
spect_aae=trA.load_aae(os.environ['FST_SPECT_AAE'],dev)
mri_aae  =trA.load_aae(os.environ['FST_MRI_XMODAL'],dev)
mnet     =trA.load_mobilenet3d(os.environ['FST_MOBILENET3D'],dev)
CACHE=Path('/workspace/IL-FST/diag/h64_cache.npz')

@torch.no_grad()
def harvest(lst):
    ds=TwoDataset(root_MRI=os.environ['ILCLDM_WHOLE_MRI'],root_Abeta=os.environ['ILCLDM_WHOLE_ABETA'],
                  task=str(lst),stage='validation')
    L=DataLoader(ds,batch_size=2,shuffle=False,num_workers=0)
    HM=[];HS=[];Y=[]
    for mri,spect,names,labels in L:
        mri=torch.as_tensor(np.asarray(mri),dtype=torch.float32,device=dev).unsqueeze(1)
        spect=torch.as_tensor(np.asarray(spect),dtype=torch.float32,device=dev).unsqueeze(1)
        HM.append(trA.encode_h64(mri_aae,mri).cpu().numpy().astype(np.float16))
        HS.append(trA.encode_h64(spect_aae,spect).cpu().numpy().astype(np.float16))
        Y.append(labels.reshape(-1).numpy())
    return np.concatenate(HM),np.concatenate(HS),np.concatenate(Y)

if CACHE.exists():
    z=np.load(CACHE); HMtr,HStr,Ytr,HMte,HSte,Yte=[z[k] for k in ('HMtr','HStr','Ytr','HMte','HSte','Yte')]
else:
    HMtr,HStr,Ytr=harvest(os.environ['FST_TRAIN_LIST'])
    HMte,HSte,Yte=harvest(os.environ['FST_TEST_LIST'])
    np.savez(CACHE,HMtr=HMtr,HStr=HStr,Ytr=Ytr,HMte=HMte,HSte=HSte,Yte=Yte)
print('h64 train',HMtr.shape,'test',HMte.shape)
SH=HMtr.shape[1:]

@torch.no_grad()
def mnet_prob(H,bs=2):
    out=[]
    for i in range(0,len(H),bs):
        h=torch.as_tensor(H[i:i+bs].astype(np.float32),device=dev)
        out.append(torch.softmax(mnet(decode_h64_via_bottleneck(spect_aae,h)),1)[:,1].cpu().numpy())
    return np.concatenate(out)
def ba_at(y,p,t): return balanced_accuracy_score(y,(p>t).astype(int))
def best_thr(y,p): return max(np.unique(p),key=lambda t:ba_at(y,p,t))

Xtr=HMtr.reshape(len(HMtr),-1).astype(np.float32); Xte=HMte.reshape(len(HMte),-1).astype(np.float32)
Ttr=HStr.reshape(len(HStr),-1).astype(np.float32); Tte=HSte.reshape(len(HSte),-1).astype(np.float32)
print('ajustando PCA...')
K=100
pm=PCA(K,random_state=0).fit(Xtr); ps=PCA(K,random_state=0).fit(Ttr)
Zm_tr,Zm_te=pm.transform(Xtr),pm.transform(Xte); Zs_tr=ps.transform(Ttr)
print('var explicada mri %.3f spect %.3f'%(pm.explained_variance_ratio_.sum(),ps.explained_variance_ratio_.sum()))

print('\n=== ORACULO A: PCA%d + Ridge, decode, MobileNet3D  (TEST 34 suj) ==='%K)
print('alpha     R2_te   | AUC     BA@0.5   BA@thr(train)')
for a in [1e0,1e1,1e2,1e3,1e4]:
    r=Ridge(alpha=a).fit(Zm_tr,Zs_tr)
    Hh_te=np.maximum(ps.inverse_transform(r.predict(Zm_te)),0).reshape(-1,*SH)
    Hh_tr=np.maximum(ps.inverse_transform(r.predict(Zm_tr)),0).reshape(-1,*SH)
    r2=1-((Tte-Hh_te.reshape(len(Hh_te),-1))**2).mean()/((Tte-Ttr.mean(0))**2).mean()
    pte=mnet_prob(Hh_te); ptr=mnet_prob(Hh_tr); t=best_thr(Ytr,ptr)
    print(f'{a:>8.0f} {r2:+.4f} |  {roc_auc_score(Yte,pte):.4f}  {ba_at(Yte,pte,.5):.4f}   {ba_at(Yte,pte,t):.4f}')

print('\n=== referencias ===')
print('techo   decode(h64_spect)  AUC=%.4f BA=%.4f'%(roc_auc_score(Yte,mnet_prob(HSte)),ba_at(Yte,mnet_prob(HSte),.5)))
pi_te=mnet_prob(HMte); pi_tr=mnet_prob(HMtr)
print('identidad decode(h64_mri)  AUC=%.4f BA@0.5=%.4f BA@thr=%.4f'%(
    roc_auc_score(Yte,pi_te),ba_at(Yte,pi_te,.5),ba_at(Yte,pi_te,best_thr(Ytr,pi_tr))))
mu=HStr.astype(np.float32).mean(0,keepdims=True)
pm_=mnet_prob(np.repeat(mu,len(HSte),0))
print('media constante            AUC=%.4f'%roc_auc_score(Yte,pm_))
print('OK')
