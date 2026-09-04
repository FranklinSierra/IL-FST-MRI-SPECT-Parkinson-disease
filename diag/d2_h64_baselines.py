"""Baselines duros para el enfoque A (h64 AAE + MobileNet3D)."""
import os, sys, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn
FST=Path('/workspace/IL-FST'); ILC=Path('/workspace/IL-CLDM')
sys.path.insert(0,str(ILC)); sys.path.insert(0,str(FST))
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0,str(FST/'scripts'))
import importlib.util
spec=importlib.util.spec_from_file_location('trA', FST/'scripts/train_fst_h64_arad.py')
trA=importlib.util.module_from_spec(spec); spec.loader.exec_module(trA)
trA._bootstrap_paths()

from dataset import TwoDataset
from code.diffusion_features import decode_h64_via_bottleneck
from code.volume_shapes import get_latent_shape, get_working_volume_shape

dev=torch.device('cuda')
spect_aae=trA.load_aae(os.environ['FST_SPECT_AAE'],dev)
mri_aae  =trA.load_aae(os.environ['FST_MRI_XMODAL'],dev)
mnet     =trA.load_mobilenet3d(os.environ['FST_MOBILENET3D'],dev)
print('latent',get_latent_shape(),'working',get_working_volume_shape())

def loader(lst):
    ds=TwoDataset(root_MRI=os.environ['ILCLDM_WHOLE_MRI'],root_Abeta=os.environ['ILCLDM_WHOLE_ABETA'],
                  task=str(lst),stage='validation')
    return DataLoader(ds,batch_size=2,shuffle=False,num_workers=0)

@torch.no_grad()
def harvest(lst, tag):
    L=loader(lst); HS=[];HM=[];Y=[];P={}
    p_recon=[];p_gt=[];p_ident=[];p_meanh=[]
    for mri,spect,names,labels in L:
        mri=torch.as_tensor(np.asarray(mri),dtype=torch.float32,device=dev).unsqueeze(1)
        spect=torch.as_tensor(np.asarray(spect),dtype=torch.float32,device=dev).unsqueeze(1)
        y=labels.reshape(-1).long().numpy()
        hm=trA.encode_h64(mri_aae,mri); hs=trA.encode_h64(spect_aae,spect)
        # gate: mnet sobre recon AAE del SPECT real
        rec_gt=decode_h64_via_bottleneck(spect_aae,hs)
        p_gt.append(torch.softmax(mnet(rec_gt),1)[:,1].cpu().numpy())
        # mnet sobre SPECT real crudo
        p_recon.append(torch.softmax(mnet(spect),1)[:,1].cpu().numpy())
        # identidad: decodificar h64_mri con el decoder SPECT
        rec_id=decode_h64_via_bottleneck(spect_aae,hm)
        p_ident.append(torch.softmax(mnet(rec_id),1)[:,1].cpu().numpy())
        HS.append(hs.cpu().numpy().astype(np.float32)); HM.append(hm.cpu().numpy().astype(np.float32))
        Y.append(y)
    return (np.concatenate(HS),np.concatenate(HM),np.concatenate(Y),
            np.concatenate(p_gt),np.concatenate(p_recon),np.concatenate(p_ident))

def rep(name,y,p,thr=0.5):
    pred=(p>thr).astype(int)
    try: auc=roc_auc_score(y,p)
    except ValueError: auc=float('nan')
    print(f'{name:48s} BA={balanced_accuracy_score(y,pred):.4f} AUC={auc:.4f}')

print('\n### harvest TEST'); HS,HM,Y,pgt,praw,pid = harvest(os.environ['FST_TEST_LIST'],'test')
print('### harvest TRAIN'); HSt,HMt,Yt,pgtt,prawt,pidt = harvest(os.environ['FST_TRAIN_LIST'],'train')

print('\n--- GATES (TEST) ---')
rep('MobileNet3D @ SPECT real crudo',Y,praw)
rep('MobileNet3D @ decode(h64_spect)  [TECHO]',Y,pgt)
rep('MobileNet3D @ decode(h64_mri)    [identidad]',Y,pid)
print('--- GATES (TRAIN) ---')
rep('MobileNet3D @ decode(h64_spect)',Yt,pgtt)
rep('MobileNet3D @ decode(h64_mri)',Yt,pidt)

print('\n--- estadisticas h64 ---')
print('h64_spect shape',HS.shape,'min',HS.min(),'max',HS.max(),'mean',HS.mean(),'std',HS.std())
print('h64_mri   shape',HM.shape,'min',HM.min(),'max',HM.max(),'mean',HM.mean(),'std',HM.std())
print('var(h64_spect) [=MSE del predictor media global]:',HS.var())
mu=HSt.mean(0,keepdims=True)
print('MSE predecir media(TRAIN) sobre TEST:',((HS-mu)**2).mean())
print('MSE identidad h64_mri vs h64_spect (TEST):',((HM-HS)**2).mean())
print('frac ceros h64_spect:',(HS==0).mean(),' h64_mri:',(HM==0).mean())

print('\n--- sonda lineal: informacion de etiqueta en cada espacio ---')
def probe(Xtr,ytr,Xte,yte,name):
    a=Xtr.reshape(len(Xtr),-1).mean(1) # placeholder
    # GAP espacial -> 64 dims
    ftr=Xtr.mean(axis=(2,3,4)); fte=Xte.mean(axis=(2,3,4))
    sc=StandardScaler().fit(ftr)
    clf=LogisticRegression(max_iter=3000,C=1.0).fit(sc.transform(ftr),ytr)
    ptr=clf.predict_proba(sc.transform(ftr))[:,1]; pte=clf.predict_proba(sc.transform(fte))[:,1]
    print(f'{name:36s} train AUC={roc_auc_score(ytr,ptr):.4f} | TEST AUC={roc_auc_score(yte,pte):.4f} '
          f'BA={balanced_accuracy_score(yte,(pte>0.5).astype(int)):.4f}')
probe(HMt,Yt,HM,Y,'GAP(h64_MRI xmodal) -> label')
probe(HSt,Yt,HS,Y,'GAP(h64_SPECT)      -> label')

print('\n--- correlacion h64_mri vs h64_spect (por canal, GAP) ---')
gm=HMt.mean(axis=(2,3,4)); gs=HSt.mean(axis=(2,3,4))
cors=[np.corrcoef(gm[:,c],gs[:,c])[0,1] for c in range(64)]
cors=np.array(cors); cors=cors[np.isfinite(cors)]
print('corr por canal: mean=%.3f  |mean|=%.3f  max=%.3f  min=%.3f'%(cors.mean(),np.abs(cors).mean(),cors.max(),cors.min()))
np.save('/workspace/IL-FST/diag/h64_test_spect.npy',HS[:4])
print('\nOK')
