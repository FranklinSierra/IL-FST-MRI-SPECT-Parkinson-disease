#!/usr/bin/env python3
"""FST h64 v2: h64_MRI -> h64_SPECT -> decoder AAE congelado -> MobileNetV2-3D congelada.

Correcciones frente a v1 (ver diag/):
 1. ce_weight 0.8 -> 0.05. El CE de v1 atravesaba un DECODER GENERATIVO congelado:
    el gradiente encontraba ataques adversariales (train_ce 0.48->0.15) que producian
    reconstrucciones basura (PSNR 15, SSIM 0.14) y no generalizaban.
 2. MSE normalizada por var(h64_spect). v1 reportaba MSE cruda 0.27-0.45 cuando el
    predictor de la MEDIA da 0.101: la UNet era 3-4x PEOR que una constante.
 3. UNet residual de 0.22M params con salida a cero (arranca en la identidad,
    AUC 0.646) en vez de la UNet de difusion de ~600M params.
 4. CV 5-fold a nivel sujeto sobre train.txt. v1 usaba validation.txt, que es
    IDENTICO a test.txt: hacia early stopping sobre el conjunto de test.
 5. Umbral calibrado out-of-fold; early stopping sobre AUC.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score, accuracy_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import StratifiedKFold

FST=Path(__file__).resolve().parents[1]; ILC=Path(os.environ.get("ILCLDM_ROOT","/workspace/IL-CLDM"))
sys.path.insert(0,str(ILC)); sys.path.insert(0,str(FST))
import importlib.util
_s=importlib.util.spec_from_file_location("trA", FST/"scripts/train_fst_h64_arad.py")
trA=importlib.util.module_from_spec(_s); _s.loader.exec_module(trA); trA._bootstrap_paths()
from fst.transfer_unet_v2 import build_transfer_unet_v2
from code.diffusion_features import decode_h64_via_bottleneck

EXP_ID="fst_h64_v2"

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h64-cache",type=Path,default=Path(os.environ.get("FST_H64_CACHE",FST/"data/h64_cache")))
    p.add_argument("--out-dir",type=Path,default=FST/f"result/{EXP_ID}")
    p.add_argument("--folds",type=int,default=5); p.add_argument("--epochs",type=int,default=40)
    p.add_argument("--batch-size",type=int,default=4); p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--patience",type=int,default=10)
    p.add_argument("--mse-weight",type=float,default=1.0)
    p.add_argument("--ce-weight",type=float,default=0.05)
    p.add_argument("--mid",type=int,default=32); p.add_argument("--dropout",type=float,default=0.1)
    p.add_argument("--weight-decay",type=float,default=1e-3); p.add_argument("--seed",type=int,default=42)
    return p.parse_args()

def best_threshold(y,p):
    return float(max(np.unique(np.concatenate([p,[0.5]])),
                     key=lambda t: balanced_accuracy_score(y,(p>t).astype(int))))

def metrics(y,p,thr):
    pred=(p>thr).astype(int)
    return {"n_subjects":int(len(y)),"threshold":float(thr),
            "accuracy":float(accuracy_score(y,pred)),
            "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
            "f1":float(f1_score(y,pred,average="macro",zero_division=0)),
            "roc_auc":float(roc_auc_score(y,p)) if len(np.unique(y))>1 else float("nan"),
            "confusion_matrix":confusion_matrix(y,pred).tolist()}

def main():
    a=parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev=torch.device("cuda")
    idx=pd.read_csv(a.h64_cache/"index.csv")
    Hm=np.load(a.h64_cache/"h64_mri.npy",mmap_mode="r")
    Hs=np.load(a.h64_cache/"h64_spect.npy",mmap_mode="r")
    tr=idx.index[idx.split=="train"].to_numpy(); te=idx.index[idx.split=="test"].to_numpy()
    y_all=idx.label.to_numpy()
    print(f"sujetos train={len(tr)} test={len(te)}",flush=True)

    spect_aae=trA.load_aae(os.environ["FST_SPECT_AAE"],dev)
    mnet=trA.load_mobilenet3d(os.environ["FST_MOBILENET3D"],dev)
    var=float(np.asarray(Hs[tr],dtype=np.float32).var())
    ytr=y_all[tr]
    cw=torch.tensor([len(ytr)/(2*(ytr==0).sum()),len(ytr)/(2*(ytr==1).sum())],
                    dtype=torch.float32,device=dev)
    print(f"var(h64_spect)={var:.4f} class_weights={cw.tolist()}",flush=True)
    ce_fn=nn.CrossEntropyLoss(weight=cw)

    def fwd(net,rows,train=False,bs=None):
        bs=bs or a.batch_size
        P=[];M=[]
        for i in range(0,len(rows),bs):
            r=rows[i:i+bs]
            hm=torch.as_tensor(np.asarray(Hm[r],dtype=np.float32),device=dev)
            hs=torch.as_tensor(np.asarray(Hs[r],dtype=np.float32),device=dev)
            hh=net(hm)
            logits=mnet(decode_h64_via_bottleneck(spect_aae,hh))
            P.append(torch.softmax(logits,1)[:,1].detach().cpu().numpy())
            M.append(float(((hh-hs)**2).mean())*len(r))
        return np.concatenate(P), sum(M)/len(rows)/var

    skf=StratifiedKFold(a.folds,shuffle=True,random_state=a.seed)
    oof_r=[];oof_p=[];test_p=[];folds=[];hist=[]
    for k,(itr,iva) in enumerate(skf.split(tr,ytr),1):
        rf,rv=tr[itr],tr[iva]
        net=build_transfer_unet_v2(mid=a.mid,dropout=a.dropout).to(dev)
        opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=a.weight_decay)
        best=-1;bw=None;bep=0;left=a.patience
        rng=np.random.default_rng(a.seed+k)
        for ep in range(1,a.epochs+1):
            t0=time.time(); net.train(); perm=rng.permutation(rf); tl=tm=tc=0.;n=0
            for i in range(0,len(perm),a.batch_size):
                r=perm[i:i+a.batch_size]
                hm=torch.as_tensor(np.asarray(Hm[r],dtype=np.float32),device=dev)
                hs=torch.as_tensor(np.asarray(Hs[r],dtype=np.float32),device=dev)
                yb=torch.as_tensor(y_all[r],dtype=torch.long,device=dev)
                hh=net(hm)
                mse=((hh-hs)**2).mean()/var                 # 1.0 = predictor media
                ce=ce_fn(mnet(decode_h64_via_bottleneck(spect_aae,hh)),yb)
                loss=a.mse_weight*mse+a.ce_weight*ce
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
                tl+=float(loss)*len(r);tm+=float(mse)*len(r);tc+=float(ce)*len(r);n+=len(r)
            net.eval()
            with torch.no_grad(): pv,mv=fwd(net,rv)
            auc=roc_auc_score(y_all[rv],pv) if len(np.unique(y_all[rv]))>1 else float("nan")
            hist.append({"fold":k,"epoch":ep,"loss":tl/n,"mse_rel":tm/n,"ce":tc/n,
                         "val_auc":auc,"val_mse_rel":mv,"sec":time.time()-t0})
            if np.isfinite(auc) and auc>best+1e-6:
                best=auc;bw={q:v.detach().clone() for q,v in net.state_dict().items()};bep=ep;left=a.patience
            else:
                left-=1
                if left<=0: break
        net.load_state_dict(bw); net.eval()
        with torch.no_grad():
            pv,mv=fwd(net,rv); pt,mt=fwd(net,te)
        print(f"fold {k}: best_epoch={bep} val_AUC={best:.4f} val_mse_rel={mv:.4f} test_AUC={roc_auc_score(y_all[te],pt):.4f}",flush=True)
        oof_r+=list(rv); oof_p+=list(pv); test_p.append(pt)
        folds.append({"fold":k,"best_epoch":bep,"val_auc":float(best),
                      "val_mse_rel":float(mv),"test_mse_rel":float(mt),
                      "test_auc":float(roc_auc_score(y_all[te],pt))})
        torch.save(net.state_dict(), a.out_dir/f"transfer_fold{k}.pt")

    oof_r=np.array(oof_r);oof_p=np.array(oof_p);yo=y_all[oof_r]
    thr=best_threshold(yo,oof_p)
    cv=metrics(yo,oof_p,thr); cv["per_fold"]=folds
    pt=np.mean(test_p,axis=0); yt=y_all[te]
    tem=metrics(yt,pt,thr); tem["mse_rel_test"]=float(np.mean([f["test_mse_rel"] for f in folds]))
    pd.DataFrame(hist).to_csv(a.out_dir/"train_history.csv",index=False)
    pd.DataFrame({"ID":idx.ID.values[oof_r],"label":yo,"prob":oof_p}).to_csv(a.out_dir/"oof_predictions.csv",index=False)
    pd.DataFrame({"ID":idx.ID.values[te],"label":yt,"prob":pt,"pred":(pt>thr).astype(int)}).to_csv(a.out_dir/"test_predictions.csv",index=False)
    (a.out_dir/"metrics_cv.json").write_text(json.dumps(cv,indent=2))
    (a.out_dir/"metrics_test.json").write_text(json.dumps(tem,indent=2))
    (a.out_dir/"config.json").write_text(json.dumps(vars(a),indent=2,default=str))
    (a.out_dir/"classification_report_test.txt").write_text(
        classification_report(yt,(pt>thr).astype(int),digits=4,zero_division=0))
    print("\n=== CV out-of-fold (147) ===");print(json.dumps({q:v for q,v in cv.items() if q!="per_fold"},indent=2))
    print("=== TEST (34, ensemble, umbral OOF) ===");print(json.dumps(tem,indent=2))

if __name__=="__main__": main()
