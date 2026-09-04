#!/usr/bin/env python3
"""FST clsfeat v2: MRI conv_pw_13 -> SPECT conv_pw_13 -> head SPECT congelada.

Correcciones frente a v1 (ver diag/):
 1. La head es GAP->fc1->fc2: SOLO consume GAP(f). La MSE va sobre GAP, no sobre
    el mapa 8x8 (v1 gastaba el 98.4% de su capacidad en detalle que la head tira).
 2. MSE NORMALIZADA por var(GAP(f_spect)) del train -> mse_rel=1.0 significa
    "igual de bueno que predecir la media". v1 reportaba MSE cruda y estaba en 2.28
    contra 2.16 del predictor media, es decir: PEOR que constante, y nadie lo vio.
 3. Pesos invertidos: el camino util es el CE. La reconstruccion de features es
    imposible aqui (el mejor traductor lineal da R^2 ~ +0.01).
 4. CE con pesos de clase (58/89) y umbral calibrado out-of-fold, no 0.5 fijo.
 5. CV 5-fold a nivel SUJETO sobre train.txt; test.txt intocable.
    v1 hacia early stopping sobre 15 sujetos (1 sujeto = 6 pts de BA -> ruido puro).
 6. Early stopping sobre AUC (continua), no BA (escalonada en 15 sujetos).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score, accuracy_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import StratifiedKFold

FST=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(FST))
import tensorflow as tf
from fst.keras_fst_models import load_classifier, build_spect_head
from fst.keras_transfer_unet_v2 import build_transfer_module_v2
for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)

EXP_ID="fst_clsfeat_mnet2d_v2"

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feat-cache",type=Path,default=Path(os.environ.get("FST_FEAT_CACHE",FST/"data/feat_cache_pw13")))
    p.add_argument("--out-dir",type=Path,default=FST/f"result/{EXP_ID}")
    p.add_argument("--spect-classifier",type=Path,default=Path(os.environ["FST_CLSFEAT_SPECT_CKPT"]))
    p.add_argument("--train-list",type=Path,default=Path(os.environ["FST_TRAIN_LIST"]))
    p.add_argument("--test-list",type=Path,default=Path(os.environ["FST_TEST_LIST"]))
    p.add_argument("--folds",type=int,default=5)
    p.add_argument("--epochs",type=int,default=60)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--patience",type=int,default=12)
    p.add_argument("--mse-weight",type=float,default=0.3)
    p.add_argument("--ce-weight",type=float,default=1.0)
    p.add_argument("--mid-channels",type=int,default=192)
    p.add_argument("--l2",type=float,default=1e-4)
    p.add_argument("--dropout",type=float,default=0.2)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--z-min",type=int,default=None,help="filtra slice_idx >= z_min")
    p.add_argument("--z-max",type=int,default=None,help="filtra slice_idx <= z_max")
    return p.parse_args()

# ---------------------------------------------------------------- datos
def load_cache(cache: Path, train_ids, test_ids, z_min=None, z_max=None):
    idx=pd.read_csv(cache/"index.csv",dtype={"subject_id":str})
    Fm=np.load(cache/"f_mri.npy",mmap_mode="r")
    Gs=np.load(cache/"gap_spect.npy")
    keep=np.ones(len(idx),bool)
    if z_min is not None: keep &= (idx.slice_idx.values>=z_min)
    if z_max is not None: keep &= (idx.slice_idx.values<=z_max)
    tr=idx.subject_id.isin(set(train_ids)).values & keep
    te=idx.subject_id.isin(set(test_ids)).values & keep
    return idx,Fm,Gs,np.where(tr)[0],np.where(te)[0]

def subject_table(idx, rows):
    sub=idx.iloc[rows]
    g=sub.groupby("subject_id")["label"].first()
    return g.index.to_numpy(), g.to_numpy()

# ---------------------------------------------------------------- modelo
def build(args, spect_head):
    xfer=build_transfer_module_v2(mid_channels=args.mid_channels,l2=args.l2,dropout=args.dropout)
    inp=tf.keras.Input(shape=(8,8,1024),name="f_mri")
    f_hat=xfer(inp)
    gap=tf.keras.layers.GlobalAveragePooling2D(name="gap_hat")(f_hat)
    logits=spect_head(f_hat)
    return tf.keras.Model(inp,{"gap":gap,"logits":logits},name="fst_v2"),xfer

def make_train_step(model,opt,mse_w,ce_w,var):
    """tf.function por fold: el optimizador se construye FUERA del grafo."""
    opt.build(model.trainable_weights)
    @tf.function
    def step(x,gt_gap,y,w):
        with tf.GradientTape() as t:
            o=model(x,training=True)
            mse=tf.reduce_mean(tf.square(o["gap"]-gt_gap))/var      # 1.0 = predictor media
            ce=tf.reduce_mean(w*tf.keras.losses.categorical_crossentropy(y,o["logits"]))
            loss=mse_w*mse+ce_w*ce
            if model.losses:
                loss=loss+tf.add_n(model.losses)
        gr=t.gradient(loss,model.trainable_weights)
        opt.apply_gradients(zip(gr,model.trainable_weights))
        return loss,mse,ce
    return step


@tf.function
def _fwd(model,x): return model(x,training=False)

def predict_rows(model,Fm,rows,bs=64):
    ps=[];gs=[]
    for i in range(0,len(rows),bs):
        x=np.asarray(Fm[rows[i:i+bs]],dtype=np.float32)
        o=_fwd(model,x); ps.append(o["logits"].numpy()[:,1]); gs.append(o["gap"].numpy())
    return np.concatenate(ps),np.concatenate(gs)

def subj_prob(idx,rows,p):
    d=pd.DataFrame({"s":idx.subject_id.values[rows],"y":idx.label.values[rows],"p":p})
    g=d.groupby("s").agg(y=("y","first"),p=("p","mean"))
    return g.index.to_numpy(),g.y.to_numpy(),g.p.to_numpy()

def best_threshold(y,p):
    cand=np.unique(np.concatenate([p,[0.5]]))
    return float(max(cand,key=lambda t: balanced_accuracy_score(y,(p>t).astype(int))))

def metrics(y,p,thr):
    pred=(p>thr).astype(int)
    return {"n_subjects":int(len(y)),"threshold":float(thr),
            "accuracy":float(accuracy_score(y,pred)),
            "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
            "f1":float(f1_score(y,pred,average="macro",zero_division=0)),
            "roc_auc":float(roc_auc_score(y,p)) if len(np.unique(y))>1 else float("nan"),
            "confusion_matrix":confusion_matrix(y,pred).tolist()}

# ---------------------------------------------------------------- main
def main():
    a=parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    tf.keras.utils.set_random_seed(a.seed)
    train_ids=[l.strip() for l in a.train_list.read_text().split() if l.strip()]
    test_ids=[l.strip() for l in a.test_list.read_text().split() if l.strip()]
    idx,Fm,Gs,tr_rows,te_rows=load_cache(a.feat_cache,train_ids,test_ids,a.z_min,a.z_max)
    print(f"slices train={len(tr_rows)} test={len(te_rows)}",flush=True)

    spect_cls=load_classifier(a.spect_classifier)
    head=build_spect_head(spect_cls)

    s_tr,y_tr=subject_table(idx,tr_rows)
    var=float(Gs[tr_rows].var())
    cw={0:len(y_tr)/(2*np.sum(y_tr==0)),1:len(y_tr)/(2*np.sum(y_tr==1))}
    print(f"var(GAP f_spect)={var:.4f}  class_weights={cw}",flush=True)

    st_ids,y_te=subject_table(idx,te_rows)
    skf=StratifiedKFold(a.folds,shuffle=True,random_state=a.seed)
    oof_s=[];oof_y=[];oof_p=[]; test_p=[]; fold_rows=[]; hist_all=[]

    for k,(itr,iva) in enumerate(skf.split(s_tr,y_tr),1):
        fit_s=set(s_tr[itr]); val_s=set(s_tr[iva])
        rf=tr_rows[np.isin(idx.subject_id.values[tr_rows],list(fit_s))]
        rv=tr_rows[np.isin(idx.subject_id.values[tr_rows],list(val_s))]
        model,xfer=build(a,head)
        opt=tf.keras.optimizers.Adam(a.lr)
        step=make_train_step(model,opt,tf.constant(a.mse_weight,tf.float32),
                             tf.constant(a.ce_weight,tf.float32),tf.constant(var,tf.float32))
        yoh=tf.keras.utils.to_categorical(idx.label.values,2).astype(np.float32)
        wvec=np.array([cw[int(l)] for l in idx.label.values],np.float32)

        best_auc=-1;best_w=None;best_ep=0;left=a.patience
        rng=np.random.default_rng(a.seed+k)
        for ep in range(1,a.epochs+1):
            t0=time.time(); perm=rng.permutation(rf); tot=[0.,0.,0.];n=0
            for i in range(0,len(perm),a.batch_size):
                b=perm[i:i+a.batch_size]
                l,m,c=step(np.asarray(Fm[b],dtype=np.float32),Gs[b],yoh[b],wvec[b])
                tot[0]+=float(l)*len(b);tot[1]+=float(m)*len(b);tot[2]+=float(c)*len(b);n+=len(b)
            pv,gv=predict_rows(model,Fm,rv)
            _,yv,pvs=subj_prob(idx,rv,pv)
            auc=roc_auc_score(yv,pvs) if len(np.unique(yv))>1 else float("nan")
            mrel=float(((gv-Gs[rv])**2).mean()/var)
            hist_all.append({"fold":k,"epoch":ep,"loss":tot[0]/n,"mse_rel":tot[1]/n,
                             "ce":tot[2]/n,"val_auc":auc,"val_mse_rel":mrel,"sec":time.time()-t0})
            if np.isfinite(auc) and auc>best_auc+1e-6:
                best_auc=auc;best_w=[w.numpy() for w in model.weights];best_ep=ep;left=a.patience
            else:
                left-=1
                if left<=0: break
        model.set_weights(best_w)
        print(f"fold {k}: best_epoch={best_ep} val_AUC={best_auc:.4f} val_mse_rel={mrel:.4f}",flush=True)
        pv,_=predict_rows(model,Fm,rv); sv,yv,pvs=subj_prob(idx,rv,pv)
        oof_s+=list(sv);oof_y+=list(yv);oof_p+=list(pvs)
        pt,gt=predict_rows(model,Fm,te_rows); st,yt,pts=subj_prob(idx,te_rows,pt)
        assert list(st)==list(st_ids)
        test_p.append(pts)
        fold_rows.append({"fold":k,"best_epoch":best_ep,"val_auc":float(best_auc),
                          "test_auc":float(roc_auc_score(yt,pts)),
                          "test_mse_rel":float(((gt-Gs[te_rows])**2).mean()/var)})
        xfer.save(a.out_dir/f"transfer_fold{k}.keras")

    oof_y=np.array(oof_y);oof_p=np.array(oof_p)
    thr=best_threshold(oof_y,oof_p)                       # calibrado OUT-OF-FOLD
    cv=metrics(oof_y,oof_p,thr)
    pts=np.mean(test_p,axis=0)                            # ensemble de los 5 folds
    te=metrics(yt,pts,thr)
    te["mse_rel_test"]=float(np.mean([f["test_mse_rel"] for f in fold_rows]))
    cv["per_fold"]=fold_rows

    pd.DataFrame(hist_all).to_csv(a.out_dir/"train_history.csv",index=False)
    pd.DataFrame({"subject_id":oof_s,"label":oof_y,"prob":oof_p}).to_csv(a.out_dir/"oof_predictions.csv",index=False)
    pd.DataFrame({"subject_id":st,"label":yt,"prob":pts,"pred":(pts>thr).astype(int)}).to_csv(a.out_dir/"test_predictions.csv",index=False)
    (a.out_dir/"metrics_cv.json").write_text(json.dumps(cv,indent=2))
    (a.out_dir/"metrics_test.json").write_text(json.dumps(te,indent=2))
    (a.out_dir/"config.json").write_text(json.dumps(vars(a),indent=2,default=str))
    (a.out_dir/"classification_report_test.txt").write_text(
        classification_report(yt,(pts>thr).astype(int),digits=4,zero_division=0))
    print("\n=== CV out-of-fold (147 sujetos) ===");print(json.dumps({k:v for k,v in cv.items() if k!="per_fold"},indent=2))
    print("=== TEST (34 sujetos, ensemble 5 folds, umbral OOF) ===");print(json.dumps(te,indent=2))

if __name__=="__main__": main()
