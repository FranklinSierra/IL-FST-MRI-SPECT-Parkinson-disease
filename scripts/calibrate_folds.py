#!/usr/bin/env python3
"""Calibracion por fold + voto, sobre modelos ya entrenados.

Problema detectado: cada fold produce probabilidades en una ESCALA distinta
(la head congelada esta en una region saturada). Agrupar las predicciones
out-of-fold en un solo vector y buscar un umbral global mezcla escalas y da
BA=0.5 aunque el AUC de test sea 0.79. Solucion estandar: umbral calibrado
DENTRO de cada fold sobre su propio validation, y voto mayoritario en test.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
FST=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(FST))
import os, tensorflow as tf
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score, accuracy_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
from fst.keras_fst_models import load_classifier, build_spect_head
for g in tf.config.list_physical_devices('GPU'): tf.config.experimental.set_memory_growth(g,True)

ap=argparse.ArgumentParser()
ap.add_argument("result_dir",type=Path)
ap.add_argument("--feat-cache",type=Path,required=True)
ap.add_argument("--z-min",type=int,default=None); ap.add_argument("--z-max",type=int,default=None)
ap.add_argument("--folds",type=int,default=5); ap.add_argument("--seed",type=int,default=42)
a=ap.parse_args()

idx=pd.read_csv(a.feat_cache/"index.csv",dtype={"subject_id":str})
Fm=np.load(a.feat_cache/"f_mri.npy",mmap_mode="r")
keep=np.ones(len(idx),bool)
if a.z_min is not None: keep&=idx.slice_idx.values>=a.z_min
if a.z_max is not None: keep&=idx.slice_idx.values<=a.z_max
tr_ids=[l.strip() for l in Path(os.environ["FST_TRAIN_LIST"]).read_text().split() if l.strip()]
te_ids=[l.strip() for l in Path(os.environ["FST_TEST_LIST"]).read_text().split() if l.strip()]
tr_rows=np.where(idx.subject_id.isin(set(tr_ids)).values&keep)[0]
te_rows=np.where(idx.subject_id.isin(set(te_ids)).values&keep)[0]
head=build_spect_head(load_classifier(os.environ["FST_CLSFEAT_SPECT_CKPT"]))

g=idx.iloc[tr_rows].groupby("subject_id")["label"].first()
s_tr,y_tr=g.index.to_numpy(),g.to_numpy()
gt=idx.iloc[te_rows].groupby("subject_id")["label"].first()
s_te,y_te=gt.index.to_numpy(),gt.to_numpy()

def probs(xfer,rows,bs=64):
    out=[]
    for i in range(0,len(rows),bs):
        x=np.asarray(Fm[rows[i:i+bs]],dtype=np.float32)
        out.append(head(xfer(x,training=False),training=False).numpy()[:,1])
    p=np.concatenate(out)
    d=pd.DataFrame({"s":idx.subject_id.values[rows],"p":p})
    m=d.groupby("s")["p"].mean()
    return m.index.to_numpy(),m.to_numpy()

def best_thr(y,p):
    return float(max(np.unique(np.concatenate([p,[0.5]])),
                     key=lambda t: balanced_accuracy_score(y,(p>t).astype(int))))

skf=StratifiedKFold(a.folds,shuffle=True,random_state=a.seed)
votes=[]; te_scores=[]; oof_rank=[]; oof_y=[]; per_fold=[]
for k,(itr,iva) in enumerate(skf.split(s_tr,y_tr),1):
    f=a.result_dir/f"transfer_fold{k}.keras"
    if not f.exists(): print(f"falta {f}"); sys.exit(1)
    xfer=tf.keras.models.load_model(str(f),compile=False)
    val_s=set(s_tr[iva]); rv=tr_rows[np.isin(idx.subject_id.values[tr_rows],list(val_s))]
    sv,pv=probs(xfer,rv); yv=np.array([y_tr[list(s_tr).index(x)] for x in sv])
    thr=best_thr(yv,pv)
    st,pt=probs(xfer,te_rows); assert list(st)==list(s_te)
    votes.append((pt>thr).astype(int)); te_scores.append(rankdata(pt)/len(pt))
    oof_rank+=list(rankdata(pv)/len(pv)); oof_y+=list(yv)
    per_fold.append({"fold":k,"threshold":thr,"val_auc":float(roc_auc_score(yv,pv)),
                     "val_ba":float(balanced_accuracy_score(yv,(pv>thr).astype(int))),
                     "test_auc":float(roc_auc_score(y_te,pt)),
                     "test_ba":float(balanced_accuracy_score(y_te,(pt>thr).astype(int)))})
    print(f"fold {k}: thr={thr:.4f} val_AUC={per_fold[-1]['val_auc']:.3f} "
          f"test_AUC={per_fold[-1]['test_auc']:.3f} test_BA={per_fold[-1]['test_ba']:.3f}",flush=True)

vote=(np.mean(votes,axis=0)>0.5).astype(int)     # voto mayoritario de los 5 folds
score=np.mean(te_scores,axis=0)                  # rank-ensemble (escala comparable)
oof_rank=np.array(oof_rank); oof_y=np.array(oof_y)

# Punto de operacion PRINCIPAL: cuantil que iguala la prevalencia del TRAIN.
# Invariante a la escala de cada fold (que es lo que rompe el umbral absoluto),
# y no mira ninguna etiqueta de test.
prior1=float((y_tr==1).mean())
thr_q=float(np.quantile(score,1.0-prior1))
pred_q=(score>thr_q).astype(int)
out={
 "operating_point":"cuantil = prevalencia clase 1 en train (%.3f)"%prior1,
 "test_ba_prior_quantile":float(balanced_accuracy_score(y_te,pred_q)),
 "test_accuracy_prior_quantile":float(accuracy_score(y_te,pred_q)),
 "test_f1_prior_quantile":float(f1_score(y_te,pred_q,average="macro",zero_division=0)),
 "test_cm_prior_quantile":confusion_matrix(y_te,pred_q).tolist(),
 "cv_oof_auc_rank_normalizada":float(roc_auc_score(oof_y,oof_rank)),
 "test_auc_rank_ensemble":float(roc_auc_score(y_te,score)),
 "test_ba_voto_mayoritario":float(balanced_accuracy_score(y_te,vote)),
 "test_accuracy_voto":float(accuracy_score(y_te,vote)),
 "test_f1_voto":float(f1_score(y_te,vote,average="macro",zero_division=0)),
 "test_confusion_matrix":confusion_matrix(y_te,vote).tolist(),
 "per_fold":per_fold,
 "test_ba_por_fold_media":float(np.mean([f["test_ba"] for f in per_fold])),
 "test_auc_por_fold_media":float(np.mean([f["test_auc"] for f in per_fold])),
}
(a.result_dir/"metrics_test_calibrated.json").write_text(json.dumps(out,indent=2))
pd.DataFrame({"subject_id":s_te,"label":y_te,"rank_score":score,"pred":pred_q,"pred_voto":vote}).to_csv(
    a.result_dir/"test_predictions_calibrated.csv",index=False)
(a.result_dir/"classification_report_test_calibrated.txt").write_text(
    classification_report(y_te,pred_q,digits=4,zero_division=0))
print("\n"+json.dumps({k:v for k,v in out.items() if k!="per_fold"},indent=2))
