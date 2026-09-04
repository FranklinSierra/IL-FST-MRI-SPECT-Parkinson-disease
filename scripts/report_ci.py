#!/usr/bin/env python3
"""Bootstrap de IC 95% para BA y AUC a nivel sujeto. Con n=34 el ruido es enorme."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

def boot(y,p,thr,B=5000,seed=0):
    rng=np.random.default_rng(seed); ba=[];auc=[]
    for _ in range(B):
        i=rng.integers(0,len(y),len(y))
        if len(np.unique(y[i]))<2: continue
        ba.append(balanced_accuracy_score(y[i],(p[i]>thr).astype(int)))
        auc.append(roc_auc_score(y[i],p[i]))
    q=lambda v:(float(np.percentile(v,2.5)),float(np.percentile(v,97.5)))
    return {"BA":float(balanced_accuracy_score(y,(p>thr).astype(int))),"BA_CI95":q(ba),
            "AUC":float(roc_auc_score(y,p)),"AUC_CI95":q(auc)}

for d in sys.argv[1:]:
    d=Path(d); f=d/"test_predictions.csv"
    if not f.exists(): print(f"{d.name}: sin test_predictions.csv"); continue
    df=pd.read_csv(f)
    pcol="prob" if "prob" in df else ("prob_pd" if "prob_pd" in df else "prob_1")
    if "subject_id" in df: df=df.groupby("subject_id").agg(label=("label","first"),p=(pcol,"mean")).reset_index()
    else: df=df.rename(columns={pcol:"p"})
    thr=json.load(open(d/"metrics_test.json")).get("threshold",0.5)
    r=boot(df.label.to_numpy(),df.p.to_numpy(),thr)
    print(f"{d.name:26s} BA={r['BA']:.3f} [{r['BA_CI95'][0]:.3f},{r['BA_CI95'][1]:.3f}]  "
          f"AUC={r['AUC']:.3f} [{r['AUC_CI95'][0]:.3f},{r['AUC_CI95'][1]:.3f}]  (thr={thr:.3f})")
    json.dump(r,open(d/"metrics_test_ci.json","w"),indent=2)
