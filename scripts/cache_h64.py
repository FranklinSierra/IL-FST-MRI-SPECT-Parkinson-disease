#!/usr/bin/env python3
"""Cachea h64_mri y h64_spect (64,24,28,29) de los AAE congelados, float16."""
import os, sys
from pathlib import Path
import numpy as np, torch
FST=Path(__file__).resolve().parents[1]; ILC=Path(os.environ.get("ILCLDM_ROOT","/workspace/IL-CLDM"))
sys.path.insert(0,str(ILC)); sys.path.insert(0,str(FST))
import importlib.util
spec=importlib.util.spec_from_file_location("trA", FST/"scripts/train_fst_h64_arad.py")
trA=importlib.util.module_from_spec(spec); spec.loader.exec_module(trA); trA._bootstrap_paths()
from dataset import TwoDataset
from torch.utils.data import DataLoader

OUT=Path(os.environ.get("FST_H64_CACHE", FST/"data/h64_cache")); OUT.mkdir(parents=True,exist_ok=True)
dev=torch.device("cuda")
spect_aae=trA.load_aae(os.environ["FST_SPECT_AAE"],dev); mri_aae=trA.load_aae(os.environ["FST_MRI_XMODAL"],dev)
ids=[]
for f in (os.environ["FST_TRAIN_LIST"],os.environ["FST_TEST_LIST"]):
    ids+= [l.strip() for l in Path(f).read_text().split() if l.strip()]
SH=(64,24,28,29); N=len(ids)
Hm=np.lib.format.open_memmap(OUT/"h64_mri.npy",mode="w+",dtype=np.float16,shape=(N,*SH))
Hs=np.lib.format.open_memmap(OUT/"h64_spect.npy",mode="w+",dtype=np.float16,shape=(N,*SH))
meta=[]
i=0
for lst in (os.environ["FST_TRAIN_LIST"],os.environ["FST_TEST_LIST"]):
    ds=TwoDataset(root_MRI=os.environ["ILCLDM_WHOLE_MRI"],root_Abeta=os.environ["ILCLDM_WHOLE_ABETA"],
                  task=str(lst),stage="validation")
    split="train" if "train" in Path(lst).name else "test"
    with torch.no_grad():
        for mri,spect,names,labels in DataLoader(ds,batch_size=2,shuffle=False,num_workers=0):
            mri=torch.as_tensor(np.asarray(mri),dtype=torch.float32,device=dev).unsqueeze(1)
            spect=torch.as_tensor(np.asarray(spect),dtype=torch.float32,device=dev).unsqueeze(1)
            hm=trA.encode_h64(mri_aae,mri).cpu().numpy().astype(np.float16)
            hs=trA.encode_h64(spect_aae,spect).cpu().numpy().astype(np.float16)
            b=len(hm); Hm[i:i+b]=hm; Hs[i:i+b]=hs
            for j,nm in enumerate(names):
                meta.append({"row":i+j,"ID":str(nm),"subject_id":str(nm).split(".")[0],
                             "label":int(labels.reshape(-1)[j]),"split":split})
            i+=b
            if i%40==0: print(i,"/",N,flush=True)
Hm.flush(); Hs.flush()
import pandas as pd; pd.DataFrame(meta).to_csv(OUT/"index.csv",index=False)
print("guardado",Hm.shape,"->",OUT,flush=True)
