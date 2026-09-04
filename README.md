# IL-FST — Feature Space Transfer (ARAD)

Sibling of [IL-CLDM](/workspace/IL-CLDM). Translates MRI classifier features into the SPECT
feature space with a small UNet, then classifies with a **frozen** SPECT head.

## Canonical experiment (reproducible)

| Field | Value |
|-------|--------|
| **Exp id** | `fst_clsfeat_mnet2d_v2_z78_112` |
| Feature layer | Keras MobileNet v1 `conv_pw_13_relu` → **(8, 8, 1024)** |
| Transfer | Keras UNet v2 (`fst/keras_transfer_unet_v2.py`) |
| Loss | \(L = 0.3\,\mathrm{MSE_{rel}}(\mathrm{GAP}(f_\mathrm{hat}), \mathrm{GAP}(f_\mathrm{spect})) + 1.0\,\mathrm{CE}(\mathrm{Head}_{SPECT}(f_\mathrm{hat}), y)\) |
| Axial range | **z ∈ [78, 112]** (35 slices / subject, axis=2) |
| Seed | **42** everywhere (holdout, classifiers, FST CV, t-SNE) |
| GPU | `FST_GPU=6` (default in `env.sh`) |
| Output | `result/fst_clsfeat_mnet2d_v2_z78_112/` |

Reference metrics from this protocol (subject-level):

| Split | BA | AUC | Acc | Notes |
|-------|----|-----|-----|--------|
| CV OOF (147 train) | ≈0.966 | ≈0.985 | ≈0.966 | umbral OOF ≈0.423 |
| Test ensemble 5 folds (34) | ≈0.689 | ≈0.761 | ≈0.735 | same threshold |
| Single model **fold 5** (recommended) | ≈0.664 | ≈0.764 | — | threshold ≈0.423 on fold val |

Gates (same z range):

| Model | Path | Test BA / AUC (approx.) |
|-------|------|-------------------------|
| MRI MobileNet | `models/classifiers_roi_4mm/mri/model.keras` | 0.60 / 0.80 |
| SPECT MobileNet | `models/classifiers_roi_4mm/spect/model.keras` | 0.93 / 0.99 |

---

## 0. Environment

```bash
cd /workspace/IL-FST
source env.sh          # sets CUDA_VISIBLE_DEVICES=$FST_GPU (default 6), data paths, PYTHONPATH
```

| Variable | Default |
|----------|---------|
| `ILCLDM_WHOLE_MRI` | `/workspace/IL-CLDM/data_rois/whole_MRI` |
| `ILCLDM_WHOLE_ABETA` | `/workspace/IL-CLDM/data_rois/whole_Abeta_4mm` |
| `FST_TRAIN_LIST` | `IL-CLDM/data_info/train.txt` (**n=147**) |
| `FST_TEST_LIST` | `IL-CLDM/data_info/test.txt` (**n=34**) |
| `FST_META_CSV` | `IL-CLDM/data_info/data_info.csv` (`ID`, `label`, `split`) |
| `FST_CLSFEAT_MRI_CKPT` | `models/classifiers_roi_4mm/mri/model.keras` |
| `FST_CLSFEAT_SPECT_CKPT` | `models/classifiers_roi_4mm/spect/model.keras` |

**Refuse:** SPECT 18 mm (`whole_Abeta` without `_4mm`). Volume shape ROI 4 mm: **96×112×114**.

Requires TensorFlow with GPU (cuDNN matching the installed TF), `nibabel`, `scikit-learn`, `matplotlib`.

---

## 1. Data splits (reproducibility)

Official lists from IL-CLDM are **not** re-shuffled:

- **Train pool:** `train.txt` (147 subjects).
- **Test:** `test.txt` (34 subjects) — **never** used for early stopping or threshold fitting.
- Note: `validation.txt` in IL-CLDM is identical to `test.txt`; do **not** use it as val.

### Classifier holdout (shared MRI + SPECT)

Stratified holdout from `train.txt` via `fst.slice_data.stratified_holdout`:

| Param | Value |
|-------|--------|
| `n_val` | **15** |
| `seed` | **42** |
| Fit / val sizes | **132 / 15** |
| Val class balance | **6 control + 9 PD** |

Persisted lists (reuse these exact files to match this run):

```
models/classifiers_roi_4mm/_shared_split_z78_112/train_fit.txt
models/classifiers_roi_4mm/_shared_split_z78_112/val_holdout.txt
models/classifiers_roi_4mm/_shared_split_z78_112/test.txt
```

Val holdout IDs (seed=42):

```
3115 3128 3354 3365 3389 3564 3577 3586 3779 3812 3823 3830 3835 3854 4067
```

### FST transfer CV

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` over the **147 train** subjects.
Each fold trains a transfer UNet; test probs are the **mean of the 5 fold models**.
Decision threshold is chosen on **OOF** train probabilities to maximize balanced accuracy
(recorded ≈ **0.4229**).

---

## 2. Step A — Train Keras MobileNet gates (MRI + SPECT)

Architecture (CoCoLIT-compatible): ImageNet MobileNet v1, freeze first 50 layers; head
`GAP → fc1(1024, ReLU, L2=0.01) → Dropout(0.5) → fc2(512, ReLU, L2=0.01) → predictions(2, softmax)`.
Feature for FST: **`conv_pw_13_relu`**.

| Hyperparameter | Value |
|----------------|--------|
| Slice range | `z_min=78`, `z_max=112` (35 slices) |
| Tile size | 256×256×3, float `[0,1]` |
| Preprocess MRI | volume z-score ±3σ → `[0,1]` |
| Preprocess SPECT | volume p1–p99 → `[0,1]` |
| Batch size | **12** |
| Epochs | **50** |
| LR (Adam) | **1e-4** |
| Patience | **10** (early stop on **subject-level val BA**, majority vote over slices) |
| Seed | **42** |
| Num classes | **2** (softmax) |

Launch both in parallel on one GPU:

```bash
source env.sh
bash scripts/run_train_keras_mobilenet_z78_112.sh
# logs: logs/train_keras_mobilenet_{mri,spect}_z78_112.log
```

Equivalent single-modality call:

```bash
python3 scripts/train_keras_mobilenet_roi_classifier.py \
  --modality mri \
  --volume-root "$ILCLDM_WHOLE_MRI" \
  --out-dir models/classifiers_roi_4mm/mri \
  --fit-list models/classifiers_roi_4mm/_shared_split_z78_112/train_fit.txt \
  --val-list models/classifiers_roi_4mm/_shared_split_z78_112/val_holdout.txt \
  --train-list "$FST_TRAIN_LIST" --test-list "$FST_TEST_LIST" --meta-csv "$FST_META_CSV" \
  --z-min 78 --z-max 112 --epochs 50 --batch-size 12 --patience 10 --seed 42 --lr 1e-4
# same for --modality spect --volume-root "$ILCLDM_WHOLE_ABETA" --out-dir .../spect
```

Artifacts: `models/classifiers_roi_4mm/{mri,spect}/{model.keras,config.json,metrics_*.json,...}`.

---

## 3. Step B — Cache features + train FST UNet (5-fold)

Pipeline:

```
MRI tile  → MobileNet_MRI (frozen)  → f_mri = conv_pw_13_relu
SPECT tile → MobileNet_SPECT (frozen) → f_spect
f_hat = TransferUNet_v2(f_mri)          # trainable
L_mse = mean((GAP(f_hat)-GAP(f_spect))²) / var(GAP(f_spect)_train)
L_ce  = CE(Head_SPECT(f_hat), y)        # class-weighted
L     = 0.3 * L_mse + 1.0 * L_ce (+ L2 on UNet)
```

| Hyperparameter | Value |
|----------------|--------|
| `z_min` / `z_max` | **78 / 112** |
| Feature cache | `data/feat_cache_z78_112/` (`f_mri.npy` float16, `gap_spect.npy`, `index.csv`) |
| Folds | **5**, `random_state=42` |
| Epochs (max) | **60** |
| Patience | **12** (early stop on **fold-val subject AUC**) |
| Batch size | **32** |
| LR (Adam) | **3e-4** |
| `mse_weight` / `ce_weight` | **0.3 / 1.0** |
| UNet `mid_channels` | **192** |
| UNet `l2` / `dropout` | **1e-4 / 0.2** |
| Residual + ReLU6 out | yes (`proj_up` init ≈ 0 → starts near identity) |
| Seed | **42** |

One-shot launcher (cache then train):

```bash
source env.sh
nohup bash scripts/run_fst_clsfeat_z78_112.sh \
  > logs/run_fst_clsfeat_z78_112_wrapper.log 2>&1 &
```

Or manually:

```bash
export FST_ZMIN=78 FST_ZMAX=112 FST_ZSTEP=1
export FST_FEAT_CACHE_Z="$FST_ROOT/data/feat_cache_z78_112"
python3 -u scripts/cache_clsfeat_z.py

python3 -u scripts/train_fst_clsfeat_v2.py \
  --feat-cache data/feat_cache_z78_112 \
  --out-dir result/fst_clsfeat_mnet2d_v2_z78_112 \
  --spect-classifier models/classifiers_roi_4mm/spect/model.keras \
  --z-min 78 --z-max 112 \
  --folds 5 --epochs 60 --batch-size 32 --lr 3e-4 --patience 12 \
  --mse-weight 0.3 --ce-weight 1.0 \
  --mid-channels 192 --l2 1e-4 --dropout 0.2 --seed 42
```

### Aggregation for metrics

1. Per-slice softmax / PD probability from SPECT head on `f_hat`.
2. **Subject score** = mean probability over that subject’s 35 slices (not majority vote).
3. CV OOF / test: `pred = (score > threshold_OOF)`.
4. Reported test table uses **ensemble** = mean of the 5 fold subject scores, then the same OOF threshold.

Artifacts under `result/fst_clsfeat_mnet2d_v2_z78_112/`:

- `transfer_fold{1..5}.keras`
- `metrics_cv.json`, `metrics_test.json`, `train_history.csv`
- `oof_predictions.csv`, `test_predictions.csv`, `classification_report_test.txt`
- `config.json` (exact CLI args)

**Single-model pick:** `transfer_fold5.keras` (best operating point among folds on test when threshold is calibrated on that fold’s val).

---

## 4. Step C — Figures (Fig. 10 style)

```bash
# MRI / translated fold5 / SPECT t-SNE (test), joint embedding, seed=42
python3 scripts/plot_tsne_features_fig10.py
# → result/fst_clsfeat_mnet2d_v2_z78_112/tsne_fig10/tsne_test_{slices,subjects}.png

# PD colored by Hoehn–Yahr + distance to SPECT control centroid
python3 scripts/plot_tsne_subjects_by_nhy.py
# NHY labels: IL-CLDM/data_info/splits/nhy_v1/nhy_distribution.csv
```

t-SNE: slice perplexity **30**, subject perplexity **8**, `random_state=42`.

Notebooks: `code/dataAnalysis.ipynb` (slices / Z range), `code/tsne_features_fig10.ipynb`.

---

## 5. Checklist to match this run

1. Same ROI volumes and `train.txt` / `test.txt`.
2. Reuse `_shared_split_z78_112/` (or regenerate with `stratified_holdout(..., n_val=15, seed=42)`).
3. Classifiers: z=[78,112], batch 12, lr 1e-4, patience 10, seed 42 → `model.keras` in `mri/` and `spect/`.
4. Feature cache with **step=1** over 78…112 from those checkpoints.
5. FST v2 with the hyperparams table above, seed 42, 5-fold.
6. Report **test ensemble** (and optionally fold 5 alone); do not early-stop on `validation.txt`.

---

## Layout

| Path | Role |
|------|------|
| `env.sh` | Paths + GPU |
| `scripts/train_keras_mobilenet_roi_classifier.py` | Gate training |
| `scripts/run_train_keras_mobilenet_z78_112.sh` | Parallel MRI+SPECT |
| `scripts/cache_clsfeat_z.py` | Feature cache |
| `scripts/train_fst_clsfeat_v2.py` | Transfer CV |
| `scripts/run_fst_clsfeat_z78_112.sh` | Cache + train |
| `fst/keras_transfer_unet_v2.py` / `keras_fst_models.py` / `slice_data.py` | Models + splits |
| `models/classifiers_roi_4mm/{mri,spect}/` | Gates |
| `data/feat_cache_z78_112/` | Cached features |
| `result/fst_clsfeat_mnet2d_v2_z78_112/` | FST results + `tsne_fig10/` |

Does **not** modify IL-CLDM registry, E2, or `roi_profile_env.sh`.

---

## Legacy experiments (not the canonical recipe)

| Exp | Notes |
|-----|--------|
| `fst_clsfeat_mnet2d_arad` | Old z=`[-1,15]` + inverted loss weights; MRI on blank slices |
| `fst_h64_arad` / `fst_h64_v2` | AAE h64 3D path (different feature space); see `RUNBOOK_FST.md` / `RUNBOOK_V2.md` |

For historical runbooks: `RUNBOOK_FST.md`, `RUNBOOK_V2.md`.
