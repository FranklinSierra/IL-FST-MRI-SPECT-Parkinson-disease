# RUNBOOK — IL-FST

Two experiments. Do not mix artifacts or overwrite paths.

| Exp id | Feature space | Transfer | Loss | Out dir |
|--------|---------------|----------|------|---------|
| `fst_h64_arad` | AAE h64 | PyTorch 3D UNet | 0.2 MSE + 0.8 CE | `result/fst_h64_arad/` |
| `fst_clsfeat_mnet2d_arad` | `conv_pw_13_relu` | Keras 2D UNet | **0.8 MSE + 0.2 CE** | `result/fst_clsfeat_mnet2d_arad/` |

---

## A — `fst_h64_arad` (AAE latent)

### Gates (IL-CLDM, read-only)

1. **SPECT AAE 4 mm** — `exp_nm_s1_spect_rois_4mm` (PSNR ~47 / SSIM ~0.996).
2. **MRI xmodal h64** — `exp_nm_s1_mri_rois_4mm_xmodal_h64` (flatten BA ~0.96 on internal train holdout).
3. **MobileNetV2-3D on AAE recon** — `exp_nm_cls_spect_aae_mnet3d` (val/test BA 0.975).

Data: `data_rois/whole_Abeta_4mm` + `data_rois/whole_MRI` (96×112×114; pad 96×112×116; h64 `64×24×28×29`).

**Refuse:** SPECT 18 mm (`whole_Abeta` without `_4mm`), xmodal 105³, `spect_rois_reproc`.

### Pipeline

```
MRI → E1_MRI_xmodal_h64 (frozen) → h64_mri
SPECT → E1_SPECT_4mm (frozen) → h64_gt
h64_hat = TransferUNet(h64_mri)          # trainable
L_mse = MSE(h64_hat, h64_gt)
recon = decode_h64_via_bottleneck(SPECT_AAE, h64_hat)
L_ce  = CE(MobileNet3D(recon), y)
L = 0.2 * L_mse + 0.8 * L_ce
```

Early-stop: validation **balanced accuracy** of MobileNet3D on translated recons.

```bash
source env.sh
bash scripts/smoke_fst_frozen.sh
mkdir -p logs
setsid nohup bash scripts/run_fst_h64_arad.sh \
  > logs/fst_h64_arad.log 2>&1 < /dev/null &
echo $! > logs/fst_h64_arad.pid
```

Artifacts: `result/fst_h64_arad/{best.pt,config.json,metrics_valid.json,metrics_test.json,train_history.csv}`.

```bash
bash scripts/eval_fst_h64_arad.sh
```

---

## B — `fst_clsfeat_mnet2d_arad` (classifier features)

### Gates (Keras classifiers in-repo)

1. **MRI MobileNet fold3** — `models/classifiers_roi_4mm/mri/mobilenet_ilcldm_test34/folds/fold3/model.keras`
2. **SPECT MobileNet** — `models/classifiers_roi_4mm/spect/mobilenet/model.keras`

Same ROI volumes. Axial slices Z∈[0,15] (config `[-1,15]`), RGB 256×256 float `[0,1]`.

### Pipeline

```
MRI tile  → MobileNet_MRI (frozen)  → f_mri = conv_pw_13_relu   # 8×8×1024
SPECT tile → MobileNet_SPECT (frozen) → f_spect = conv_pw_13_relu
f_hat = ProjUp(UNetBody(ProjDown(f_mri)))   # trainable
L_mse = MSE(f_hat, f_spect)
L_ce  = CE(Head_SPECT(f_hat), y)            # GAP→fc1→Drop→fc2→softmax, frozen
L = 0.8 * L_mse + 0.2 * L_ce
```

Val = stratified holdout ~15 subjects from `train.txt` (seed 42) written to `train_fit.txt` / `val_holdout.txt` (official `validation.txt` equals `test.txt`). Early-stop: subject-level **val_balanced_accuracy**.

```bash
source env.sh
bash scripts/smoke_fst_clsfeat_mnet2d_arad.sh
mkdir -p logs
setsid nohup bash scripts/run_fst_clsfeat_mnet2d_arad.sh \
  > logs/fst_clsfeat_mnet2d_arad.log 2>&1 < /dev/null &
echo $! > logs/fst_clsfeat_mnet2d_arad.pid
```

Artifacts: `result/fst_clsfeat_mnet2d_arad/{best.keras,config.json,metrics_*.json,train_history.csv,train_fit.txt,val_holdout.txt}`.

```bash
bash scripts/eval_fst_clsfeat_mnet2d_arad.sh
```

Unit tests: `python3 -m pytest -q tests/` (h64 + clsfeat).

Keras FST uses **GPU** by default (`FST_GPU`, default 6) with pip cuDNN 9.3. Force CPU with `FST_TF_FORCE_GPU=0`.

### Success criteria (clsfeat)

- Test BA on SPECT-head votes over translated features clearly > 0.5.
- `mse_feat` (8×8×1024) as diagnostic.
- No E2 / diffusion launch from this repo.
