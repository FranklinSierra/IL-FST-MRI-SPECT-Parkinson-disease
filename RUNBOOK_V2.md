# RUNBOOK v2 — IL-FST corregido

Diagnóstico completo: `report/diagnostico.html`. Scripts de diagnóstico: `diag/`.

## Hallazgo principal

`z∈[0,15]` **no contiene cerebro en MRI** (fracción de señal 0.000–0.030; el cerebro
empieza en z=18). El rango viene del pipeline CoCoLIT, donde es válido para SPECT.
El MobileNet de MRI se entrenó sobre láminas en blanco — su propio
`models/classifiers_roi_4mm/mri/mobilenet_ilcldm_test34/config.json` reporta
`test BA = 0.511` y `mean_cv_valid_BA = 0.603`.

Información de etiqueta en las features (sonda logística, train→test):

| ventana | features MRI | features SPECT |
|---|---|---|
| `z=0..15` (original) | 0.518 | 0.982 |
| `z=18..47` | 0.743 | 0.986 |
| `z=78..112` | 0.804 | 0.971 |
| `z=18..112` | 0.779 | 0.996 |

## Elección de la ventana axial

Todo medido en la misma base: media de probabilidades por sujeto sobre el test oficial
(34 sujetos), umbral por cuantil del prior de train. «sonda» = regresión logística sobre
GAP(features), calculada sobre los MISMOS cortes que usó cada entrenamiento. Los
clasificadores MobileNet nunca se reentrenaron (siguen siendo los de `z=0..15`); solo
cambia qué cortes se les da de entrada. La sonda es una referencia, no una cota: la
UNet puede superarla, y en `z=18..47` lo hace.

| ventana | cortes | sonda MRI | sonda SPECT | AUC UNet | BA UNet |
|---|---|---|---|---|---|
| `z=0..15` (original) | 16 | 0.518 | 0.982 | 0.557 | 0.514 |
| `z=70..85` | 16 | 0.746 | **1.000** | 0.668 | 0.636 |
| `z=18..47` | 10 | 0.671 | 0.982 | 0.721 | 0.696 |
| `z=78..112` | 12 | **0.811** | 0.975 | 0.686 | 0.636 |
| `z=62..93` | 32 | 0.764 | 0.996 | 0.764 | 0.636 |
| **`z=18..112`** (paso 3) | **32** | 0.750 | 0.996 | **0.789** | 0.636 |

**El número de cortes importa tanto como dónde cortas.** Comparación controlada:
`z=70..85` y `z=62..93` cubren la misma zona anatómica y tienen la misma información
(sonda 0.746 vs 0.764), pero con 16 y 32 cortes respectivamente. Resultado:

```
z=70..85 (16 cortes)  AUC 0.668   folds [0.41, 0.54, 0.66, 0.68, 0.39]  <- dos bajo el azar
z=62..93 (32 cortes)  AUC 0.764   folds [0.78, 0.65, 0.82, 0.61, 0.74]
```

Con 16 cortes hay 2112 muestras de entrenamiento y el early stopping sobre folds de
29 sujetos elige épocas casi al azar. Usa **32 cortes como mínimo**.

Recomendación: `z=18..112` paso 3 si quieres el mejor número; `z=62..93` si prefieres
una ventana compacta y anatómicamente motivada — pierde 0.025 de AUC.

Con n=34 el IC 95% de la BA es ±0.15: las BA de la tabla **no son distinguibles entre sí**.
Elige por AUC y por el argumento anatómico, no por décimas de BA.

## Reproducir

```bash
source env.sh

# 1. Diagnósticos (opcional, ~15 min)
python3 diag/d1_clsfeat_baselines.py    # compuertas congeladas
python3 diag/d3_oracle.py               # oráculo lineal en clsfeat
python3 diag/d5_oracle_h64.py           # oráculo lineal en h64
python3 diag/d6_slice_range.py          # barrido de rango axial  <-- el importante

# 2. Enfoque B (clsfeat) — el recuperable
python3 scripts/cache_clsfeat_z.py                       # caché con z=18..112 step 3
# otra ventana:  FST_ZMIN=62 FST_ZMAX=93 FST_ZSTEP=1 \
#                FST_FEAT_CACHE_Z=data/feat_cache_z62_93 python3 scripts/cache_clsfeat_z.py
python3 scripts/train_fst_clsfeat_v2.py \
    --feat-cache data/feat_cache_z --out-dir result/fst_clsfeat_v2_z \
    --mid-channels 64 --l2 1e-2 --dropout 0.5 --lr 1e-4 --mse-weight 0.3
python3 scripts/calibrate_folds.py result/fst_clsfeat_v2_z --feat-cache data/feat_cache_z
python3 scripts/report_ci.py result/fst_clsfeat_v2_z

# 3. Enfoque A (h64) — ya está en su techo
python3 scripts/cache_h64.py
python3 scripts/train_fst_h64_v2.py --epochs 30 --patience 8
```

## Qué cambió en el código

| Archivo | Cambio |
|---|---|
| `fst/keras_transfer_unet_v2.py` | UNet residual, conv final a cero (arranca en la identidad), salida ReLU6, L2+dropout, cuello 1024→192 en vez de 1024→64 |
| `fst/transfer_unet_v2.py` | UNet 3D residual de 0.22M params (v1: ~600M, checkpoint de 2.4 GB), salida ReLU |
| `scripts/train_fst_clsfeat_v2.py` | MSE sobre GAP (la head solo consume GAP), CV 5-fold a nivel sujeto, early stop sobre AUC, pesos de clase, filtro `--z-min/--z-max` |
| `scripts/train_fst_h64_v2.py` | `ce_weight` 0.8→0.05 (el CE atravesaba un decoder generativo → ataque adversarial), MSE normalizada, CV 5-fold |
| `scripts/calibrate_folds.py` | Umbral por cuantil del prior de train; cada fold produce probabilidades en escala distinta y el umbral absoluto no transfiere |
| `scripts/report_ci.py` | IC 95% bootstrap — con n=34 la BA tiene ±0.15 |

## Reglas que conviene fijar

1. **Nunca reportes una MSE cruda.** Divide por la MSE del predictor de media del train.
   Si no baja de 1.0, el modelo no ha aprendido. v1 estaba en 4.46× (A) y 1.06× (B)
   y nadie podía verlo.
2. **`validation.txt` es idéntico a `test.txt`.** No lo uses para early stopping.
   Usa la CV 5-fold sobre `train.txt`.
3. **Mide siempre los encoders congelados en el test oficial**, no en holdout del train.
   Los dos encoders de MRI dan AUC ≈0.95 en train y ≈0.64–0.72 en test.
4. **El peso del CE depende de la libertad adversarial de la cadena congelada.**
   Alto si solo cruza `GAP→fc1→fc2`; bajo si cruza un decoder generativo.

## Techo de cada enfoque

- **B**: recuperable. Con el encoder actual (entrenado en blanco) llega a AUC 0.796.
  Reentrenar los MobileNet sobre `z=18..112` debería subirlo más.
- **A**: en su techo, AUC ≈0.67. El oráculo lineal en h64 da AUC 0.632, *peor que la
  identidad* (0.646), con R² negativo. El cuello de botella es el AAE xmodal de MRI,
  no el traductor. Ninguna UNet lo arregla.


## Restaurar los clasificadores congelados

`models/classifiers_roi_4mm/` se vació el 26/08/2026. Hay copias idénticas en CoCoLIT
(mismo `protocol: ilcldm_test34`, `selected_fold: 3`, `mean_cv_valid_BA: 0.6034016636957814`).
Ocupan ~409 MB.

```bash
S=/workspace/CoCoLIT/results/exp_cocolit_data_rois_4mm/classifiers
D=/workspace/IL-FST/models/classifiers_roi_4mm
mkdir -p "$D/spect" "$D/mri"
cp -r "$S/spect/mobilenet"                  "$D/spect/"
cp -r "$S/mri/mobilenet_ilcldm_test34"      "$D/mri/"

# verificar
python3 -c "import json;d=json.load(open('$D/mri/mobilenet_ilcldm_test34/config.json'));print(d['selected_fold'],d['mean_cv_valid_balanced_accuracy'])"
ls "$D/mri/mobilenet_ilcldm_test34/folds/fold3/model.keras" "$D/spect/mobilenet/model.keras"
```

Los entrenamientos usan features cacheadas y no necesitan los modelos; solo hacen falta
para `cache_clsfeat_z.py` (extraer features) y `calibrate_folds.py` (aplicar la head).
