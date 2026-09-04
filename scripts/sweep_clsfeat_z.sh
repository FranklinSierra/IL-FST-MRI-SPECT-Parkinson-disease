#!/usr/bin/env bash
set -uo pipefail
cd /workspace/IL-FST; source env.sh
C="data/feat_cache_z"
R(){ n="$1"; shift; echo "########## $n"; python3 scripts/train_fst_clsfeat_v2.py \
  --feat-cache "$C" --out-dir "result/sweep_z/$n" \
  --mid-channels 64 --l2 1e-2 --dropout 0.5 --lr 1e-4 --mse-weight 0.3 "$@" 2>&1 \
  | grep -E '^(fold|  "balanced|  "roc_auc|  "mse_rel|=== )'; }
mkdir -p result/sweep_z
R z_all
R z_sup      --z-min 78
R z_mid      --z-min 18 --z-max 47
R z_all_mse0 --mse-weight 0.0
R z_all_w128 --mid-channels 128 --l2 3e-3
