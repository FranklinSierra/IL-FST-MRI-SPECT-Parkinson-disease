#!/usr/bin/env bash
set -uo pipefail
cd /workspace/IL-FST; source env.sh
run(){ n="$1"; shift; echo "########## $n"; python3 scripts/train_fst_clsfeat_v2.py \
  --out-dir "result/sweep_v2/$n" "$@" 2>&1 | grep -E '^(fold|  "balanced|  "roc_auc|  "mse_rel|=== )'; }
mkdir -p result/sweep_v2
run reg_hi   --mid-channels 64  --l2 1e-2 --dropout 0.5 --lr 1e-4 --mse-weight 0.3
run reg_mid  --mid-channels 128 --l2 1e-3 --dropout 0.4 --lr 2e-4 --mse-weight 0.3
run ce_only  --mid-channels 128 --l2 1e-3 --dropout 0.4 --lr 2e-4 --mse-weight 0.0
run mse_hi   --mid-channels 128 --l2 1e-3 --dropout 0.4 --lr 2e-4 --mse-weight 3.0
