#!/usr/bin/env bash
set -uo pipefail
cd /workspace/IL-FST; source env.sh
C="data/feat_cache_z70_85"
R(){ n="$1"; shift; echo "########## $n"; python3 scripts/train_fst_clsfeat_v2.py \
  --feat-cache "$C" --out-dir "result/z70_85/$n" \
  --mid-channels 64 --l2 1e-2 --dropout 0.5 --lr 1e-4 "$@" 2>&1 | grep -E '^(fold|=== )'; }
mkdir -p result/z70_85
R best   --mse-weight 0.3
R mse0   --mse-weight 0.0
