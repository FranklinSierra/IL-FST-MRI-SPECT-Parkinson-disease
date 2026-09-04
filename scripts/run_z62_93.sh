#!/usr/bin/env bash
set -uo pipefail
cd /workspace/IL-FST; source env.sh
python3 scripts/train_fst_clsfeat_v2.py --feat-cache data/feat_cache_z62_93 \
  --out-dir result/z62_93/best --mid-channels 64 --l2 1e-2 --dropout 0.5 --lr 1e-4 \
  --mse-weight 0.3 2>&1 | grep -E '^(fold|=== )'
