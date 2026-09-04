#!/usr/bin/env python3
"""Build 5 NHY-stratified subject folds over the full 181-subject pool (train+test).

Stratifies on `stratum` from nhy_distribution.csv: control / pd_nhy1 / pd_nhy2.
Does not use the CSV `split` column (that is the nhy_v1 146/35 holdout).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

FST = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--nhy-csv",
        type=Path,
        default=FST / "data" / "nhy_distribution.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=FST / "result" / "fst_clsfeat_mnet2d_v2_z78_112" / "ownKFold" / "files",
    )
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def counts(ids, strata):
    return dict(Counter(strata[i] for i in ids))


def main():
    a = parse_args()
    df = pd.read_csv(a.nhy_csv, dtype={"ID": str})
    if "stratum" not in df.columns:
        raise SystemExit(f"{a.nhy_csv} must have a stratum column")
    df["stratum"] = df["stratum"].fillna("control").astype(str)
    ids = sorted(df["ID"].tolist())
    strata = dict(zip(df["ID"], df["stratum"]))
    labels = dict(zip(df["ID"], df["label"].astype(int)))
    y = [strata[i] for i in ids]

    a.out_dir.mkdir(parents=True, exist_ok=True)
    skf = StratifiedKFold(a.folds, shuffle=True, random_state=a.seed)
    manifest = {
        "n_subjects": len(ids),
        "folds": a.folds,
        "seed": a.seed,
        "stratification": sorted(set(y)),
        "pool_counts": dict(Counter(y)),
        "nhy_csv": str(a.nhy_csv),
        "per_fold": [],
    }
    seen_test = []
    for k, (tr, te) in enumerate(skf.split(ids, y), 1):
        train_ids = sorted(ids[i] for i in tr)
        test_ids = sorted(ids[i] for i in te)
        overlap = set(train_ids) & set(test_ids)
        if overlap:
            raise SystemExit(f"fold {k} train/test overlap: {sorted(overlap)[:8]}")
        seen_test.extend(test_ids)
        (a.out_dir / f"trainFold{k}.txt").write_text("\n".join(train_ids) + "\n")
        (a.out_dir / f"testFold{k}.txt").write_text("\n".join(test_ids) + "\n")
        row = {
            "fold": k,
            "n_train": len(train_ids),
            "n_test": len(test_ids),
            "train_stratum": counts(train_ids, strata),
            "test_stratum": counts(test_ids, strata),
            "train_label": dict(Counter(labels[i] for i in train_ids)),
            "test_label": dict(Counter(labels[i] for i in test_ids)),
        }
        manifest["per_fold"].append(row)
        print(
            f"fold {k}: train={len(train_ids)} test={len(test_ids)} "
            f"test_stratum={row['test_stratum']}",
            flush=True,
        )

    if sorted(seen_test) != ids:
        missing = set(ids) - set(seen_test)
        extra = set(seen_test) - set(ids)
        raise SystemExit(f"test folds do not partition the pool missing={missing} extra={extra}")
    if len(seen_test) != len(set(seen_test)):
        raise SystemExit("duplicate subject in test folds")

    (a.out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
