#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.table)
    y = df["label_coarse_class"].astype(str).values
    sss = StratifiedShuffleSplit(
        n_splits=args.n_repeats,
        test_size=args.test_size,
        random_state=args.seed,
    )

    rows = []
    for i, (dev_idx, test_idx) in enumerate(sss.split(df, y)):
        dev = df.iloc[dev_idx].copy()
        test = df.iloc[test_idx].copy()

        split_name = f"split_{i:02d}"
        (out_dir / f"{split_name}_dev_ids.txt").write_text("\n".join(dev["record_id"].astype(str).tolist()), encoding="utf-8")
        (out_dir / f"{split_name}_test_ids.txt").write_text("\n".join(test["record_id"].astype(str).tolist()), encoding="utf-8")

        for subset, part in [("dev", dev), ("test", test)]:
            counts = part["label_coarse_class"].value_counts()
            rows.append(
                {
                    "split_id": split_name,
                    "subset": subset,
                    "n": int(len(part)),
                    "n_ok": int(counts.get("insulator_ok", 0)),
                    "n_flashover": int(counts.get("defect_flashover", 0)),
                    "n_broken": int(counts.get("defect_broken", 0)),
                }
            )

    pd.DataFrame(rows).to_csv(out_dir / "split_summary.csv", index=False)


if __name__ == "__main__":
    main()
