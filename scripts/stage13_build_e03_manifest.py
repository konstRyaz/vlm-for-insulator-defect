#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-csv", default="outputs/stage10/vlm_topk_inference_manifest/stage10_vlm_eval_reference.csv")
    ap.add_argument("--out-csv", default="outputs/stage13_tradeoff_benefit_expansion/E03_flashover_overclaim_checker/flashover_claim_manifest.csv")
    ap.add_argument("--limit", type=int, default=120)
    args = ap.parse_args()

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.reference_csv)
    test = df[df["split"] == "val"].copy()

    fa = test[(test["dino_top1"] == "defect_flashover") & (test["label_coarse_class"] == "insulator_ok")].copy()
    tp = test[(test["dino_top1"] == "defect_flashover") & (test["label_coarse_class"] == "defect_flashover")].copy()
    ctrl = test[(test["dino_top2"] == "defect_flashover") & (test["dino_top1"] != "defect_flashover")].copy()

    n = args.limit
    n_fa = min(len(fa), int(n * 0.45))
    n_tp = min(len(tp), int(n * 0.35))
    n_ctrl = max(0, n - n_fa - n_tp)
    man = pd.concat([fa.head(n_fa), tp.head(n_tp), ctrl.head(n_ctrl)], ignore_index=True).drop_duplicates("record_id")
    man = man.head(n).copy()
    man["task"] = "flashover_claim_check"
    keep = [
        "record_id",
        "split",
        "resolved_image_path",
        "label_coarse_class",
        "dino_top1",
        "dino_top2",
        "dino_top3",
        "dino_top1_score",
        "dino_top2_score",
        "dino_top3_score",
        "dino_margin",
        "task",
    ]
    man[keep].to_csv(out, index=False)
    print(f"manifest rows: {len(man)}")


if __name__ == "__main__":
    main()

