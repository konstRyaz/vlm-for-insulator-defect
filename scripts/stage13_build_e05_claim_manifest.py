#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def claim_for_row(dino_top1: str) -> tuple[str, str]:
    if dino_top1 == "defect_flashover":
        return "flashover_surface_evidence", "Visible burn/arc/dark trace is attached to insulator surface."
    if dino_top1 == "defect_broken":
        return "broken_structure_evidence", "Visible missing fragment/crack/edge discontinuity exists."
    return "ok_intact_evidence", "Visible insulator appears intact with no direct defect evidence."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-csv", default="outputs/stage10/vlm_topk_inference_manifest/stage10_vlm_eval_reference.csv")
    ap.add_argument("--out-csv", default="outputs/stage13_tradeoff_benefit_expansion/E05_claim_verification/claim_manifest.csv")
    ap.add_argument("--limit", type=int, default=180)
    args = ap.parse_args()

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(args.reference_csv)
    dev = ref[ref["split"] == "train"].copy()
    hard = dev[(dev["dino_top1"] != dev["label_coarse_class"]) | (dev["dino_margin"] < 0.2)].copy()
    ctrl = dev[(dev["dino_top1"] == dev["label_coarse_class"]) & (dev["dino_margin"] > 0.5)].copy()
    n = args.limit
    nh = min(len(hard), int(n * 0.65))
    nc = max(0, n - nh)
    sel = pd.concat([hard.head(nh), ctrl.head(nc)], ignore_index=True).drop_duplicates("record_id").head(n)
    sel["claim_id"] = sel["dino_top1"].map(lambda x: claim_for_row(str(x))[0])
    sel["claim_text"] = sel["dino_top1"].map(lambda x: claim_for_row(str(x))[1])
    keep = [
        "record_id",
        "split",
        "resolved_image_path",
        "label_coarse_class",
        "dino_top1",
        "dino_top2",
        "dino_top3",
        "dino_margin",
        "claim_id",
        "claim_text",
    ]
    sel[keep].to_csv(out, index=False)
    print(f"manifest rows: {len(sel)}")


if __name__ == "__main__":
    main()
