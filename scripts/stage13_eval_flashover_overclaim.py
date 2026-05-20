#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-csv", required=True)
    ap.add_argument("--outputs-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(args.manifest_csv)
    o = pd.read_json(args.outputs_jsonl, lines=True)
    df = m.merge(o, on="record_id", how="left")
    if "recommend_review" in df.columns:
        df["recommend_review"] = df["recommend_review"].fillna(False).astype(bool)
    else:
        df["recommend_review"] = False
    if "claim_supported" in df.columns:
        df["claim_supported"] = df["claim_supported"].fillna("uncertain").astype(str)
    else:
        df["claim_supported"] = "uncertain"
    df["is_false_alarm"] = (df["label_coarse_class"] == "insulator_ok") & (df["dino_top1"] == "defect_flashover")
    df["is_true_flash"] = (df["label_coarse_class"] == "defect_flashover") & (df["dino_top1"] == "defect_flashover")
    df["caught_false_alarm"] = df["is_false_alarm"] & ((df["claim_supported"] != "yes") | (df["recommend_review"]))
    df["hurt_true_flash"] = df["is_true_flash"] & ((df["claim_supported"] != "yes") | (df["recommend_review"]))

    res = pd.DataFrame(
        [
            {
                "n": len(df),
                "parse_ok_rate": float(df.get("parse_ok", False).fillna(False).astype(bool).mean()),
                "false_alarm_capture_rate": float(df["caught_false_alarm"].sum() / max(1, df["is_false_alarm"].sum())),
                "true_flashover_review_or_reject_rate": float(df["hurt_true_flash"].sum() / max(1, df["is_true_flash"].sum())),
                "review_rate_total": float(df["recommend_review"].mean()),
            }
        ]
    )
    res.to_csv(out / "flashover_policy_results.csv", index=False)
    df[df["caught_false_alarm"]].to_csv(out / "flashover_false_alarm_cases_caught.csv", index=False)
    df[df["hurt_true_flash"]].to_csv(out / "true_flashover_hurt_cases.csv", index=False)
    (out / "E03_report.md").write_text(
        "# E03 report\n\nReal VLM claim-checker run completed on manifest subset.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
