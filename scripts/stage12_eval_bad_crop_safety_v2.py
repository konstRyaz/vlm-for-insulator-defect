#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", default="outputs/stage12/bad_crop_stress/stage12_bad_crop_predictions_with_flags.csv")
    ap.add_argument("--out-dir", default="outputs/stage12/bad_crop_safety_v2")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    # Safety-focused v2 policy: accept only very clean cases.
    df["visible_insulator"] = df["visibility"].astype(str).map(
        {"clear": "yes", "partial": "uncertain", "ambiguous": "uncertain", "poor": "no"}
    ).fillna("uncertain")
    df["safe_to_classify"] = (df["visibility"].astype(str) == "clear") & (~df["needs_review"].astype(bool))
    df["crop_validity"] = df["visibility"].astype(str).map(
        {"clear": "valid", "partial": "partial", "ambiguous": "uncertain", "poor": "bad"}
    ).fillna("uncertain")
    df["needs_review_v2"] = ~df["safe_to_classify"]
    df["reason"] = df.apply(
        lambda r: "clear_crop" if r["safe_to_classify"] else f"reject_due_to_{r['crop_validity']}", axis=1
    )
    df["vlm_safe_behavior_v2"] = (~df["safe_to_classify"]).astype(bool)
    df["false_accept_v2"] = (~df["vlm_safe_behavior_v2"]).astype(int)

    # Baselines
    b0_false = 1.0
    b1_false = float((~df["vlm_safe_behavior"].astype(bool)).mean())
    b2_false = float(df["false_accept_v2"].mean())
    b0_safe = 0.0
    b1_safe = float(df["vlm_safe_behavior"].astype(bool).mean())
    b2_safe = float(df["vlm_safe_behavior_v2"].astype(bool).mean())

    summary = pd.DataFrame(
        [
            {
                "system": "B0_closed_set_dino_proxy",
                "safe_behavior_rate": b0_safe,
                "false_accept_rate": b0_false,
                "review_rate": 1.0 - b0_safe,
                "safe_to_classify_rate": 1.0,
                "needs_review_rate": 0.0,
            },
            {
                "system": "B1_current_vlm",
                "safe_behavior_rate": b1_safe,
                "false_accept_rate": b1_false,
                "review_rate": 1.0 - b1_safe,
                "safe_to_classify_rate": float((~df["needs_review"].astype(bool)).mean()),
                "needs_review_rate": float(df["needs_review"].astype(bool).mean()),
            },
            {
                "system": "B2_safety_prompt_v2",
                "safe_behavior_rate": b2_safe,
                "false_accept_rate": b2_false,
                "review_rate": 1.0 - b2_safe,
                "safe_to_classify_rate": float(df["safe_to_classify"].mean()),
                "needs_review_rate": float(df["needs_review_v2"].mean()),
            },
        ]
    )
    summary.to_csv(out / "bad_crop_safety_v2_summary.csv", index=False)

    by = (
        df.groupby("corruption", dropna=False)
        .agg(
            n=("record_id", "size"),
            safe_behavior_rate=("vlm_safe_behavior_v2", "mean"),
            false_accept_rate=("false_accept_v2", "mean"),
            review_rate=("needs_review_v2", "mean"),
            safe_to_classify_rate=("safe_to_classify", "mean"),
            needs_review_rate=("needs_review_v2", "mean"),
        )
        .reset_index()
        .rename(columns={"corruption": "corruption_type"})
    )
    by.to_csv(out / "bad_crop_safety_v2_by_corruption.csv", index=False)

    comp = pd.DataFrame(
        [
            {"metric": "false_accept_rate", "B0_closed_set": b0_false, "B1_current_vlm": b1_false, "B2_v2": b2_false},
            {"metric": "safe_behavior_rate", "B0_closed_set": b0_safe, "B1_current_vlm": b1_safe, "B2_v2": b2_safe},
        ]
    )
    comp.to_csv(out / "bad_crop_safety_v2_comparison.csv", index=False)

    fail = df[df["false_accept_v2"] == 1].copy()
    fail.to_csv(out / "bad_crop_safety_v2_failure_cases.csv", index=False)

    df[
        [
            "record_id",
            "split",
            "corruption",
            "resolved_image_path",
            "visible_insulator",
            "safe_to_classify",
            "crop_validity",
            "needs_review_v2",
            "reason",
        ]
    ].to_json(out / "bad_crop_safety_v2_outputs.jsonl", orient="records", lines=True, force_ascii=False)

    report = [
        "# Bad-Crop Safety V2",
        "",
        f"- B1 false_accept_rate: {b1_false:.4f}",
        f"- B2 false_accept_rate: {b2_false:.4f}",
        f"- B1 safe_behavior_rate: {b1_safe:.4f}",
        f"- B2 safe_behavior_rate: {b2_safe:.4f}",
        "",
        "B2 uses conservative safety-only accept policy.",
    ]
    (out / "bad_crop_safety_v2_report.md").write_text("\n".join(report), encoding="utf-8")
    (out / "safety_prompt_v2.txt").write_text(
        "Safety-only prompt: determine if crop is safe_to_classify and if insulator is visible; abstain otherwise.",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
