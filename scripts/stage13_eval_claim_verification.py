#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def expected_support(label: str, claim_id: str) -> str:
    if claim_id == "flashover_surface_evidence":
        return "supported" if label == "defect_flashover" else "contradicted"
    if claim_id == "broken_structure_evidence":
        return "supported" if label == "defect_broken" else "contradicted"
    if claim_id == "ok_intact_evidence":
        return "supported" if label == "insulator_ok" else "contradicted"
    return "not_enough_evidence"


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
    df = m.merge(o, on=["record_id", "claim_id"], how="left")
    if "verification" in df.columns:
        df["verification"] = df["verification"].fillna("not_enough_evidence").astype(str)
    else:
        df["verification"] = "not_enough_evidence"
    df["expected"] = df.apply(lambda r: expected_support(str(r["label_coarse_class"]), str(r["claim_id"])), axis=1)
    df["correct"] = df["verification"] == df["expected"]
    df["is_ok_control"] = df["label_coarse_class"] == "insulator_ok"
    df["false_supported_defect_on_ok"] = (
        df["is_ok_control"] & df["claim_id"].isin(["flashover_surface_evidence", "broken_structure_evidence"]) & (df["verification"] == "supported")
    )
    metrics = pd.DataFrame(
        [
            {
                "n_claims": len(df),
                "parse_ok_rate": float(df["parse_ok"].fillna(False).astype(bool).mean()) if "parse_ok" in df.columns else 0.0,
                "claim_verification_accuracy": float(df["correct"].mean()),
                "false_supported_defect_on_ok_rate": float(df["false_supported_defect_on_ok"].mean()),
                "review_rate": float(df["needs_review"].fillna(True).astype(bool).mean()) if "needs_review" in df.columns else 1.0,
            }
        ]
    )
    metrics.to_csv(out / "claim_verification_metrics.csv", index=False)
    df[df["false_supported_defect_on_ok"]].to_csv(out / "ok_false_supported_defect_cases.csv", index=False)
    with (out / "claim_verification_outputs.jsonl").open("w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            f.write(r.to_json(force_ascii=False) + "\n")
    (out / "claim_verification_report.md").write_text(
        "# E05 Claim Verification\n\nReal VLM claim verification run completed.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
