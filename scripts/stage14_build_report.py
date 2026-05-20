#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    agg = pd.read_csv(root / "aggregate_metrics.csv")
    aup = pd.read_csv(root / "auprc_minus_prevalence.csv")
    acc = pd.read_csv(root / "accepted_accuracy_by_review_budget.csv")
    wins = pd.read_csv(root / "method_wins_by_split.csv") if (root / "method_wins_by_split.csv").exists() else pd.DataFrame()

    # Core claim for general_error
    ge = aup[(aup["target"] == "general_error")]
    dino = ge[(ge["method"] == "dino_only") & (ge["metric"] == "auprc_minus_prevalence")]
    dpv = ge[(ge["method"] == "dino_plus_vlm") & (ge["metric"] == "auprc_minus_prevalence")]
    claim = "split-sensitive"
    rationale = "insufficient data"
    if len(dino) and len(dpv):
        delta = float(dpv["mean"].iloc[0] - dino["mean"].iloc[0])
        if delta > 0.02:
            claim = "robust_positive"
            rationale = f"dino_plus_vlm mean(AUPRC-prevalence) is higher than dino_only by {delta:.4f}"
        elif delta > 0:
            claim = "mixed_promising"
            rationale = f"dino_plus_vlm slightly higher than dino_only by {delta:.4f}"
        else:
            claim = "historical_split_dependent"
            rationale = f"dino_plus_vlm not above dino_only (delta {delta:.4f})"

    lines = [
        "# Stage14 Stratified Resplit Robustness Report",
        "",
        "## Summary",
        f"- Final claim class: **{claim}**",
        f"- Rationale: {rationale}",
        "",
        "## Required AUPRC Context",
        "- For each target/method, AUPRC is reported together with prevalence and AUPRC-prevalence.",
        "",
        "## Main Files",
        "- aggregate_metrics.csv",
        "- auprc_minus_prevalence.csv",
        "- accepted_accuracy_by_review_budget.csv",
        "- capture_rates_by_review_budget.csv",
        "- method_wins_by_split.csv",
    ]
    (out / "robustness_report.md").write_text("\n".join(lines), encoding="utf-8")

    rec_lines = [
        "# Final Claim Recommendation",
        "",
        f"recommended_claim = {claim}",
        f"reason = {rationale}",
        "",
        "Use this claim wording in the paper/report according to Stage14 interpretation rules.",
    ]
    (out / "final_claim_recommendation.md").write_text("\n".join(rec_lines), encoding="utf-8")

    artifacts = []
    for p in sorted(root.glob("*")):
        if p.is_file():
            artifacts.append({"path": str(p).replace("\\", "/"), "size_bytes": p.stat().st_size})
    pd.DataFrame(artifacts).to_csv(out / "artifact_index.csv", index=False)


if __name__ == "__main__":
    main()
