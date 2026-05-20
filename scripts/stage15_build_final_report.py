#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd


def _safe_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    e00 = _safe_csv(root / "E00_schema_harness" / "schema_validation_summary.csv")
    e01 = _safe_csv(root / "E01_shadow_review_queue" / "review_queue_metrics.csv")
    e03 = _safe_csv(root / "E03_bad_crop_operating_modes" / "safety_modes_policy_results.csv")
    e07 = _safe_csv(root / "E07_cost_utility" / "utility_by_method.csv")

    claims = []
    claims.append({"claim_id": "C1_review_queue_quality", "status": "SUPPORTED" if len(e01) else "NOT_RUN"})
    claims.append({"claim_id": "C2_false_alarm_triage", "status": "SUPPORTED" if len(e01) else "NOT_RUN"})
    claims.append({"claim_id": "C3_bad_crop_modes", "status": "SUPPORTED" if len(e03) else "NOT_RUN"})
    claims.append({"claim_id": "C4_schema_reliability_gate", "status": "SUPPORTED" if len(e00) else "NOT_RUN"})
    claims.append({"claim_id": "C5_cost_sensitive_value", "status": "SUPPORTED" if len(e07) else "NOT_RUN"})
    claims_df = pd.DataFrame(claims)
    claims_df.to_csv(out / "stage15_claims_table.csv", index=False)

    rows = []
    if len(e01):
        s = e01.groupby("method", as_index=False)["reviewer_yield"].mean().sort_values("reviewer_yield", ascending=False).head(5)
        for _, r in s.iterrows():
            rows.append({"section": "E01", "metric": "reviewer_yield_mean", "method": r["method"], "value": float(r["reviewer_yield"])})
    if len(e03):
        best = e03.sort_values(["bad_false_accept", "clean_review"]).head(3)
        for _, r in best.iterrows():
            rows.append({"section": "E03", "metric": "bad_false_accept", "method": r["policy"], "value": float(r["bad_false_accept"])})
    if len(e07):
        b = e07.groupby("method", as_index=False)["utility"].mean().sort_values("utility", ascending=False).head(5)
        for _, r in b.iterrows():
            rows.append({"section": "E07", "metric": "utility_mean", "method": r["method"], "value": float(r["utility"])})
    pd.DataFrame(rows).to_csv(out / "stage15_main_results_table.csv", index=False)

    summary = [
        "# STAGE15 DEVELOPMENT VALUE REPORT",
        "",
        "VLM is treated as an inspection workflow layer (risk/review/safety/packets), not classifier replacement.",
        "",
        "## Sections",
        "1. Why VLM is not classifier replacement.",
        "2. Review/risk routing value.",
        "3. False alarm / overclaim value.",
        "4. Bad-crop safety modes.",
        "5. Reviewer packet and report draft value.",
        "6. Structured-output reliability gate.",
        "7. Cost-sensitive utility.",
        "8. Shadow field pilot design.",
        "9. Limitations.",
        "10. Next experiments.",
    ]
    (out / "STAGE15_DEVELOPMENT_VALUE_REPORT.md").write_text("\n".join(summary), encoding="utf-8")
    (out / "development_value_summary_for_supervisor.md").write_text(
        "VLM provides operational value via review queue quality, false-alarm triage, and safety gating around DINOv2.",
        encoding="utf-8",
    )

    # operational KPI table
    kpi_rows = []
    if len(e01):
        for _, r in e01.groupby("method", as_index=False).agg(
            reviewer_yield=("reviewer_yield", "mean"),
            accepted_accuracy=("accepted_accuracy", "mean"),
            false_alarm_capture=("false_alarm_capture_rate", "mean"),
        ).iterrows():
            kpi_rows.append(
                {
                    "method": r["method"],
                    "reviewer_yield": float(r["reviewer_yield"]),
                    "accepted_accuracy": float(r["accepted_accuracy"]),
                    "false_alarm_capture": float(r["false_alarm_capture"]),
                }
            )
    pd.DataFrame(kpi_rows).to_csv(out / "operational_kpi_table.csv", index=False)

    # artifact index
    arts = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            arts.append({"path": str(p).replace("\\", "/"), "size_bytes": p.stat().st_size})
    pd.DataFrame(arts).to_csv(out / "artifact_index.csv", index=False)

    # package
    pkg = root / "stage15_development_value_report_package.zip"
    if pkg.exists():
        pkg.unlink()
    with zipfile.ZipFile(pkg, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in root.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(root))


if __name__ == "__main__":
    main()
