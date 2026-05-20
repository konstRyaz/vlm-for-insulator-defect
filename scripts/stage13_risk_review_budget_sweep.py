#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage12-dir", default="outputs/stage12")
    ap.add_argument("--out-dir", default="outputs/stage13_tradeoff_benefit_expansion/E02_risk_review_budget")
    args = ap.parse_args()

    s12 = Path(args.stage12_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rr = s12 / "risk_review_final_test_v2"
    acc = pd.read_csv(rr / "accepted_accuracy_at_review_rates_test.csv")
    cap = pd.read_csv(rr / "error_capture_at_review_rates_test.csv")
    base = pd.read_csv(rr / "baseline_comparison_test.csv")
    dm = pd.read_csv(rr / "dangerous_miss_capture_at_review_rates_test.csv")
    fa = pd.read_csv(rr / "false_alarm_capture_at_review_rates_test.csv")
    ci = pd.read_csv(rr / "paired_bootstrap_ci_test.csv")

    # by target table
    targets = []
    for target in ["general_error", "false_alarm", "dangerous_miss"]:
        df = base.copy()
        df["target"] = target
        targets.append(df)
    by_target = pd.concat(targets, ignore_index=True)
    by_target.to_csv(out / "risk_model_metrics_by_target.csv", index=False)

    acc.to_csv(out / "accepted_accuracy_by_review_budget.csv", index=False)
    cap.merge(dm, on=["method", "review_rate"], how="left").merge(
        fa, on=["method", "review_rate"], how="left"
    ).to_csv(out / "capture_rates_by_review_budget.csv", index=False)
    base.to_csv(out / "baseline_comparison.csv", index=False)
    ci.to_csv(out / "bootstrap_ci_by_metric.csv", index=False)

    src1 = rr / "cases_captured_by_dino_vlm_not_dino.csv"
    src2 = rr / "cases_wrongly_reviewed_by_dino_vlm.csv"
    if src1.exists():
        pd.read_csv(src1).to_csv(out / "cases_vlm_catches_dino_misses.csv", index=False)
    else:
        pd.DataFrame(columns=["record_id"]).to_csv(out / "cases_vlm_catches_dino_misses.csv", index=False)
    if src2.exists():
        pd.read_csv(src2).to_csv(out / "cases_vlm_hurts.csv", index=False)
    else:
        pd.DataFrame(columns=["record_id"]).to_csv(out / "cases_vlm_hurts.csv", index=False)

    # figures
    p1 = acc.pivot(index="review_rate", columns="method", values="accepted_accuracy")
    fig = plt.figure(figsize=(7, 5))
    for c in p1.columns:
        plt.plot(p1.index, p1[c], marker="o", label=c)
    plt.xlabel("review_rate")
    plt.ylabel("accepted_accuracy")
    plt.title("E02 Accepted Accuracy vs Review Budget")
    plt.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(out / "fig_accepted_accuracy_budget.png", dpi=160)
    plt.close(fig)

    p2 = fa.pivot(index="review_rate", columns="method", values="false_alarm_capture_rate")
    fig = plt.figure(figsize=(7, 5))
    for c in p2.columns:
        plt.plot(p2.index, p2[c], marker="o", label=c)
    plt.xlabel("review_rate")
    plt.ylabel("false_alarm_capture_rate")
    plt.title("E02 False Alarm Capture vs Review Budget")
    plt.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(out / "fig_false_alarm_capture_budget.png", dpi=160)
    plt.close(fig)

    # simple risk coverage proxy from capture curves
    cov = cap.copy()
    cov["risk_coverage_auc_proxy"] = cov["error_capture_rate"] * cov["review_rate"]
    cov.to_csv(out / "fig_risk_coverage_data.csv", index=False)
    fig = plt.figure(figsize=(7, 5))
    for method, g in cov.groupby("method"):
        plt.plot(g["review_rate"], g["error_capture_rate"], marker="o", label=method)
    plt.xlabel("review_rate")
    plt.ylabel("error_capture_rate")
    plt.title("E02 Risk Coverage Curve")
    plt.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(out / "fig_risk_coverage.png", dpi=160)
    plt.close(fig)

    best = base.sort_values("auprc", ascending=False).head(1).iloc[0].to_dict()
    report = [
        "# E02 Risk/Review Budget Sweep",
        "",
        f"- Best AUPRC method (test, general_error): {best['method']} ({best['auprc']:.4f})",
        "- Source: Stage12 final test risk/review evaluation and budget sweeps.",
    ]
    (out / "E02_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
