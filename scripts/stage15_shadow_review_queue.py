#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage13-dir", required=True)
    ap.add_argument("--stage14-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(Path(args.stage14_dir) / "metrics_by_split.csv")
    b = m[(m["metric_group"] == "budget") & (m["target"] == "general_error")].copy()
    rank = m[(m["metric_group"] == "ranking")].copy()

    # reviewer yield proxy = error_capture/review_rate
    b["reviewer_yield"] = b["error_capture_rate"] / b["review_rate"].replace(0, pd.NA)

    review_queue_metrics = b.groupby(["method", "review_budget"], as_index=False).agg(
        review_rate=("review_rate", "mean"),
        accepted_accuracy=("accepted_accuracy", "mean"),
        error_capture_rate=("error_capture_rate", "mean"),
        false_alarm_capture_rate=("false_alarm_capture_rate", "mean"),
        dangerous_miss_capture_rate=("dangerous_miss_capture_rate", "mean"),
        reviewer_yield=("reviewer_yield", "mean"),
    )
    review_queue_metrics.to_csv(out / "review_queue_metrics.csv", index=False)
    review_queue_metrics.to_csv(out / "review_budget_curves.csv", index=False)
    review_queue_metrics[["method", "review_budget", "reviewer_yield"]].to_csv(out / "reviewer_yield_by_budget.csv", index=False)

    # top cases from per-record predictions
    pr = pd.read_csv(Path(args.stage14_dir) / "per_record_predictions_by_split.csv")
    dino = pr[(pr["method"] == "dino_only") & (pr["target"] == "general_error")]
    dvlm = pr[(pr["method"] == "dino_plus_vlm") & (pr["target"] == "general_error")]
    top = dvlm.sort_values("risk_score", ascending=False).head(20).copy()
    top.to_csv(out / "top_review_cases.csv", index=False)

    miss = dino.merge(dvlm, on=["split_id", "record_id", "label", "target", "subset"], suffixes=("_dino", "_dvlm"))
    miss = miss[(miss["risk_score_dvlm"] > miss["risk_score_dino"]) & (miss["label"] == 1)].copy()
    miss.head(200).to_csv(out / "dino_missed_vlm_caught_cases.csv", index=False)

    # figures
    plt.figure(figsize=(10, 5))
    for method in sorted(review_queue_metrics["method"].unique()):
        sub = review_queue_metrics[review_queue_metrics["method"] == method].sort_values("review_budget")
        plt.plot(sub["review_budget"], sub["reviewer_yield"], marker="o", label=method)
    plt.title("Reviewer yield by review budget")
    plt.xlabel("review budget")
    plt.ylabel("reviewer yield")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "fig_reviewer_yield.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    for method in sorted(review_queue_metrics["method"].unique()):
        sub = review_queue_metrics[review_queue_metrics["method"] == method].sort_values("review_budget")
        plt.plot(sub["review_budget"], sub["accepted_accuracy"], marker="o", label=method)
    plt.title("Accepted accuracy by review budget")
    plt.xlabel("review budget")
    plt.ylabel("accepted accuracy")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "fig_accepted_accuracy_budget.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    for method in sorted(review_queue_metrics["method"].unique()):
        sub = review_queue_metrics[review_queue_metrics["method"] == method].sort_values("review_budget")
        plt.plot(sub["review_budget"], sub["false_alarm_capture_rate"], marker="o", label=method)
    plt.title("False alarm capture by review budget")
    plt.xlabel("review budget")
    plt.ylabel("false alarm capture rate")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "fig_false_alarm_capture.png", dpi=150)
    plt.close()

    rep = [
        "# E01 Shadow Review Queue Report",
        "",
        f"- metrics rows: {len(review_queue_metrics)}",
        f"- ranking rows: {len(rank)}",
        "- Built from Stage14 repeated stratified splits.",
    ]
    (out / "E01_report.md").write_text("\n".join(rep), encoding="utf-8")


if __name__ == "__main__":
    main()
