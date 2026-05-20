#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def agg_stats(df: pd.DataFrame, metric_cols: list[str], group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, part in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        base = {k: v for k, v in zip(group_cols, key)}
        for mc in metric_cols:
            vals = part[mc].dropna()
            if len(vals) == 0:
                continue
            rows.append(
                {
                    **base,
                    "metric": mc,
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=0)),
                    "median": float(vals.median()),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "p2_5": float(vals.quantile(0.025)),
                    "p97_5": float(vals.quantile(0.975)),
                    "n_splits": int(vals.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(args.metrics)

    rank = m[m["metric_group"] == "ranking"].copy()
    budget = m[m["metric_group"] == "budget"].copy()

    rank_agg = agg_stats(
        rank,
        ["auroc", "auprc", "auprc_minus_prevalence", "positive_prevalence"],
        ["target", "method"],
    )
    budget_agg = agg_stats(
        budget,
        [
            "review_rate",
            "accepted_accuracy",
            "error_capture_rate",
            "false_alarm_capture_rate",
            "ok_to_flashover_capture_rate",
            "dangerous_miss_capture_rate",
            "correct_rejection_rate",
            "unnecessary_review_rate",
        ],
        ["target", "method", "review_budget"],
    )

    rank_agg.to_csv(out / "aggregate_metrics.csv", index=False)
    rank_agg[rank_agg["metric"] == "auprc_minus_prevalence"].to_csv(out / "auprc_minus_prevalence.csv", index=False)
    budget_agg[budget_agg["metric"] == "accepted_accuracy"].to_csv(out / "accepted_accuracy_by_review_budget.csv", index=False)
    budget_agg[
        budget_agg["metric"].isin(["false_alarm_capture_rate", "ok_to_flashover_capture_rate", "dangerous_miss_capture_rate", "error_capture_rate"])
    ].to_csv(out / "capture_rates_by_review_budget.csv", index=False)

    prev = rank[["split_id", "target", "method", "positive_prevalence"]].drop_duplicates()
    prev = prev.groupby(["split_id", "target"], as_index=False)["positive_prevalence"].mean()
    prev.to_csv(out / "target_prevalence_by_split.csv", index=False)

    # wins by split on auprc_minus_prevalence
    wins = []
    cmp = rank[["split_id", "target", "method", "auprc_minus_prevalence"]].copy()
    for (sid, tgt), part in cmp.groupby(["split_id", "target"]):
        part = part.dropna(subset=["auprc_minus_prevalence"])
        if len(part) == 0:
            continue
        best = part.sort_values("auprc_minus_prevalence", ascending=False).iloc[0]
        wins.append({"split_id": sid, "target": tgt, "winner_method": best["method"], "winner_value": float(best["auprc_minus_prevalence"])})
    wins_df = pd.DataFrame(wins)
    wins_df.to_csv(out / "method_wins_by_split.csv", index=False)

    # figures
    fig_dir = out
    # AUPRC-prevalence figure
    f = rank_agg[rank_agg["metric"] == "auprc_minus_prevalence"]
    if len(f):
        plt.figure(figsize=(12, 5))
        for tgt in sorted(f["target"].unique()):
            sub = f[f["target"] == tgt].sort_values("mean", ascending=False).head(8)
            plt.plot(sub["method"], sub["mean"], marker="o", label=tgt)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("mean AUPRC - prevalence")
        plt.title("Stage14: AUPRC minus prevalence")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_auprc_minus_prevalence.png", dpi=150)
        plt.close()

    f2 = budget_agg[budget_agg["metric"] == "accepted_accuracy"]
    if len(f2):
        plt.figure(figsize=(10, 5))
        for method in sorted(f2["method"].unique()):
            sub = f2[(f2["method"] == method) & (f2["target"] == "general_error")].sort_values("review_budget")
            if len(sub):
                plt.plot(sub["review_budget"], sub["mean"], marker="o", label=method)
        plt.xlabel("review budget")
        plt.ylabel("accepted accuracy (mean)")
        plt.title("Accepted accuracy by review budget (general_error)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_accepted_accuracy_by_review_budget.png", dpi=150)
        plt.close()

    f3 = budget_agg[budget_agg["metric"] == "false_alarm_capture_rate"]
    if len(f3):
        plt.figure(figsize=(10, 5))
        for method in sorted(f3["method"].unique()):
            sub = f3[(f3["method"] == method) & (f3["target"] == "false_alarm")].sort_values("review_budget")
            if len(sub):
                plt.plot(sub["review_budget"], sub["mean"], marker="o", label=method)
        plt.xlabel("review budget")
        plt.ylabel("false alarm capture rate (mean)")
        plt.title("False-alarm capture by review budget")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_false_alarm_capture_by_review_budget.png", dpi=150)
        plt.close()

    if len(wins_df):
        wcnt = wins_df["winner_method"].value_counts().reset_index()
        wcnt.columns = ["method", "wins"]
        plt.figure(figsize=(9, 4))
        plt.bar(wcnt["method"], wcnt["wins"])
        plt.xticks(rotation=45, ha="right")
        plt.title("Method wins by split/target (AUPRC-prevalence)")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_method_wins.png", dpi=150)
        plt.close()

    if len(prev):
        plt.figure(figsize=(10, 4))
        for tgt in sorted(prev["target"].unique()):
            sub = prev[prev["target"] == tgt]
            plt.plot(sub["split_id"], sub["positive_prevalence"], marker=".", label=tgt)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("positive prevalence")
        plt.title("Target prevalence by split")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_target_prevalence_by_split.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
