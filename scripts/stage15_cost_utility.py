#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCENARIOS = [
    ("routine", 1.0, 0.2, 1.0, 2.0, 1.0, 0.8),
    ("safety_first", 1.0, 0.3, 1.2, 3.0, 1.2, 0.9),
    ("high_false_alarm_cost", 1.0, 0.2, 2.0, 2.0, 1.0, 0.8),
    ("high_dangerous_miss_cost", 1.0, 0.2, 1.0, 4.0, 1.0, 0.8),
    ("limited_review_budget", 1.0, 0.5, 1.0, 2.0, 1.0, 0.8),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage15-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    root = Path(args.stage15_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rq = pd.read_csv(root / "E01_shadow_review_queue" / "review_queue_metrics.csv")
    bc = pd.read_csv(root / "E03_bad_crop_operating_modes" / "safety_modes_policy_results.csv")

    rows = []
    for scen, corr_w, rev_c, fa_c, dm_c, bad_c, def_c in SCENARIOS:
        for _, r in rq.iterrows():
            method = r["method"]
            review = float(r["review_rate"])
            acc = float(r["accepted_accuracy"])
            fa_cap = float(r["false_alarm_capture_rate"])
            dm_cap = float(r["dangerous_miss_capture_rate"])
            utility = corr_w * acc - rev_c * review + fa_c * fa_cap + dm_c * dm_cap
            rows.append(
                {
                    "scenario": scen,
                    "method": method,
                    "review_budget": float(r["review_budget"]),
                    "utility": utility,
                }
            )
    util = pd.DataFrame(rows)
    util.to_csv(out / "utility_scenarios.csv", index=False)

    by_method = util.groupby(["scenario", "method"], as_index=False)["utility"].mean()
    by_method.to_csv(out / "utility_by_method.csv", index=False)

    sens = util.groupby("method", as_index=False).agg(mean_utility=("utility", "mean"), std_utility=("utility", "std"))
    sens.to_csv(out / "utility_sensitivity.csv", index=False)

    calc = by_method.copy()
    calc["utility_10000_items"] = calc["utility"] * 10000.0
    calc.to_csv(out / "illustrative_10000_items_calculator.csv", index=False)

    plt.figure(figsize=(11, 5))
    piv = by_method.pivot(index="scenario", columns="method", values="utility").fillna(0)
    piv.plot(kind="bar", ax=plt.gca())
    plt.title("Utility by scenario")
    plt.tight_layout()
    plt.savefig(out / "fig_utility_by_scenario.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sub = util[util["scenario"] == "limited_review_budget"].copy()
    for m in sorted(sub["method"].unique()):
        s = sub[sub["method"] == m].sort_values("review_budget")
        plt.plot(s["review_budget"], s["utility"], marker="o", label=m)
    plt.title("Review-cost sensitivity proxy")
    plt.xlabel("review budget")
    plt.ylabel("utility")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "fig_review_cost_sensitivity.png", dpi=150)
    plt.close()

    (out / "E07_report.md").write_text("# E07 Cost Utility\n\nIllustrative cost-sensitive utility scenarios.", encoding="utf-8")


if __name__ == "__main__":
    main()
