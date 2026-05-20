#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def utility(row: pd.Series, cfg: dict) -> float:
    return (
        cfg["reward_correct_auto"] * row.get("correct_auto", 0.0)
        - cfg["cost_review"] * row.get("reviewed", 0.0)
        - cfg["cost_false_alarm"] * row.get("false_alarm", 0.0)
        - cfg["cost_dangerous_miss"] * row.get("dangerous_miss", 0.0)
        - cfg["cost_defect_confusion"] * row.get("defect_confusion", 0.0)
        - cfg["cost_bad_crop_accept"] * row.get("bad_crop_accepted", 0.0)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage12-dir", default="outputs/stage12")
    ap.add_argument("--out-dir", default="outputs/stage13_tradeoff_benefit_expansion/E06_cost_sensitive_utility")
    args = ap.parse_args()

    s12 = Path(args.stage12_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    acc = pd.read_csv(s12 / "risk_review_final_test_v2" / "accepted_accuracy_at_review_rates_test.csv")
    ec = pd.read_csv(s12 / "risk_review_final_test_v2" / "error_capture_at_review_rates_test.csv")
    fa = pd.read_csv(s12 / "risk_review_final_test_v2" / "false_alarm_capture_at_review_rates_test.csv")
    dm = pd.read_csv(s12 / "risk_review_final_test_v2" / "dangerous_miss_capture_at_review_rates_test.csv")
    bad = pd.read_csv(s12 / "bad_crop_safety_pareto" / "bad_crop_safety_pareto.csv")

    df = acc.merge(ec, on=["method", "review_rate"], how="left").merge(
        fa, on=["method", "review_rate"], how="left"
    ).merge(dm, on=["method", "review_rate"], how="left")
    df["correct_auto"] = df["accepted_accuracy"] * df["coverage"]
    df["reviewed"] = df["review_rate"]
    df["false_alarm"] = 1 - df["false_alarm_capture_rate"].fillna(0.0)
    df["dangerous_miss"] = 1 - df["dangerous_miss_capture_rate"].fillna(0.0)
    df["defect_confusion"] = 1 - df["error_capture_rate"].fillna(0.0)
    df["bad_crop_accepted"] = float(bad.loc[bad["policy"] == "current_vlm_policy", "bad_false_accept_rate"].iloc[0])

    scenarios = {
        "low_review_cost": dict(reward_correct_auto=1.0, cost_review=0.2, cost_false_alarm=1.0, cost_dangerous_miss=2.0, cost_defect_confusion=0.8, cost_bad_crop_accept=1.0),
        "high_review_cost": dict(reward_correct_auto=1.0, cost_review=0.8, cost_false_alarm=1.0, cost_dangerous_miss=2.0, cost_defect_confusion=0.8, cost_bad_crop_accept=1.0),
        "high_false_alarm_cost": dict(reward_correct_auto=1.0, cost_review=0.3, cost_false_alarm=2.5, cost_dangerous_miss=2.0, cost_defect_confusion=0.8, cost_bad_crop_accept=1.0),
        "high_dangerous_miss_cost": dict(reward_correct_auto=1.0, cost_review=0.3, cost_false_alarm=1.0, cost_dangerous_miss=4.0, cost_defect_confusion=0.8, cost_bad_crop_accept=1.0),
        "high_bad_crop_cost": dict(reward_correct_auto=1.0, cost_review=0.3, cost_false_alarm=1.0, cost_dangerous_miss=2.0, cost_defect_confusion=0.8, cost_bad_crop_accept=3.0),
        "balanced_industrial": dict(reward_correct_auto=1.0, cost_review=0.4, cost_false_alarm=1.5, cost_dangerous_miss=3.0, cost_defect_confusion=1.0, cost_bad_crop_accept=2.0),
    }

    rows = []
    for sc, cfg in scenarios.items():
        tmp = df.copy()
        tmp["scenario"] = sc
        tmp["utility"] = tmp.apply(lambda r: utility(r, cfg), axis=1)
        rows.append(tmp)
    allu = pd.concat(rows, ignore_index=True)
    allu.to_csv(out / "utility_scenarios.csv", index=False)

    best = (
        allu.sort_values("utility", ascending=False)
        .groupby("scenario", as_index=False)
        .first()[["scenario", "method", "review_rate", "utility"]]
    )
    best.to_csv(out / "utility_by_method.csv", index=False)

    breakdown = allu.groupby(["scenario", "method"], as_index=False)[
        ["correct_auto", "reviewed", "false_alarm", "dangerous_miss", "defect_confusion", "bad_crop_accepted", "utility"]
    ].mean()
    breakdown.to_csv(out / "utility_error_breakdown.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 5))
        for sc, g in best.groupby("scenario"):
            plt.scatter([sc], [float(g["utility"].iloc[0])])
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("best utility")
        plt.title("E06 Utility by Scenario")
        plt.tight_layout()
        fig.savefig(out / "fig_utility_by_scenario.png", dpi=160)
        plt.close(fig)
    except Exception:
        pass

    (out / "E06_report.md").write_text(
        "# E06 Cost-Sensitive Utility\n\nComputed utility across six cost scenarios using Stage12 risk/safety metrics.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
