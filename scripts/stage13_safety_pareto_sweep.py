#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage12-dir", default="outputs/stage12")
    ap.add_argument("--out-dir", default="outputs/stage13_tradeoff_benefit_expansion/E01_safety_pareto")
    args = ap.parse_args()

    stage12 = Path(args.stage12_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    src = stage12 / "bad_crop_safety_pareto"
    pareto = pd.read_csv(src / "bad_crop_safety_pareto.csv")
    by_corruption = pd.read_csv(src / "bad_crop_by_corruption.csv")

    pareto["policy_kind"] = pareto["policy"].map(
        lambda x: "strict" if "strict" in x else ("balanced" if "balanced" in x else ("closed_set" if "closed_set" in x else "other"))
    )
    pareto["dominates_baseline"] = (
        (pareto["bad_false_accept_rate"] <= float(pareto.loc[pareto["policy"] == "current_vlm_policy", "bad_false_accept_rate"].iloc[0]))
        & (pareto["clean_review_rate"] <= float(pareto.loc[pareto["policy"] == "current_vlm_policy", "clean_review_rate"].iloc[0]))
    )

    # Frontier: sort by bad false accept then keep strictly improving clean review
    frontier_rows = []
    best_clean_review = float("inf")
    for _, r in pareto.sort_values("bad_false_accept_rate").iterrows():
        if float(r["clean_review_rate"]) < best_clean_review:
            frontier_rows.append(r.to_dict())
            best_clean_review = float(r["clean_review_rate"])
    frontier = pd.DataFrame(frontier_rows)

    pareto.to_csv(out / "policy_sweep_bad_vs_clean.csv", index=False)
    frontier.to_csv(out / "pareto_frontier.csv", index=False)
    by_corruption.to_csv(out / "bad_crop_by_corruption.csv", index=False)

    clean_cls = pd.read_csv(stage12 / "risk_models" / "dev_risk_features.csv")
    clean_cls = clean_cls[["record_id", "split", "label_coarse_class", "dino_top1"]].copy()
    clean_cls["is_clean_accept_proxy"] = (clean_cls["dino_top1"] == clean_cls["label_coarse_class"]).astype(int)
    clean_cls.groupby("label_coarse_class", as_index=False)["is_clean_accept_proxy"].mean().rename(
        columns={"is_clean_accept_proxy": "proxy_accept_rate"}
    ).to_csv(out / "clean_crop_by_class.csv", index=False)

    fail = pd.read_csv(src / "bad_crop_failure_cases.csv")
    fail.to_csv(out / "failure_cases_bad_accepted.csv", index=False)
    false_review = pd.read_csv(src / "clean_false_review_cases.csv")
    false_review.to_csv(out / "false_review_clean_cases.csv", index=False)

    fig = plt.figure(figsize=(7, 5))
    plt.scatter(pareto["clean_review_rate"], pareto["bad_false_accept_rate"])
    for _, r in pareto.iterrows():
        plt.annotate(str(r["policy"]), (r["clean_review_rate"], r["bad_false_accept_rate"]), fontsize=7)
    plt.xlabel("clean_review_rate (lower better)")
    plt.ylabel("bad_false_accept_rate (lower better)")
    plt.title("E01 Safety Pareto")
    plt.tight_layout()
    fig.savefig(out / "fig_safety_pareto.png", dpi=160)
    plt.close(fig)

    success = pareto[(pareto["bad_false_accept_rate"] <= 0.10) & (pareto["clean_review_rate"] <= 0.40)]
    report = [
        "# E01 Safety Pareto Report",
        "",
        f"- Policies evaluated: {len(pareto)}",
        f"- Frontier points: {len(frontier)}",
        f"- Success candidates (bad_false_accept<=0.10 and clean_review<=0.40): {len(success)}",
    ]
    if len(success):
        report.append("- Best success policy:")
        best = success.sort_values(["bad_false_accept_rate", "clean_review_rate"]).iloc[0].to_dict()
        report.append(f"  - {best['policy']} | bad_false_accept={best['bad_false_accept_rate']:.4f} | clean_review={best['clean_review_rate']:.4f}")
    else:
        report.append("- No policy meets target simultaneously; keep strict/balanced as operating modes.")
    (out / "E01_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
