#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_clean_table() -> pd.DataFrame:
    # Real clean table from stage10 + stage12 risk features
    base = pd.read_csv("outputs/stage10/full_dataset_all_splits_dinov2_oof_plus_test/stage10_full_dataset_table.csv")
    rf = Path("outputs/stage12/risk_models/dev_risk_features.csv")
    if rf.exists():
        extra = pd.read_csv(rf)
        keep = [c for c in ["record_id", "vlm_needs_review", "vlm_has_quality_or_confounder", "dino_margin"] if c in extra.columns]
        if keep:
            base = base.merge(extra[keep].drop_duplicates("record_id"), on="record_id", how="left", suffixes=("", "_rf"))
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage12-dir", required=True)
    ap.add_argument("--stage13-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    bad = pd.read_csv(Path(args.stage12_dir) / "bad_crop_stress" / "stage12_bad_crop_predictions_with_flags.csv")
    clean = load_clean_table()

    if "is_bad_crop" not in bad.columns:
        bad["is_bad_crop"] = 1
    bad["is_bad_crop"] = bad["is_bad_crop"].astype(int)
    bad["dino_margin"] = bad.get("dino_margin", 1.0).fillna(1.0)
    bad["needs_review"] = bad.get("needs_review", True).fillna(True).astype(bool)
    bad["visibility"] = bad.get("visibility", "ambiguous").fillna("ambiguous").astype(str)
    bad["predicted_class"] = bad.get("predicted_class", bad.get("dino_top1", "")).fillna("").astype(str)

    clean["dino_margin"] = clean.get("dino_margin", 1.0).fillna(1.0)
    clean["needs_review"] = clean.get("vlm_needs_review", clean.get("label_needs_review", 0)).fillna(0).astype(int).astype(bool)
    clean["visibility"] = clean.get("label_visibility", "clear").fillna("clear").astype(str)
    clean["predicted_class"] = clean.get("dino_top1", "").fillna("").astype(str)
    clean["is_bad_crop"] = 0
    clean["record_id"] = clean["record_id"].astype(str)

    def review_flags(df: pd.DataFrame) -> pd.DataFrame:
        out_df = df.copy()
        out_df["strict_review"] = 1
        out_df["balanced_review"] = out_df["needs_review"].astype(bool).astype(int)
        out_df["lenient_review"] = out_df["visibility"].isin(["poor", "ambiguous"]).astype(int)
        out_df["score_aware_review"] = (out_df["needs_review"] | (out_df["dino_margin"] < 0.25)).astype(int)
        out_df["class_aware_review"] = (
            out_df["needs_review"] | out_df["predicted_class"].isin(["defect_flashover", "defect_broken"])
        ).astype(int)
        out_df["hybrid_review"] = (out_df["score_aware_review"].astype(bool) | out_df["class_aware_review"].astype(bool)).astype(int)
        out_df["closed_set_review"] = 0
        return out_df

    bad = review_flags(bad)
    clean = review_flags(clean)

    policies = [
        "closed_set_review",
        "strict_review",
        "balanced_review",
        "lenient_review",
        "score_aware_review",
        "class_aware_review",
        "hybrid_review",
    ]

    rows = []
    for p in policies:
        rv_bad = bad[p].astype(int)
        rv_clean = clean[p].astype(int)
        rows.append(
            {
                "policy": p.replace("_review", ""),
                "bad_false_accept": float((rv_bad == 0).sum() / max(1, len(rv_bad))),
                "bad_safe_reject": float((rv_bad == 1).sum() / max(1, len(rv_bad))),
                "clean_review": float((rv_clean == 1).sum() / max(1, len(rv_clean))),
                "clean_accept": float((rv_clean == 0).sum() / max(1, len(rv_clean))),
                "clean_accepted_accuracy": float(
                    (
                        (rv_clean == 0)
                        & (clean["dino_top1"].astype(str) == clean["label_coarse_class"].astype(str))
                    ).sum()
                    / max(1, (rv_clean == 0).sum())
                ),
                "review_rate_overall": float((rv_bad.sum() + rv_clean.sum()) / max(1, len(rv_bad) + len(rv_clean))),
            }
        )
    pol = pd.DataFrame(rows).sort_values(["bad_false_accept", "clean_review"])
    pol.to_csv(out / "safety_modes_policy_results.csv", index=False)
    pol.to_csv(out / "safety_pareto_frontier.csv", index=False)

    by_cor = bad.groupby("corruption", dropna=False).agg(n=("record_id", "count"), review_rate=("balanced_review", "mean")).reset_index()
    by_cor.to_csv(out / "bad_crop_by_corruption.csv", index=False)

    by_cls = clean.groupby("label_coarse_class").agg(n=("record_id", "count"), clean_review=("balanced_review", "mean")).reset_index()
    by_cls.to_csv(out / "clean_crop_by_class.csv", index=False)

    rec = pol.copy()
    rec["recommended_use_case"] = rec["policy"].map(
        {
            "strict": "safety_first",
            "balanced": "routine",
            "lenient": "high_throughput",
            "hybrid": "mixed",
        }
    ).fillna("custom")
    rec.to_csv(out / "policy_recommendations.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.scatter(pol["clean_review"], pol["bad_false_accept"])
    for _, r in pol.iterrows():
        plt.text(r["clean_review"], r["bad_false_accept"], r["policy"], fontsize=8)
    plt.xlabel("clean_review")
    plt.ylabel("bad_false_accept")
    plt.title("Bad false accept vs clean review")
    plt.tight_layout()
    plt.savefig(out / "fig_bad_false_accept_vs_clean_review.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(pol["policy"], pol["review_rate_overall"])
    plt.xticks(rotation=40, ha="right")
    plt.ylabel("review_rate_overall")
    plt.title("Mode review-rate tradeoff")
    plt.tight_layout()
    plt.savefig(out / "fig_modes_tradeoff.png", dpi=150)
    plt.close()

    (out / "E03_report.md").write_text(
        "# E03 Bad Crop Operating Modes (clean-vs-bad)\n\nComputed with real clean crop table + bad crop stress table.",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
