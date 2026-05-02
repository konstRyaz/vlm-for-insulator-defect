#!/usr/bin/env python3
"""Compare Stage 4 tight-crop and context-crop eval runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tight-eval-dir", required=True, help="Stage 4 tight-crop eval directory.")
    parser.add_argument("--context-eval-dir", required=True, help="Stage 4 context-crop eval directory.")
    parser.add_argument("--context-label", default="context_pad030", help="Label for the context run.")
    parser.add_argument("--out-dir", required=True, help="Output report directory.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_run(eval_dir: Path, label: str) -> tuple[dict[str, Any], pd.DataFrame]:
    metrics = read_json(eval_dir / "stage4_metrics.json")
    cases = pd.read_csv(eval_dir / "stage4_case_table.csv")
    for col in ["detector_found", "is_good_crop", "vlm_correct_on_good_crop", "ceiling_correct"]:
        if col in cases.columns:
            cases[col] = bool_series(cases[col])
    cases["run_label"] = label
    return metrics, cases


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_bar_chart(path: Path, title: str, labels: list[str], values: list[float], ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    colors = ["#344E41", "#A3B18A", "#DDA15E", "#BC6C25"][: len(labels)]
    bars = ax.bar(labels, values, color=colors)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values + [0.5]) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_pattern_chart(path: Path, df: pd.DataFrame, title: str) -> None:
    plot_df = df.copy()
    plot_df["pattern"] = plot_df["gt_coarse_class"].fillna("") + " -> " + plot_df["pred_vlm_coarse_class"].fillna("")
    plot_df = plot_df.sort_values("tight_count", ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=160)
    y = range(len(plot_df))
    ax.barh([i - 0.18 for i in y], plot_df["tight_count"], height=0.34, label="tight 0.15", color="#6B705C")
    ax.barh([i + 0.18 for i in y], plot_df["context_count"], height=0.34, label="context", color="#CB997E")
    ax.set_yticks(list(y), plot_df["pattern"])
    ax.set_xlabel("Error count")
    ax.set_title(title, fontsize=14, weight="bold")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        else:
            display[col] = display[col].fillna("").astype(str)
    headers = [str(col) for col in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def summarize_patterns(cases: pd.DataFrame) -> pd.DataFrame:
    good = cases[cases["is_good_crop"] == True].copy()
    errors = good[good["vlm_correct_on_good_crop"] != True].copy()
    out = (
        errors.groupby(["gt_coarse_class", "pred_vlm_coarse_class"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    return out


def per_class_accuracy(cases: pd.DataFrame, label: str) -> pd.DataFrame:
    out = (
        cases.groupby("gt_coarse_class")
        .agg(n=("record_id", "count"), correct=("vlm_correct_on_good_crop", "sum"))
        .reset_index()
    )
    out[f"{label}_acc"] = out["correct"] / out["n"]
    out = out.rename(columns={"correct": f"{label}_correct"})
    return out[["gt_coarse_class", "n", f"{label}_correct", f"{label}_acc"]]


def main() -> None:
    args = parse_args()
    tight_dir = Path(args.tight_eval_dir)
    context_dir = Path(args.context_eval_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    tight_metrics, tight_cases = load_run(tight_dir, "tight_015")
    context_metrics, context_cases = load_run(context_dir, args.context_label)

    joined = tight_cases.merge(
        context_cases,
        on=["record_id", "gt_coarse_class", "gt_visibility"],
        suffixes=("_tight", "_context"),
    )
    joined["tight_correct"] = joined["vlm_correct_on_good_crop_tight"] == True
    joined["context_correct"] = joined["vlm_correct_on_good_crop_context"] == True
    joined["change_bucket"] = "unchanged"
    joined.loc[(joined["tight_correct"] == False) & (joined["context_correct"] == True), "change_bucket"] = "helped"
    joined.loc[(joined["tight_correct"] == True) & (joined["context_correct"] == False), "change_bucket"] = "hurt"
    joined.loc[(joined["tight_correct"] == True) & (joined["context_correct"] == True), "change_bucket"] = "both_correct"
    joined.loc[(joined["tight_correct"] == False) & (joined["context_correct"] == False), "change_bucket"] = "both_wrong"

    change_cols = [
        "record_id",
        "gt_coarse_class",
        "gt_visibility",
        "change_bucket",
        "matched_pred_record_id_tight",
        "pred_vlm_coarse_class_tight",
        "pred_vlm_visibility_tight",
        "matched_pred_record_id_context",
        "pred_vlm_coarse_class_context",
        "pred_vlm_visibility_context",
        "ceiling_coarse_class_tight",
        "ceiling_correct_tight",
    ]
    joined[change_cols].sort_values(["change_bucket", "gt_coarse_class", "record_id"]).to_csv(
        out_dir / "helped_hurt_cases.csv", index=False
    )

    metrics_rows = []
    for label, metrics in [("tight_015", tight_metrics), (args.context_label, context_metrics)]:
        row = {"run": label, **metrics["counts"], **metrics["rates"]}
        metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "metrics_comparison.csv", index=False)

    tight_patterns = summarize_patterns(tight_cases).rename(columns={"count": "tight_count"})
    context_patterns = summarize_patterns(context_cases).rename(columns={"count": "context_count"})
    pattern_cmp = tight_patterns.merge(
        context_patterns,
        on=["gt_coarse_class", "pred_vlm_coarse_class"],
        how="outer",
    ).fillna(0)
    pattern_cmp["tight_count"] = pattern_cmp["tight_count"].astype(int)
    pattern_cmp["context_count"] = pattern_cmp["context_count"].astype(int)
    pattern_cmp["delta_context_minus_tight"] = pattern_cmp["context_count"] - pattern_cmp["tight_count"]
    pattern_cmp = pattern_cmp.sort_values(["tight_count", "context_count"], ascending=False)
    pattern_cmp.to_csv(out_dir / "error_pattern_comparison.csv", index=False)

    class_cmp = per_class_accuracy(tight_cases, "tight").merge(
        per_class_accuracy(context_cases, "context"),
        on=["gt_coarse_class", "n"],
        how="outer",
    )
    class_cmp["delta_context_minus_tight"] = class_cmp["context_acc"] - class_cmp["tight_acc"]
    class_cmp.to_csv(out_dir / "per_class_accuracy_comparison.csv", index=False)

    bucket_counts = joined["change_bucket"].value_counts().reindex(
        ["helped", "hurt", "both_correct", "both_wrong"], fill_value=0
    )
    bucket_counts.rename_axis("change_bucket").reset_index(name="count").to_csv(
        out_dir / "change_bucket_counts.csv", index=False
    )

    save_bar_chart(
        out_dir / "pipeline_rate_comparison.png",
        "Stage 4 pipeline correctness",
        metrics_df["run"].tolist(),
        metrics_df["pipeline_correct_rate"].tolist(),
        "Pipeline correct rate",
    )
    save_bar_chart(
        out_dir / "ceiling_gap_comparison.png",
        "Gap to Stage 3 ceiling",
        metrics_df["run"].tolist(),
        metrics_df["ceiling_vs_actual_gap"].tolist(),
        "Ceiling minus actual",
    )
    save_bar_chart(
        out_dir / "helped_hurt_counts.png",
        "Object-level change from tight crop to context crop",
        bucket_counts.index.tolist(),
        [float(v) for v in bucket_counts.tolist()],
        "GT objects",
    )
    save_pattern_chart(
        out_dir / "error_pattern_comparison.png",
        pattern_cmp.head(10),
        "Top coarse error patterns",
    )

    tight_rate = float(tight_metrics["rates"]["pipeline_correct_rate"])
    context_rate = float(context_metrics["rates"]["pipeline_correct_rate"])
    tight_hits = int(tight_metrics["counts"]["pipeline_correct_total"])
    context_hits = int(context_metrics["counts"]["pipeline_correct_total"])
    ceiling = float(context_metrics["rates"]["ceiling_correct_rate"])
    helped = int(bucket_counts.get("helped", 0))
    hurt = int(bucket_counts.get("hurt", 0))

    lines = [
        "# Stage 4 Context Comparison",
        "",
        f"Compared runs: `tight_015` vs `{args.context_label}`.",
        "",
        "## Headline",
        "",
        f"- Tight crop pipeline correctness: `{tight_hits}/58 = {tight_rate:.4f}`.",
        f"- Context crop pipeline correctness: `{context_hits}/58 = {context_rate:.4f}`.",
        f"- Clean Stage 3 ceiling: `{ceiling:.4f}`.",
        f"- Object-level changes: `{helped}` helped, `{hurt}` hurt.",
        "",
        "The context crop improves total Stage 4 correctness, but the gain is small and comes with a class trade-off.",
        "",
        "## Metrics",
        "",
        markdown_table(metrics_df[
            [
                "run",
                "pipeline_correct_total",
                "pipeline_correct_rate",
                "vlm_correct_rate_among_good_pred_crops",
                "ceiling_correct_rate",
                "ceiling_vs_actual_gap",
            ]
        ]),
        "",
        "## Per-Class Accuracy",
        "",
        markdown_table(class_cmp),
        "",
        "## Error Pattern Shift",
        "",
        markdown_table(pattern_cmp.head(12)),
        "",
        "## Helped / Hurt Cases",
        "",
        markdown_table(bucket_counts.rename_axis("bucket").reset_index(name="count")),
        "",
        "Full case list: `helped_hurt_cases.csv`.",
        "",
        "## Charts",
        "",
        "![Pipeline rate](pipeline_rate_comparison.png)",
        "",
        "![Ceiling gap](ceiling_gap_comparison.png)",
        "",
        "![Helped hurt counts](helped_hurt_counts.png)",
        "",
        "![Error patterns](error_pattern_comparison.png)",
        "",
        "## Interpretation",
        "",
        "The best current input strategy candidate is `padding_ratio=0.30` with `max_pixels=401408`.",
        "It is more useful than the tight crop because it closes part of the Stage 3-to-Stage 4 gap.",
        "It should not be interpreted as solving VLM semantics: flashover-vs-normal confusion remains the key bottleneck.",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
