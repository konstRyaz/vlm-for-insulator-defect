#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COARSE_CLASS_LABELS = ["insulator_ok", "defect_flashover", "defect_broken", "unknown", "other"]
VISIBILITY_LABELS = ["clear", "partial", "ambiguous"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an extended visualization and table package for Stage 3 eval artifacts."
    )
    parser.add_argument(
        "--eval-dir",
        required=True,
        help="Path to eval directory containing metrics.json/review_table.csv/confusion_*.csv",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for plots/report (default: <eval-dir>/visuals)",
    )
    parser.add_argument(
        "--ground-truth-jsonl",
        default=None,
        help="Optional GT JSONL for advanced tag analysis. If omitted, script tries auto-detection.",
    )
    parser.add_argument(
        "--predictions-jsonl",
        default=None,
        help="Optional predictions JSONL for advanced text/tag analysis. Default: <run-dir>/predictions_vlm_labels_v1.jsonl",
    )
    parser.add_argument(
        "--sweep-csv",
        default=None,
        help="Optional prompt sweep comparison CSV for extra Stage 3 summary charts.",
    )
    parser.add_argument(
        "--ablation-csv",
        default=None,
        help="Optional v6d-vs-v6f ablation CSV for extra comparison chart.",
    )
    parser.add_argument(
        "--top-tags",
        type=int,
        default=15,
        help="Top-N tags to show in tag frequency chart (default: 15).",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def load_confusion_csv(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError(f"Confusion CSV has no data rows: {path}")
    col_labels = rows[0][1:]
    row_labels = []
    values: list[list[float]] = []
    for row in rows[1:]:
        row_labels.append(row[0])
        values.append([float(x) for x in row[1:]])
    return row_labels, col_labels, np.array(values, dtype=float)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num) / float(den)


def normalize_tags(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    out: set[str] = set()
    for item in value:
        if isinstance(item, str) and item.strip():
            out.add(item.strip())
    return out


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def style_setup() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")


def save_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    out_path: Path,
    cmap: str = "Blues",
    normalize_rows: bool = False,
) -> None:
    data = matrix.copy()
    if normalize_rows:
        row_sum = data.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            data = np.divide(data, row_sum, where=row_sum != 0)
        data = np.nan_to_num(data)

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0.0)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title, pad=12)

    max_val = data.max() if data.size else 0.0
    threshold = max_val * 0.62
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if normalize_rows:
                txt = f"{val:.2f}"
            else:
                txt = f"{int(val)}"
            color = "white" if val > threshold else "#222222"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Rate" if normalize_rows else "Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_kpi_bar(metrics: dict[str, Any], out_path: Path) -> None:
    rates = metrics.get("rates", {})
    f1 = metrics.get("f1", {})

    labels = [
        "Parse success",
        "Schema valid",
        "Coarse acc",
        "Coarse macro-F1",
        "Visibility acc",
        "Visibility macro-F1",
        "Needs-review acc",
        "Tag mean Jaccard",
    ]
    values = [
        float(rates.get("parse_success_rate", 0.0)),
        float(rates.get("schema_valid_rate", 0.0)),
        float(rates.get("coarse_class_accuracy", 0.0)),
        float(f1.get("coarse_class_macro_f1", 0.0)),
        float(rates.get("visibility_accuracy", 0.0)),
        float(f1.get("visibility_macro_f1", 0.0)),
        float(rates.get("needs_review_accuracy", 0.0)),
        float(rates.get("tag_mean_jaccard", 0.0)),
    ]
    colors = ["#2E8B57", "#2E8B57", "#3B82F6", "#3B82F6", "#2563EB", "#2563EB", "#1D4ED8", "#0EA5A4"]

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Score")
    ax.set_title("Stage 3 KPI Overview", pad=12)

    for b, v in zip(bars, values):
        ax.text(min(0.985, v + 0.01), b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_distribution_comparison(
    review_df: pd.DataFrame,
    gt_col: str,
    pred_col: str,
    labels: list[str],
    title: str,
    out_path: Path,
) -> None:
    gt_counts = review_df[gt_col].value_counts().to_dict()
    pred_counts = review_df[pred_col].value_counts().to_dict()
    gt_vals = [int(gt_counts.get(x, 0)) for x in labels]
    pred_vals = [int(pred_counts.get(x, 0)) for x in labels]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.bar(x - width / 2, gt_vals, width, label="GT", color="#16A34A")
    ax.bar(x + width / 2, pred_vals, width, label="Pred", color="#DC2626")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(title, pad=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_visibility_bias_chart(metrics: dict[str, Any], review_df: pd.DataFrame, out_path: Path) -> None:
    labels = VISIBILITY_LABELS
    gt = review_df["gt_visibility"].value_counts().to_dict()
    pred = review_df["pred_visibility"].value_counts().to_dict()
    gt_vals = [int(gt.get(x, 0)) for x in labels]
    pred_vals = [int(pred.get(x, 0)) for x in labels]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.bar(x - width / 2, gt_vals, width, label="GT", color="#15803D")
    ax.bar(x + width / 2, pred_vals, width, label="Pred", color="#B91C1C")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_title("Visibility Distribution: GT vs Pred", pad=10)
    ax.legend()

    pred_amb = float(metrics.get("rates", {}).get("pred_ambiguous_rate", 0.0))
    gt_amb = float(metrics.get("rates", {}).get("gt_ambiguous_rate", 0.0))
    ax.text(
        0.02,
        0.96,
        f"ambiguous rate: pred={pred_amb:.3f}, gt={gt_amb:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "#94A3B8"},
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def failure_mode_table(review_df: pd.DataFrame) -> pd.DataFrame:
    df = review_df.copy()
    df["coarse_class_match"] = bool_series(df["coarse_class_match"])
    df["visibility_match"] = bool_series(df["visibility_match"])
    df["needs_review_match"] = bool_series(df["needs_review_match"])

    all_match = df["coarse_class_match"] & df["visibility_match"] & df["needs_review_match"]
    coarse_only = (~df["coarse_class_match"]) & df["visibility_match"] & df["needs_review_match"]
    visibility_only = df["coarse_class_match"] & (~df["visibility_match"]) & df["needs_review_match"]
    review_only = df["coarse_class_match"] & df["visibility_match"] & (~df["needs_review_match"])
    multiple = (~all_match) & (~coarse_only) & (~visibility_only) & (~review_only)

    mode_rows = [
        ("all_match", int(all_match.sum())),
        ("coarse_only_mismatch", int(coarse_only.sum())),
        ("visibility_only_mismatch", int(visibility_only.sum())),
        ("needs_review_only_mismatch", int(review_only.sum())),
        ("multiple_mismatch", int(multiple.sum())),
    ]
    return pd.DataFrame(mode_rows, columns=["failure_mode", "count"])


def save_failure_modes(mode_df: pd.DataFrame, out_path: Path) -> None:
    labels = mode_df["failure_mode"].tolist()
    vals = mode_df["count"].astype(int).tolist()
    colors = ["#16A34A", "#F59E0B", "#EF4444", "#8B5CF6", "#334155"]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    bars = ax.bar(labels, vals, color=colors[: len(labels)])
    ax.set_title("Failure Modes", pad=10)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, str(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_group_mismatch_table(review_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    df = review_df.copy()
    for col in ["coarse_class_match", "visibility_match", "needs_review_match"]:
        df[col] = bool_series(df[col])
    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(
            samples=("record_id", "count"),
            coarse_error_rate=("coarse_class_match", lambda s: 1.0 - float(s.mean())),
            visibility_error_rate=("visibility_match", lambda s: 1.0 - float(s.mean())),
            needs_review_error_rate=("needs_review_match", lambda s: 1.0 - float(s.mean())),
        )
        .reset_index()
        .sort_values("samples", ascending=False)
    )
    return grouped


def save_group_mismatch_chart(
    mismatch_df: pd.DataFrame,
    group_col: str,
    out_path: Path,
    title: str,
) -> None:
    if mismatch_df.empty:
        return
    labels = mismatch_df[group_col].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    ax.bar(x - width, mismatch_df["coarse_error_rate"], width, label="Coarse error", color="#EF4444")
    ax.bar(x, mismatch_df["visibility_error_rate"], width, label="Visibility error", color="#F59E0B")
    ax.bar(x + width, mismatch_df["needs_review_error_rate"], width, label="Needs-review error", color="#6366F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Error rate")
    ax.set_title(title, pad=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_text_length_assets(pred_rows: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    lengths: list[dict[str, Any]] = []
    for row in pred_rows:
        short = str(row.get("short_canonical_description_en", "") or "")
        report = str(row.get("report_snippet_en", "") or "")
        lengths.append(
            {
                "record_id": row.get("record_id"),
                "short_chars": len(short),
                "short_words": len([w for w in short.split() if w]),
                "report_chars": len(report),
                "report_words": len([w for w in report.split() if w]),
            }
        )
    length_df = pd.DataFrame(lengths)

    summary_df = pd.DataFrame(
        [
            {
                "field": "short_canonical_description_en",
                "mean_chars": float(length_df["short_chars"].mean()),
                "median_chars": float(length_df["short_chars"].median()),
                "p90_chars": float(length_df["short_chars"].quantile(0.9)),
                "mean_words": float(length_df["short_words"].mean()),
                "median_words": float(length_df["short_words"].median()),
                "p90_words": float(length_df["short_words"].quantile(0.9)),
            },
            {
                "field": "report_snippet_en",
                "mean_chars": float(length_df["report_chars"].mean()),
                "median_chars": float(length_df["report_chars"].median()),
                "p90_chars": float(length_df["report_chars"].quantile(0.9)),
                "mean_words": float(length_df["report_words"].mean()),
                "median_words": float(length_df["report_words"].median()),
                "p90_words": float(length_df["report_words"].quantile(0.9)),
            },
        ]
    )
    table_path = out_dir / "table_text_lengths.csv"
    summary_df.to_csv(table_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    axes[0].hist(length_df["short_chars"], bins=12, color="#2563EB", alpha=0.85)
    axes[0].set_title("Short Description Length (chars)")
    axes[0].set_xlabel("Chars")
    axes[0].set_ylabel("Count")
    axes[1].hist(length_df["report_chars"], bins=12, color="#0F766E", alpha=0.85)
    axes[1].set_title("Report Snippet Length (chars)")
    axes[1].set_xlabel("Chars")
    axes[1].set_ylabel("Count")
    fig.tight_layout()
    plot_path = out_dir / "text_length_distribution.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    return table_path, plot_path


def save_tag_assets(
    gt_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    review_df: pd.DataFrame,
    out_dir: Path,
    top_n: int,
) -> tuple[Path, Path, Path, Path]:
    gt_by_id = {str(r.get("record_id")): r for r in gt_rows if isinstance(r.get("record_id"), str)}
    pred_by_id = {str(r.get("record_id")): r for r in pred_rows if isinstance(r.get("record_id"), str)}

    record_ids = [str(x) for x in review_df["record_id"].tolist()]
    common_ids = [rid for rid in record_ids if rid in gt_by_id and rid in pred_by_id]

    gt_counter: Counter[str] = Counter()
    pred_counter: Counter[str] = Counter()
    jaccard_rows: list[dict[str, Any]] = []

    for rid in common_ids:
        gt = gt_by_id[rid]
        pred = pred_by_id[rid]
        gt_tags = normalize_tags(gt.get("visual_evidence_tags"))
        pred_tags = normalize_tags(pred.get("visual_evidence_tags"))
        gt_counter.update(gt_tags)
        pred_counter.update(pred_tags)

        union = gt_tags | pred_tags
        jaccard = safe_div(len(gt_tags & pred_tags), len(union)) if union else 1.0
        jaccard_rows.append(
            {
                "record_id": rid,
                "gt_coarse_class": gt.get("coarse_class", "other"),
                "jaccard": jaccard,
            }
        )

    all_tags = sorted(set(gt_counter.keys()) | set(pred_counter.keys()))
    freq_df = pd.DataFrame(
        [
            {
                "tag": tag,
                "gt_count": int(gt_counter.get(tag, 0)),
                "pred_count": int(pred_counter.get(tag, 0)),
                "abs_gap": int(abs(gt_counter.get(tag, 0) - pred_counter.get(tag, 0))),
            }
            for tag in all_tags
        ]
    )
    freq_df = freq_df.sort_values(["gt_count", "pred_count", "abs_gap"], ascending=False)
    table_freq = out_dir / "table_tag_frequency_comparison.csv"
    freq_df.to_csv(table_freq, index=False)

    top_df = freq_df.head(max(1, top_n)).copy().iloc[::-1]
    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    y = np.arange(len(top_df))
    h = 0.38
    ax.barh(y - h / 2, top_df["gt_count"], h, label="GT", color="#16A34A")
    ax.barh(y + h / 2, top_df["pred_count"], h, label="Pred", color="#DC2626")
    ax.set_yticks(y)
    ax.set_yticklabels(top_df["tag"])
    ax.set_xlabel("Count")
    ax.set_title(f"Tag Frequency: GT vs Pred (top {len(top_df)})", pad=10)
    ax.legend()
    fig.tight_layout()
    plot_freq = out_dir / "tag_frequency_gt_vs_pred.png"
    fig.savefig(plot_freq, dpi=180)
    plt.close(fig)

    jaccard_df = pd.DataFrame(jaccard_rows)
    by_class = (
        jaccard_df.groupby("gt_coarse_class", dropna=False)
        .agg(mean_jaccard=("jaccard", "mean"), samples=("record_id", "count"))
        .reset_index()
        .sort_values("samples", ascending=False)
    )
    table_jaccard = out_dir / "table_tag_jaccard_by_gt_coarse.csv"
    by_class.to_csv(table_jaccard, index=False)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bars = ax.bar(by_class["gt_coarse_class"].astype(str), by_class["mean_jaccard"], color="#0EA5A4")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean Jaccard")
    ax.set_title("Tag Overlap by GT Coarse Class", pad=10)
    ax.tick_params(axis="x", rotation=20)
    for bar, sample_count in zip(bars, by_class["samples"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"n={int(sample_count)}", ha="center", fontsize=8)
    fig.tight_layout()
    plot_jaccard = out_dir / "tag_jaccard_by_gt_coarse.png"
    fig.savefig(plot_jaccard, dpi=180)
    plt.close(fig)

    return table_freq, plot_freq, table_jaccard, plot_jaccard


def resolve_ground_truth_jsonl(eval_dir: Path, explicit: str | None) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))

    run_dir = eval_dir.parent
    run_summary_path = run_dir / "run_summary.json"
    if run_summary_path.exists():
        try:
            run_summary = read_json(run_summary_path)
            ds = run_summary.get("dataset_jsonl")
            if isinstance(ds, str) and ds.strip():
                candidates.append(Path(ds))
        except Exception:
            pass

    repo_root = Path(__file__).resolve().parent.parent
    candidates.append(repo_root / "outputs/stage3_regrouped_v2/val/vlm_labels_v1_val_v2.annotated.jsonl")
    candidates.append(repo_root / "outputs/stage3_pilot_mini/val/vlm_labels_v1_pilot.annotated.jsonl")

    for p in candidates:
        if p.exists():
            return p.resolve()
    return None


def resolve_predictions_jsonl(eval_dir: Path, explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p.resolve()
    run_dir = eval_dir.parent
    p = run_dir / "predictions_vlm_labels_v1.jsonl"
    if p.exists():
        return p.resolve()
    return None


def build_metrics_tables(metrics: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    rates = metrics.get("rates", {})
    f1 = metrics.get("f1", {})

    kpi_rows = [
        ("parse_success_rate", float(rates.get("parse_success_rate", 0.0)), "==1.00"),
        ("schema_valid_rate", float(rates.get("schema_valid_rate", 0.0)), "==1.00"),
        ("coarse_class_accuracy", float(rates.get("coarse_class_accuracy", 0.0)), ">=0.96"),
        ("coarse_class_macro_f1", float(f1.get("coarse_class_macro_f1", 0.0)), ">=0.55"),
        ("visibility_accuracy", float(rates.get("visibility_accuracy", 0.0)), "-"),
        ("visibility_macro_f1", float(f1.get("visibility_macro_f1", 0.0)), ">=0.45"),
        ("needs_review_accuracy", float(rates.get("needs_review_accuracy", 0.0)), "-"),
        ("tag_exact_match_rate", float(rates.get("tag_exact_match_rate", 0.0)), "-"),
        ("tag_mean_jaccard", float(rates.get("tag_mean_jaccard", 0.0)), ">=0.34"),
        ("pred_ambiguous_rate", float(rates.get("pred_ambiguous_rate", 0.0)), "0.10..0.20"),
        ("gt_ambiguous_rate", float(rates.get("gt_ambiguous_rate", 0.0)), "reference"),
    ]

    kpi_df = pd.DataFrame(kpi_rows, columns=["metric", "value", "target"])
    table_kpi = out_dir / "table_kpi_summary.csv"
    kpi_df.to_csv(table_kpi, index=False)

    coarse_dict = metrics.get("f1", {}).get("coarse_class_f1_per_label", {})
    vis_dict = metrics.get("f1", {}).get("visibility_f1_per_label", {})
    f1_df = pd.DataFrame(
        [
            {"task": "coarse_class", "label": k, "f1": float(v)}
            for k, v in coarse_dict.items()
        ]
        + [
            {"task": "visibility", "label": k, "f1": float(v)}
            for k, v in vis_dict.items()
        ]
    )
    table_f1 = out_dir / "table_f1_per_label.csv"
    f1_df.to_csv(table_f1, index=False)
    return table_kpi, table_f1


def save_f1_per_label_chart(metrics: dict[str, Any], out_path: Path) -> None:
    coarse_dict = metrics.get("f1", {}).get("coarse_class_f1_per_label", {})
    vis_dict = metrics.get("f1", {}).get("visibility_f1_per_label", {})

    coarse_labels = list(coarse_dict.keys())
    coarse_vals = [float(coarse_dict[k]) for k in coarse_labels]
    vis_labels = list(vis_dict.keys())
    vis_vals = [float(vis_dict[k]) for k in vis_labels]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    axes[0].bar(coarse_labels, coarse_vals, color="#2563EB")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Coarse Class F1 by Label")
    axes[0].tick_params(axis="x", rotation=22)
    axes[1].bar(vis_labels, vis_vals, color="#EA580C")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Visibility F1 by Label")
    axes[1].tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def decide_prompt_action(metrics: dict[str, Any]) -> tuple[str, list[str]]:
    rates = metrics.get("rates", {})
    f1 = metrics.get("f1", {})
    reasons: list[str] = []

    coarse_acc = float(rates.get("coarse_class_accuracy", 0.0))
    coarse_f1 = float(f1.get("coarse_class_macro_f1", 0.0))
    visibility_macro_f1 = float(f1.get("visibility_macro_f1", 0.0))
    pred_amb = float(rates.get("pred_ambiguous_rate", 0.0))
    gt_amb = float(rates.get("gt_ambiguous_rate", 0.0))
    tag_j = float(rates.get("tag_mean_jaccard", 0.0))

    if visibility_macro_f1 < 0.45:
        reasons.append(f"visibility macro-F1 is below target ({visibility_macro_f1:.3f} < 0.45)")
    if gt_amb > 0 and abs(pred_amb - gt_amb) > 0.07:
        reasons.append(
            f"ambiguous calibration gap is high (pred={pred_amb:.3f}, gt={gt_amb:.3f}, gap={abs(pred_amb - gt_amb):.3f})"
        )
    if coarse_acc < 0.96 or coarse_f1 < 0.55:
        reasons.append(f"coarse metrics are below preferred guardrail (acc={coarse_acc:.3f}, macro-F1={coarse_f1:.3f})")
    if tag_j < 0.34:
        reasons.append(f"tag overlap is below target (mean Jaccard={tag_j:.3f})")

    if reasons:
        return "IMPROVE_PROMPT_OR_DATA_POLICY", reasons
    return "STAGE3_READY_TO_FREEZE", ["metrics satisfy current Stage 3 guardrails"]


def write_markdown_report(
    out_path: Path,
    metrics: dict[str, Any],
    verdict: str,
    reasons: list[str],
    mode_df: pd.DataFrame,
    extra_sections: list[str],
) -> None:
    lines: list[str] = []
    lines.append("# Stage 3 Final Visual Report")
    lines.append("")
    lines.append("## Verdict")
    lines.append(f"- `{verdict}`")
    for r in reasons:
        lines.append(f"- {r}")
    lines.append("")
    lines.append("## Key Scores")
    rates = metrics.get("rates", {})
    f1 = metrics.get("f1", {})
    lines.append(f"- Coarse accuracy: `{float(rates.get('coarse_class_accuracy', 0.0)):.3f}`")
    lines.append(f"- Coarse macro-F1: `{float(f1.get('coarse_class_macro_f1', 0.0)):.3f}`")
    lines.append(f"- Visibility accuracy: `{float(rates.get('visibility_accuracy', 0.0)):.3f}`")
    lines.append(f"- Visibility macro-F1: `{float(f1.get('visibility_macro_f1', 0.0)):.3f}`")
    lines.append(f"- Needs-review accuracy: `{float(rates.get('needs_review_accuracy', 0.0)):.3f}`")
    lines.append(f"- Tag mean Jaccard: `{float(rates.get('tag_mean_jaccard', 0.0)):.3f}`")
    lines.append("")
    lines.append("## Failure Modes")
    for row in mode_df.to_dict(orient="records"):
        lines.append(f"- {row['failure_mode']}: `{row['count']}`")
    lines.append("")
    lines.append("## Core Charts")
    lines.append("![KPI](kpi_overview.png)")
    lines.append("")
    lines.append("![F1 per label](f1_per_label.png)")
    lines.append("")
    lines.append("![Coarse confusion](confusion_coarse_class.png)")
    lines.append("")
    lines.append("![Coarse confusion normalized](confusion_coarse_class_norm.png)")
    lines.append("")
    lines.append("![Visibility confusion](confusion_visibility.png)")
    lines.append("")
    lines.append("![Visibility confusion normalized](confusion_visibility_norm.png)")
    lines.append("")
    lines.append("![Coarse distribution](coarse_class_distribution.png)")
    lines.append("")
    lines.append("![Visibility distribution](visibility_distribution.png)")
    lines.append("")
    lines.append("![Visibility bias](visibility_bias.png)")
    lines.append("")
    lines.append("![Failure modes](failure_modes.png)")
    lines.append("")
    lines.append("![Mismatch by GT coarse](mismatch_rate_by_gt_coarse.png)")
    lines.append("")
    lines.append("![Mismatch by GT visibility](mismatch_rate_by_gt_visibility.png)")
    lines.append("")
    lines.extend(extra_sections)
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sweep_short_name(version: str) -> str:
    prefix = "qwen_vlm_labels_v1_prompt_"
    if version.startswith(prefix):
        return version[len(prefix) :]
    return version


def create_sweep_assets(sweep_csv: Path, out_dir: Path) -> tuple[list[str], list[Path]]:
    df = pd.read_csv(sweep_csv)
    generated: list[Path] = []
    sections: list[str] = []
    if df.empty:
        return sections, generated

    if "rank" in df.columns:
        df = df.sort_values("rank")
    df["prompt_short"] = df["prompt_version"].astype(str).map(sweep_short_name)

    keep_cols = [
        "prompt_version",
        "prompt_short",
        "rank",
        "verdict",
        "coarse_acc",
        "coarse_macro_f1",
        "visibility_acc",
        "visibility_macro_f1",
        "needs_review_acc",
        "tag_mean_jaccard",
        "pred_ambiguous_rate",
        "gt_ambiguous_rate",
        "abs_ambiguous_gap",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    table_path = out_dir / "table_prompt_sweep_leaderboard.csv"
    df[keep_cols].to_csv(table_path, index=False)
    generated.append(table_path)

    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    ax.bar(df["prompt_short"], df["visibility_macro_f1"], color="#2563EB")
    ax.set_ylim(0.0, max(0.6, float(df["visibility_macro_f1"].max()) + 0.05))
    ax.set_title("Prompt Sweep: Visibility Macro-F1")
    ax.set_ylabel("Macro-F1")
    ax.tick_params(axis="x", rotation=28)
    plt.setp(ax.get_xticklabels(), ha="right")
    fig.tight_layout()
    plot_vis = out_dir / "sweep_visibility_macro_f1.png"
    fig.savefig(plot_vis, dpi=180)
    plt.close(fig)
    generated.append(plot_vis)

    if "pred_ambiguous_rate" in df.columns and "visibility_macro_f1" in df.columns:
        fig, ax = plt.subplots(figsize=(8.6, 5.6))
        ax.scatter(
            df["pred_ambiguous_rate"],
            df["visibility_macro_f1"],
            s=170,
            c=df["coarse_acc"] if "coarse_acc" in df.columns else "#0EA5A4",
            cmap="viridis",
            alpha=0.9,
        )
        for _, row in df.iterrows():
            ax.annotate(
                str(row["prompt_short"]),
                (row["pred_ambiguous_rate"], row["visibility_macro_f1"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_xlabel("Pred ambiguous rate")
        ax.set_ylabel("Visibility macro-F1")
        ax.set_title("Prompt Sweep Trade-off: Ambiguous Rate vs Visibility Macro-F1")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        plot_tradeoff = out_dir / "sweep_tradeoff_ambiguous_vs_visibility.png"
        fig.savefig(plot_tradeoff, dpi=180)
        plt.close(fig)
        generated.append(plot_tradeoff)

    sections.append("## Prompt Sweep (v3/v4/v5a/v6*)")
    sections.append("![Sweep visibility macro-F1](sweep_visibility_macro_f1.png)")
    if (out_dir / "sweep_tradeoff_ambiguous_vs_visibility.png").exists():
        sections.append("")
        sections.append("![Sweep trade-off](sweep_tradeoff_ambiguous_vs_visibility.png)")
    return sections, generated


def create_ablation_assets(ablation_csv: Path, out_dir: Path) -> tuple[list[str], list[Path]]:
    df = pd.read_csv(ablation_csv)
    generated: list[Path] = []
    sections: list[str] = []
    if df.empty or "prompt_version" not in df.columns:
        return sections, generated

    df["prompt_short"] = df["prompt_version"].astype(str).map(sweep_short_name)
    table_path = out_dir / "table_v6d_vs_v6f_ablation.csv"
    df.to_csv(table_path, index=False)
    generated.append(table_path)

    metrics_to_plot = [
        "coarse_acc",
        "visibility_macro_f1",
        "needs_review_acc",
        "tag_mean_jaccard",
        "pred_ambiguous_rate",
        "abs_ambiguous_gap",
    ]
    plot_rows = []
    for m in metrics_to_plot:
        if m in df.columns:
            for _, r in df.iterrows():
                plot_rows.append({"metric": m, "value": float(r[m]), "prompt": r["prompt_short"]})
    plot_df = pd.DataFrame(plot_rows)
    if not plot_df.empty:
        pivot = plot_df.pivot(index="metric", columns="prompt", values="value")
        fig, ax = plt.subplots(figsize=(9.4, 5.2))
        x = np.arange(len(pivot.index))
        cols = list(pivot.columns)
        width = 0.36 if len(cols) == 2 else max(0.18, 0.7 / max(1, len(cols)))
        offsets = np.linspace(-width * (len(cols) - 1) / 2, width * (len(cols) - 1) / 2, len(cols))
        colors = ["#2563EB", "#DC2626", "#0EA5A4", "#7C3AED"]
        for idx, col in enumerate(cols):
            vals = pivot[col].values
            ax.bar(x + offsets[idx], vals, width=width, label=col, color=colors[idx % len(colors)])
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=24, ha="right")
        ax.set_title("v6d vs v6f: Metric Comparison")
        ax.set_ylabel("Value")
        ax.legend()
        fig.tight_layout()
        plot_path = out_dir / "ablation_v6d_vs_v6f_metrics.png"
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        generated.append(plot_path)

    sections.append("## Final Micro-Ablation (v6d vs v6f)")
    if (out_dir / "ablation_v6d_vs_v6f_metrics.png").exists():
        sections.append("![v6d vs v6f](ablation_v6d_vs_v6f_metrics.png)")
    return sections, generated


def main() -> None:
    args = parse_args()
    style_setup()

    eval_dir = Path(args.eval_dir).resolve()
    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval dir not found: {eval_dir}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else eval_dir / "visuals"
    ensure_dir(out_dir)

    metrics_path = eval_dir / "metrics.json"
    review_table_path = eval_dir / "review_table.csv"
    conf_coarse_path = eval_dir / "confusion_coarse_class.csv"
    conf_visibility_path = eval_dir / "confusion_visibility.csv"

    metrics = read_json(metrics_path)
    review_df = pd.read_csv(review_table_path)
    coarse_rows, coarse_cols, coarse_mat = load_confusion_csv(conf_coarse_path)
    vis_rows, vis_cols, vis_mat = load_confusion_csv(conf_visibility_path)

    generated_paths: list[Path] = []

    save_kpi_bar(metrics, out_dir / "kpi_overview.png")
    generated_paths.append(out_dir / "kpi_overview.png")

    save_f1_per_label_chart(metrics, out_dir / "f1_per_label.png")
    generated_paths.append(out_dir / "f1_per_label.png")

    save_heatmap(
        coarse_mat,
        coarse_rows,
        coarse_cols,
        "Coarse Class Confusion (counts)",
        out_dir / "confusion_coarse_class.png",
        cmap="Blues",
        normalize_rows=False,
    )
    generated_paths.append(out_dir / "confusion_coarse_class.png")

    save_heatmap(
        coarse_mat,
        coarse_rows,
        coarse_cols,
        "Coarse Class Confusion (row-normalized)",
        out_dir / "confusion_coarse_class_norm.png",
        cmap="GnBu",
        normalize_rows=True,
    )
    generated_paths.append(out_dir / "confusion_coarse_class_norm.png")

    save_heatmap(
        vis_mat,
        vis_rows,
        vis_cols,
        "Visibility Confusion (counts)",
        out_dir / "confusion_visibility.png",
        cmap="OrRd",
        normalize_rows=False,
    )
    generated_paths.append(out_dir / "confusion_visibility.png")

    save_heatmap(
        vis_mat,
        vis_rows,
        vis_cols,
        "Visibility Confusion (row-normalized)",
        out_dir / "confusion_visibility_norm.png",
        cmap="YlOrBr",
        normalize_rows=True,
    )
    generated_paths.append(out_dir / "confusion_visibility_norm.png")

    save_distribution_comparison(
        review_df,
        gt_col="gt_coarse_class",
        pred_col="pred_coarse_class",
        labels=COARSE_CLASS_LABELS,
        title="Coarse Class Distribution: GT vs Pred",
        out_path=out_dir / "coarse_class_distribution.png",
    )
    generated_paths.append(out_dir / "coarse_class_distribution.png")

    save_distribution_comparison(
        review_df,
        gt_col="gt_visibility",
        pred_col="pred_visibility",
        labels=VISIBILITY_LABELS,
        title="Visibility Distribution: GT vs Pred",
        out_path=out_dir / "visibility_distribution.png",
    )
    generated_paths.append(out_dir / "visibility_distribution.png")

    save_visibility_bias_chart(metrics, review_df, out_dir / "visibility_bias.png")
    generated_paths.append(out_dir / "visibility_bias.png")

    mode_df = failure_mode_table(review_df)
    mode_df.to_csv(out_dir / "table_failure_modes.csv", index=False)
    generated_paths.append(out_dir / "table_failure_modes.csv")
    save_failure_modes(mode_df, out_dir / "failure_modes.png")
    generated_paths.append(out_dir / "failure_modes.png")

    mismatch_coarse_df = build_group_mismatch_table(review_df, "gt_coarse_class")
    mismatch_vis_df = build_group_mismatch_table(review_df, "gt_visibility")
    mismatch_coarse_df.to_csv(out_dir / "table_mismatch_by_gt_coarse.csv", index=False)
    mismatch_vis_df.to_csv(out_dir / "table_mismatch_by_gt_visibility.csv", index=False)
    generated_paths.extend(
        [
            out_dir / "table_mismatch_by_gt_coarse.csv",
            out_dir / "table_mismatch_by_gt_visibility.csv",
        ]
    )

    save_group_mismatch_chart(
        mismatch_coarse_df,
        group_col="gt_coarse_class",
        out_path=out_dir / "mismatch_rate_by_gt_coarse.png",
        title="Mismatch Rates by GT Coarse Class",
    )
    generated_paths.append(out_dir / "mismatch_rate_by_gt_coarse.png")

    save_group_mismatch_chart(
        mismatch_vis_df,
        group_col="gt_visibility",
        out_path=out_dir / "mismatch_rate_by_gt_visibility.png",
        title="Mismatch Rates by GT Visibility",
    )
    generated_paths.append(out_dir / "mismatch_rate_by_gt_visibility.png")

    kpi_table_path, f1_table_path = build_metrics_tables(metrics, out_dir)
    generated_paths.extend([kpi_table_path, f1_table_path])

    # Advanced text/tag analysis when prediction and GT records are available.
    predictions_jsonl = resolve_predictions_jsonl(eval_dir, args.predictions_jsonl)
    gt_jsonl = resolve_ground_truth_jsonl(eval_dir, args.ground_truth_jsonl)

    extra_sections: list[str] = []

    if predictions_jsonl and predictions_jsonl.exists():
        pred_rows = load_jsonl(predictions_jsonl)
        text_table, text_plot = save_text_length_assets(pred_rows, out_dir)
        generated_paths.extend([text_table, text_plot])
        extra_sections.extend(
            [
                "## Text Length Diagnostics",
                "![Text length distribution](text_length_distribution.png)",
                "",
            ]
        )

        if gt_jsonl and gt_jsonl.exists():
            gt_rows = load_jsonl(gt_jsonl)
            tag_table, tag_plot, j_table, j_plot = save_tag_assets(
                gt_rows=gt_rows,
                pred_rows=pred_rows,
                review_df=review_df,
                out_dir=out_dir,
                top_n=args.top_tags,
            )
            generated_paths.extend([tag_table, tag_plot, j_table, j_plot])
            extra_sections.extend(
                [
                    "## Tag Diagnostics",
                    "![Tag frequency GT vs Pred](tag_frequency_gt_vs_pred.png)",
                    "",
                    "![Tag Jaccard by GT coarse class](tag_jaccard_by_gt_coarse.png)",
                    "",
                ]
            )

    if args.sweep_csv:
        sweep_path = Path(args.sweep_csv).resolve()
        if sweep_path.exists():
            sec, generated = create_sweep_assets(sweep_path, out_dir)
            extra_sections.extend(sec + [""])
            generated_paths.extend(generated)

    if args.ablation_csv:
        ablation_path = Path(args.ablation_csv).resolve()
        if ablation_path.exists():
            sec, generated = create_ablation_assets(ablation_path, out_dir)
            extra_sections.extend(sec + [""])
            generated_paths.extend(generated)

    verdict, reasons = decide_prompt_action(metrics)
    verdict_payload = {"verdict": verdict, "reasons": reasons}
    (out_dir / "verdict.json").write_text(
        json.dumps(verdict_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    generated_paths.append(out_dir / "verdict.json")

    write_markdown_report(
        out_path=out_dir / "report.md",
        metrics=metrics,
        verdict=verdict,
        reasons=reasons,
        mode_df=mode_df,
        extra_sections=extra_sections,
    )
    generated_paths.append(out_dir / "report.md")

    manifest_df = pd.DataFrame(
        {
            "artifact": [
                str(p.relative_to(out_dir))
                for p in generated_paths
                if p.exists() and p.is_relative_to(out_dir)
            ]
        }
    )
    manifest_path = out_dir / "artifacts_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    print(f"Visual report generated: {out_dir}")
    print(f"Verdict: {verdict}")
    for reason in reasons:
        print(f"- {reason}")
    print(f"Advanced GT path: {gt_jsonl if gt_jsonl else 'not found'}")
    print(f"Predictions path: {predictions_jsonl if predictions_jsonl else 'not found'}")
    print(f"Artifacts manifest: {manifest_path}")


if __name__ == "__main__":
    main()
