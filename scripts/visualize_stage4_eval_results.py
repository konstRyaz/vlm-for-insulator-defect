#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import tarfile
import textwrap
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COARSE_CLASS_LABELS = ["insulator_ok", "defect_flashover", "defect_broken", "unknown", "other"]
ERROR_BUCKET_ORDER = [
    "correct_pipeline_hit",
    "vlm_error_on_good_pred_crop",
    "bad_crop_from_detector",
    "detector_miss",
    "routing_or_filtering_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a visual explanation package for Stage 4 eval artifacts.")
    parser.add_argument("--eval-dir", required=True, type=str, help="Stage 4 eval directory.")
    parser.add_argument("--pred-manifest-jsonl", required=True, type=str, help="Pred manifest JSONL used by Stage 4.")
    parser.add_argument(
        "--stage4-archive",
        default=None,
        type=str,
        help="Optional .tar.gz archive with Stage 4 outputs; used for crop galleries.",
    )
    parser.add_argument(
        "--user-prompt-template",
        default=None,
        type=str,
        help="Optional user prompt template path to document metadata leakage risk.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        type=str,
        help="Output directory for visual package. Default: <eval-dir>/visuals",
    )
    parser.add_argument(
        "--top-gallery-errors",
        default=12,
        type=int,
        help="Number of top error examples to render in gallery.",
    )
    parser.add_argument(
        "--top-gallery-hits",
        default=9,
        type=int,
        help="Number of correct examples to render in gallery.",
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def boolish(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def style_setup() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num) / float(den)


def save_bar_chart(
    labels: list[str],
    values: list[float],
    out_path: Path,
    title: str,
    ylabel: str,
    colors: list[str],
    ylim: tuple[float, float] | None = None,
    fmt: str = "{:.3f}",
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.72)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=16, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12)
    if ylim is not None:
        ax.set_ylim(*ylim)

    top = max(values) if values else 1.0
    pad = 0.02 * (ylim[1] if ylim else max(1.0, top))
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + pad,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_grouped_bar_chart(
    labels: list[str],
    series_map: dict[str, list[float]],
    out_path: Path,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    colors = ["#2563EB", "#0F766E", "#D97706", "#DC2626"]
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    x = np.arange(len(labels))
    keys = list(series_map.keys())
    width = 0.8 / max(1, len(keys))

    for idx, key in enumerate(keys):
        values = series_map[key]
        offset = (idx - (len(keys) - 1) / 2.0) * width
        bars = ax.bar(x + offset, values, width=width, label=key, color=colors[idx % len(colors)])
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_counts_bar_chart(labels: list[str], values: list[int], out_path: Path, title: str) -> None:
    colors = ["#0F766E", "#2563EB", "#6D28D9", "#DC2626", "#B45309", "#64748B"]
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors[: len(labels)], width=0.72)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(title, pad=12)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5, str(value), ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    out_path: Path,
    title: str,
    normalize_rows: bool = False,
) -> None:
    data = matrix.copy().astype(float)
    if normalize_rows:
        row_sum = data.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            data = np.divide(data, row_sum, where=row_sum != 0)
        data = np.nan_to_num(data)

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    im = ax.imshow(data, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title, pad=12)

    threshold = data.max() * 0.62 if data.size else 0.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            text = f"{val:.2f}" if normalize_rows else f"{int(val)}"
            color = "white" if val > threshold else "#1f2937"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Rate" if normalize_rows else "Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_scatter(case_df: pd.DataFrame, out_path: Path) -> None:
    df = case_df.copy()
    df["matched_pred_score"] = pd.to_numeric(df["matched_pred_score"], errors="coerce")
    df["match_iou"] = pd.to_numeric(df["match_iou"], errors="coerce")

    color_map = {
        "correct_pipeline_hit": "#0F766E",
        "vlm_error_on_good_pred_crop": "#DC2626",
        "bad_crop_from_detector": "#B45309",
        "detector_miss": "#64748B",
        "routing_or_filtering_error": "#7C3AED",
    }

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    for bucket in ERROR_BUCKET_ORDER:
        sub = df[df["error_bucket"] == bucket]
        if sub.empty:
            continue
        ax.scatter(
            sub["matched_pred_score"],
            sub["match_iou"],
            s=60,
            alpha=0.82,
            label=bucket,
            c=color_map.get(bucket, "#334155"),
            edgecolors="white",
            linewidths=0.6,
        )

    ax.set_xlabel("Detector score")
    ax.set_ylabel("GT-pred IoU")
    ax.set_title("Stage 4 outcomes by detector score and crop IoU", pad=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.48, 1.01)
    ax.legend(frameon=True, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_text_panel(lines: list[str], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=12)
    body = "\n".join(lines)
    ax.text(0.01, 0.98, body, va="top", ha="left", fontsize=11, family="monospace")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def default_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 18)
    except Exception:
        return ImageFont.load_default()


def wrap_caption(text: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def extract_image_from_archive(archive_path: Path, crop_rel_path: str) -> Image.Image | None:
    suffix = f"02_pred_crops/{crop_rel_path.replace(chr(92), '/')}"
    with tarfile.open(archive_path, "r:gz") as tf:
        matches = [name for name in tf.getnames() if name.endswith(suffix)]
        if not matches:
            return None
        data = tf.extractfile(matches[0]).read()
    return Image.open(BytesIO(data)).convert("RGB")


def make_gallery(
    rows: list[dict[str, Any]],
    pred_manifest_by_id: dict[str, dict[str, Any]],
    archive_path: Path | None,
    out_path: Path,
    title: str,
    subtitle: str,
    columns: int = 3,
    tile_size: tuple[int, int] = (360, 260),
) -> bool:
    if archive_path is None or not archive_path.exists() or not rows:
        return False

    font = default_font()
    title_font = font
    caption_h = 120
    pad = 18
    width, height = tile_size

    prepared: list[tuple[Image.Image, str]] = []
    for row in rows:
        pred_id = str(row.get("matched_pred_record_id", "")).strip()
        manifest = pred_manifest_by_id.get(pred_id)
        if not isinstance(manifest, dict):
            continue
        crop_path = str(manifest.get("crop_path", "")).strip()
        if not crop_path:
            continue
        image = extract_image_from_archive(archive_path, crop_path)
        if image is None:
            continue
        image = ImageOps.contain(image, tile_size)
        canvas = Image.new("RGB", tile_size, "white")
        canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
        caption = (
            f"{row['record_id']}\n"
            f"GT {row['gt_coarse_class']} | Det {row['matched_pred_category_name']} | VLM {row['pred_vlm_coarse_class']}\n"
            f"IoU {float(row['match_iou']):.3f} | score {float(row['matched_pred_score']):.3f}"
        )
        prepared.append((canvas, wrap_caption(caption)))

    if not prepared:
        return False

    rows_n = math.ceil(len(prepared) / columns)
    header_h = 96
    gallery = Image.new(
        "RGB",
        (columns * (width + pad) + pad, header_h + rows_n * (height + caption_h + pad) + pad),
        "#F8FAFC",
    )
    draw = ImageDraw.Draw(gallery)
    draw.text((pad, 18), title, fill="#0F172A", font=title_font)
    draw.text((pad, 48), subtitle, fill="#334155", font=font)

    for idx, (img, caption) in enumerate(prepared):
        row_idx = idx // columns
        col_idx = idx % columns
        x = pad + col_idx * (width + pad)
        y = header_h + row_idx * (height + caption_h + pad)
        gallery.paste(img, (x, y))
        draw.rectangle([x, y, x + width, y + height], outline="#CBD5E1", width=2)
        draw.multiline_text((x + 8, y + height + 10), caption, fill="#0F172A", font=font, spacing=4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gallery.save(out_path)
    return True


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (eval_dir / "visuals")
    pred_manifest_path = Path(args.pred_manifest_jsonl).resolve()
    archive_path = Path(args.stage4_archive).resolve() if args.stage4_archive else None
    prompt_template_path = Path(args.user_prompt_template).resolve() if args.user_prompt_template else None

    metrics_path = eval_dir / "stage4_metrics.json"
    breakdown_path = eval_dir / "stage4_error_breakdown.json"
    case_table_path = eval_dir / "stage4_case_table.csv"
    ceiling_path = eval_dir / "ceiling_vs_actual.json"

    for path in [metrics_path, breakdown_path, case_table_path, ceiling_path, pred_manifest_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    ensure_dir(out_dir)
    style_setup()

    metrics = read_json(metrics_path)
    breakdown = read_json(breakdown_path)
    ceiling = read_json(ceiling_path)
    case_df = pd.read_csv(case_table_path)
    pred_manifest_rows = load_jsonl(pred_manifest_path)
    pred_manifest_by_id = {str(row.get("record_id", "")).strip(): row for row in pred_manifest_rows}

    for col in ["detector_found", "is_good_crop", "vlm_correct_on_good_crop", "raw_has_match_candidate", "filtered_has_match_candidate", "ceiling_correct"]:
        if col in case_df.columns:
            case_df[col] = boolish(case_df[col])

    case_df["matched_pred_score"] = pd.to_numeric(case_df["matched_pred_score"], errors="coerce")
    case_df["match_iou"] = pd.to_numeric(case_df["match_iou"], errors="coerce")

    good_df = case_df[case_df["is_good_crop"] == True].copy()
    error_df = case_df[case_df["error_bucket"] == "vlm_error_on_good_pred_crop"].copy()
    extra_drop_df = error_df[error_df["ceiling_correct"] == True].copy()

    counts = metrics.get("counts", {})
    rates = metrics.get("rates", {})
    error_counts = breakdown.get("counts", {})

    save_counts_bar_chart(
        labels=["GT total", "Detector found", "Good crops", "Actual correct", "Ceiling correct"],
        values=[
            int(counts.get("gt_objects_total", 0)),
            int(counts.get("detector_found_total", 0)),
            int(counts.get("good_crop_total", 0)),
            int(counts.get("pipeline_correct_total", 0)),
            int(counts.get("ceiling_correct_total", 0)),
        ],
        out_path=out_dir / "pipeline_funnel_counts.png",
        title="Stage 4 pipeline funnel and ceiling",
    )

    save_bar_chart(
        labels=["Match rate", "Good crop rate", "VLM on good crop", "Actual pipeline", "Stage 3 ceiling"],
        values=[
            float(rates.get("detector_match_rate", 0.0)),
            float(rates.get("good_crop_rate_among_matched", 0.0)),
            float(rates.get("vlm_correct_rate_among_good_pred_crops", 0.0)),
            float(rates.get("pipeline_correct_rate", 0.0)),
            float(rates.get("ceiling_correct_rate", 0.0)),
        ],
        out_path=out_dir / "pipeline_rates.png",
        title="Stage 4 rates: where accuracy is lost",
        ylabel="Rate",
        colors=["#0F766E", "#2563EB", "#6D28D9", "#DC2626", "#0EA5A4"],
        ylim=(0.0, 1.03),
    )

    save_counts_bar_chart(
        labels=ERROR_BUCKET_ORDER,
        values=[int(error_counts.get(label, 0)) for label in ERROR_BUCKET_ORDER],
        out_path=out_dir / "error_buckets.png",
        title="Stage 4 error bucket breakdown",
    )

    conf_df = pd.crosstab(good_df["gt_coarse_class"], good_df["pred_vlm_coarse_class"])
    rows = [label for label in COARSE_CLASS_LABELS if label in conf_df.index]
    cols = [label for label in COARSE_CLASS_LABELS if label in conf_df.columns]
    matrix = conf_df.reindex(index=rows, columns=cols, fill_value=0).to_numpy(dtype=float)
    save_heatmap(
        matrix=matrix,
        row_labels=rows,
        col_labels=cols,
        out_path=out_dir / "coarse_confusion_good_crops.png",
        title="GT vs VLM coarse class on good predicted crops",
        normalize_rows=False,
    )
    save_heatmap(
        matrix=matrix,
        row_labels=rows,
        col_labels=cols,
        out_path=out_dir / "coarse_confusion_good_crops_norm.png",
        title="GT vs VLM coarse class on good predicted crops (row-normalized)",
        normalize_rows=True,
    )

    class_rows: list[dict[str, Any]] = []
    for gt_class, sub in good_df.groupby("gt_coarse_class"):
        detector_correct = safe_div((sub["matched_pred_category_name"] == sub["gt_coarse_class"]).sum(), len(sub))
        actual_correct = safe_div((sub["pred_vlm_coarse_class"] == sub["gt_coarse_class"]).sum(), len(sub))
        ceiling_correct = safe_div(sub["ceiling_correct"].sum(), len(sub))
        class_rows.append(
            {
                "gt_coarse_class": gt_class,
                "n": int(len(sub)),
                "detector_label_acc": detector_correct,
                "stage4_actual_acc": actual_correct,
                "stage3_ceiling_acc": ceiling_correct,
            }
        )
    class_rows = sorted(class_rows, key=lambda x: COARSE_CLASS_LABELS.index(x["gt_coarse_class"]))
    class_df = pd.DataFrame(class_rows)
    save_grouped_bar_chart(
        labels=class_df["gt_coarse_class"].tolist(),
        series_map={
            "detector label vs GT": class_df["detector_label_acc"].tolist(),
            "Stage 4 actual": class_df["stage4_actual_acc"].tolist(),
            "Stage 3 ceiling": class_df["stage3_ceiling_acc"].tolist(),
        },
        out_path=out_dir / "class_accuracy_detector_actual_ceiling.png",
        title="Per-class accuracy: detector label vs Stage 4 actual vs Stage 3 ceiling",
        ylabel="Correct rate",
        ylim=(0.0, 1.03),
    )

    good_df["detector_equals_vlm"] = good_df["matched_pred_category_name"] == good_df["pred_vlm_coarse_class"]
    good_df["detector_equals_gt"] = good_df["matched_pred_category_name"] == good_df["gt_coarse_class"]
    good_df["vlm_equals_gt"] = good_df["pred_vlm_coarse_class"] == good_df["gt_coarse_class"]

    agreement_rows = {
        "good predicted crops": [
            safe_div(good_df["detector_equals_vlm"].sum(), len(good_df)),
            safe_div(good_df["detector_equals_gt"].sum(), len(good_df)),
            safe_div(good_df["vlm_equals_gt"].sum(), len(good_df)),
        ],
        "Stage 4 VLM errors": [
            safe_div((error_df["matched_pred_category_name"] == error_df["pred_vlm_coarse_class"]).sum(), len(error_df)),
            safe_div((error_df["matched_pred_category_name"] == error_df["gt_coarse_class"]).sum(), len(error_df)),
            safe_div((error_df["pred_vlm_coarse_class"] == error_df["gt_coarse_class"]).sum(), len(error_df)),
        ],
    }
    save_grouped_bar_chart(
        labels=["detector==VLM", "detector==GT", "VLM==GT"],
        series_map=agreement_rows,
        out_path=out_dir / "agreement_detector_vlm_gt.png",
        title="Agreement diagnostic: detector labels, VLM outputs, and GT",
        ylabel="Rate",
        ylim=(0.0, 1.03),
    )

    error_patterns = (
        error_df.groupby(["gt_coarse_class", "pred_vlm_coarse_class"]).size().reset_index(name="count").sort_values("count", ascending=False)
    )
    pattern_labels = [f"{row.gt_coarse_class} -> {row.pred_vlm_coarse_class}" for row in error_patterns.itertuples()]
    pattern_values = error_patterns["count"].tolist()
    main_error_pattern = "no good-crop VLM errors"
    main_error_count = 0
    if pattern_labels:
        main_error_pattern = pattern_labels[0]
        main_error_count = int(pattern_values[0])
    save_counts_bar_chart(
        labels=pattern_labels,
        values=pattern_values,
        out_path=out_dir / "top_error_patterns.png",
        title="Top coarse-class failure patterns on good predicted crops",
    )

    save_scatter(case_df[case_df["detector_found"] == True].copy(), out_dir / "iou_score_outcomes.png")

    leakage_lines = [
        f"good predicted crops: {len(good_df)}",
        f"detector == VLM on good crops: {good_df['detector_equals_vlm'].sum()} / {len(good_df)} = {safe_div(good_df['detector_equals_vlm'].sum(), len(good_df)):.4f}",
        f"detector == GT on good crops: {good_df['detector_equals_gt'].sum()} / {len(good_df)} = {safe_div(good_df['detector_equals_gt'].sum(), len(good_df)):.4f}",
        f"VLM == GT on good crops: {good_df['vlm_equals_gt'].sum()} / {len(good_df)} = {safe_div(good_df['vlm_equals_gt'].sum(), len(good_df)):.4f}",
    ]
    prompt_has_crop_path = False
    if prompt_template_path and prompt_template_path.exists():
        prompt_text = prompt_template_path.read_text(encoding="utf-8")
        prompt_has_crop_path = "{{crop_path}}" in prompt_text or "crop_path" in prompt_text
        leakage_lines.append(f"user prompt includes crop_path metadata: {prompt_has_crop_path}")
    crop_examples = pred_manifest_rows[:3]
    for idx, row in enumerate(crop_examples, start=1):
        leakage_lines.append(f"crop_path example {idx}: {row.get('crop_path', '')}")
    save_text_panel(
        lines=leakage_lines,
        out_path=out_dir / "leakage_diagnostic_panel.png",
        title="Potential metadata leakage diagnostic",
    )

    top_error_rows = (
        extra_drop_df.assign(pattern=extra_drop_df["gt_coarse_class"] + " -> " + extra_drop_df["pred_vlm_coarse_class"])
        .sort_values(["gt_coarse_class", "pred_vlm_coarse_class", "matched_pred_score"], ascending=[True, True, False])
        .head(args.top_gallery_errors)
    )
    top_hit_rows = (
        case_df[case_df["error_bucket"] == "correct_pipeline_hit"]
        .sort_values(["gt_coarse_class", "matched_pred_score"], ascending=[True, False])
        .groupby("gt_coarse_class", group_keys=False)
        .head(max(1, args.top_gallery_hits // 3))
        .head(args.top_gallery_hits)
    )

    gallery_errors_created = make_gallery(
        rows=top_error_rows.to_dict(orient="records"),
        pred_manifest_by_id=pred_manifest_by_id,
        archive_path=archive_path,
        out_path=out_dir / "gallery_extra_drop_errors.png",
        title="Stage 4 extra-drop errors",
        subtitle="Cases where Stage 3 ceiling was correct but Stage 4 actual failed",
    )
    gallery_hits_created = make_gallery(
        rows=top_hit_rows.to_dict(orient="records"),
        pred_manifest_by_id=pred_manifest_by_id,
        archive_path=archive_path,
        out_path=out_dir / "gallery_correct_hits.png",
        title="Stage 4 correct hits",
        subtitle="Representative crops where the full detector -> VLM path stayed correct",
    )

    top_error_csv_rows = top_error_rows[
        [
            "record_id",
            "gt_coarse_class",
            "matched_pred_record_id",
            "matched_pred_category_name",
            "pred_vlm_coarse_class",
            "match_iou",
            "matched_pred_score",
            "ceiling_coarse_class",
            "ceiling_correct",
        ]
    ].to_dict(orient="records")
    write_csv(
        out_dir / "top_error_cases.csv",
        top_error_csv_rows,
        [
            "record_id",
            "gt_coarse_class",
            "matched_pred_record_id",
            "matched_pred_category_name",
            "pred_vlm_coarse_class",
            "match_iou",
            "matched_pred_score",
            "ceiling_coarse_class",
            "ceiling_correct",
        ],
    )
    write_csv(
        out_dir / "class_accuracy_detector_actual_ceiling.csv",
        class_rows,
        ["gt_coarse_class", "n", "detector_label_acc", "stage4_actual_acc", "stage3_ceiling_acc"],
    )

    detector_equals_vlm = int(good_df["detector_equals_vlm"].sum())
    detector_equals_gt = int(good_df["detector_equals_gt"].sum())
    vlm_equals_gt = int(good_df["vlm_equals_gt"].sum())
    detector_equals_vlm_rate = safe_div(detector_equals_vlm, len(good_df))
    detector_equals_gt_rate = safe_div(detector_equals_gt, len(good_df))
    vlm_equals_gt_rate = safe_div(vlm_equals_gt, len(good_df))

    if prompt_has_crop_path:
        leakage_heading = "## Important caution: detector-label leakage is likely"
        leakage_takeaway = (
            "This pattern is too strong to ignore. It suggests the current Stage 4 run is not a clean test of visual reasoning alone."
        )
        bottom_line_tail = "- and the current Stage 4 setup appears to leak detector class metadata into the VLM prompt."
        next_step = (
            "So the next clean experiment should first remove `crop_path` leakage from the prompt, then rerun Stage 4 before doing more prompt tuning."
        )
    elif detector_equals_vlm_rate >= 0.85:
        leakage_heading = "## Important caution: detector-VLM agreement is still suspiciously high"
        leakage_takeaway = (
            "The prompt no longer exposes `crop_path`, but agreement remains high enough that the interface should still be checked carefully."
        )
        bottom_line_tail = "- and detector-VLM agreement is still suspiciously high enough to justify another interface check."
        next_step = "The next step is to audit remaining metadata fields before trusting this run as the final Stage 4 estimate."
    else:
        leakage_heading = "## Detector-VLM agreement diagnostic"
        leakage_takeaway = (
            "The extreme leakage pattern from the earlier run is no longer present. This run is a more trustworthy estimate of current `pred crop -> VLM` quality."
        )
        bottom_line_tail = "- and the earlier leakage signal is largely removed, so this run is the clean Stage 4 reference."
        next_step = "The next step is to analyze the remaining coarse-class failures and decide whether one narrow Stage 4 improvement pass is worthwhile."

    report_lines = [
        "# Stage 4 Visual Report",
        "",
        "## What these metrics mean",
        "",
        "- `detector_match_rate`: how often a GT object had a predicted box with IoU above the match threshold.",
        "- `good_crop_rate_among_matched`: among matched boxes, how often the predicted crop was still geometrically good enough for VLM.",
        "- `vlm_correct_rate_among_good_pred_crops`: on good crops only, how often the VLM predicted the correct coarse class.",
        "- `pipeline_correct_rate`: end-to-end coarse-class correctness on all GT objects.",
        "- `ceiling_correct_rate`: Stage 3 GT-crop -> VLM correctness; this is the upper bound if detector/crop noise disappeared.",
        "- `ceiling_vs_actual_gap`: how much performance is lost when moving from GT crops to predicted crops.",
        "",
        "## Current reading",
        "",
        f"- GT objects: `{counts.get('gt_objects_total', 0)}`",
        f"- Detector match rate: `{rates.get('detector_match_rate', 0.0):.4f}`",
        f"- Good crop rate among matched: `{rates.get('good_crop_rate_among_matched', 0.0):.4f}`",
        f"- VLM correct rate on good predicted crops: `{rates.get('vlm_correct_rate_among_good_pred_crops', 0.0):.4f}`",
        f"- Stage 4 actual pipeline correct rate: `{rates.get('pipeline_correct_rate', 0.0):.4f}`",
        f"- Stage 3 ceiling correct rate: `{rates.get('ceiling_correct_rate', 0.0):.4f}`",
        f"- Ceiling minus actual gap: `{rates.get('ceiling_vs_actual_gap', 0.0):.4f}`",
        "",
        "The geometry side is strong: detector boxes match every GT object and only one matched crop falls below the good-crop threshold.",
        "Most loss happens after that, inside the coarse-class decision on predicted crops.",
        "",
        "## Core charts",
        "",
        "![Pipeline funnel](pipeline_funnel_counts.png)",
        "",
        "![Pipeline rates](pipeline_rates.png)",
        "",
        "![Error buckets](error_buckets.png)",
        "",
        "![Coarse confusion](coarse_confusion_good_crops.png)",
        "",
        "![Coarse confusion normalized](coarse_confusion_good_crops_norm.png)",
        "",
        "![Per-class accuracy](class_accuracy_detector_actual_ceiling.png)",
        "",
        "![Outcome scatter](iou_score_outcomes.png)",
        "",
        "![Top error patterns](top_error_patterns.png)",
        "",
        "## What actually breaks",
        "",
        f"- `vlm_error_on_good_pred_crop`: `{error_counts.get('vlm_error_on_good_pred_crop', 0)}`",
        f"- `bad_crop_from_detector`: `{error_counts.get('bad_crop_from_detector', 0)}`",
        f"- `detector_miss`: `{error_counts.get('detector_miss', 0)}`",
        "",
        f"The largest coarse-class failure pattern is `{main_error_pattern}` with `{main_error_count}` cases.",
        "Visibility is not the main Stage 4 problem in this run; the drop is mostly in coarse-class separation on predicted crops.",
        "",
        leakage_heading,
        "",
        f"- On good predicted crops, `detector category == VLM coarse class` in `{detector_equals_vlm}/{len(good_df)}` cases = `{detector_equals_vlm_rate:.4f}`.",
        f"- On the same set, `detector category == GT` only in `{detector_equals_gt}/{len(good_df)}` cases = `{detector_equals_gt_rate:.4f}`.",
        f"- `VLM == GT` is `{vlm_equals_gt}/{len(good_df)}` = `{vlm_equals_gt_rate:.4f}`.",
        "",
        leakage_takeaway,
        "",
        "Why this is plausible:",
        "",
        f"- user prompt template contains `crop_path`: `{prompt_has_crop_path}`" if prompt_template_path else "- user prompt template was not provided for direct check.",
        f"- predicted crop paths encode detector class folders, e.g. `{pred_manifest_rows[0].get('crop_path', '')}`",
        "- If `crop_path` is passed into the prompt, the model can read detector class tokens such as `insulator_ok` or `defect_flashover`.",
        "",
        "![Leakage diagnostic](leakage_diagnostic_panel.png)",
        "",
        "![Agreement diagnostic](agreement_detector_vlm_gt.png)",
        "",
        "## Example crops",
        "",
    ]
    if gallery_errors_created:
        report_lines.extend(
            [
                "### Extra-drop errors",
                "These are cases where the Stage 3 ceiling was correct, but Stage 4 actual failed.",
                "",
                "![Extra-drop errors](gallery_extra_drop_errors.png)",
                "",
            ]
        )
    if gallery_hits_created:
        report_lines.extend(
            [
                "### Correct hits",
                "Representative cases where the full detector -> VLM path stayed correct.",
                "",
                "![Correct hits](gallery_correct_hits.png)",
                "",
            ]
        )

    report_lines.extend(
        [
            "## Bottom line",
            "",
            "This run is very useful diagnostically, but it should be interpreted with care.",
            "Right now it says:",
            "",
            "- localization/crop quality is strong,",
            "- end-to-end coarse accuracy is below the Stage 3 ceiling,",
            bottom_line_tail,
            "",
            next_step,
        ]
    )
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest_rows = []
    for path in sorted(out_dir.iterdir()):
        if path.is_file():
            manifest_rows.append({"name": path.name, "path": str(path.resolve())})
    write_csv(out_dir / "artifacts_manifest.csv", manifest_rows, ["name", "path"])

    print(f"Stage 4 visual package: {out_dir}")
    print(f"Report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
