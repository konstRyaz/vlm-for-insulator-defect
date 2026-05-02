#!/usr/bin/env python3
"""Build a focused Stage 4 error-review gallery.

The script does not change evaluation logic. It joins the Stage 4 case table,
pred-crop manifest, and VLM outputs into a compact HTML/Markdown review set.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-table", required=True, help="Stage 4 stage4_case_table.csv")
    parser.add_argument("--pred-manifest-jsonl", required=True, help="Predicted-crop manifest JSONL")
    parser.add_argument("--predictions-jsonl", required=True, help="VLM predictions_vlm_labels_v1.jsonl")
    parser.add_argument("--raw-responses-jsonl", default=None, help="Optional raw_responses.jsonl")
    parser.add_argument("--pred-crops-dir", required=True, help="Directory that contains crops/...")
    parser.add_argument("--out-dir", required=True, help="Output directory for review artifacts")
    parser.add_argument("--max-per-group", type=int, default=24, help="Maximum image cards per review group")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def short_text(value: Any, max_len: int = 260) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def filesystem_path(path: Path) -> Path:
    """Return a Windows long-path-safe Path when needed."""
    if os.name != "nt":
        return path
    resolved = path.resolve()
    text = str(resolved)
    if text.startswith("\\\\?\\"):
        return Path(text)
    return Path("\\\\?\\" + text)


def copy_crop(
    row: dict[str, Any],
    manifest_by_id: dict[str, dict[str, Any]],
    pred_crops_dir: Path,
    images_dir: Path,
) -> str:
    pred_id = str(row.get("matched_pred_record_id") or "")
    manifest = manifest_by_id.get(pred_id, {})
    crop_path = manifest.get("crop_path") or row.get("crop_path") or ""
    if not crop_path:
        return ""

    src = pred_crops_dir / crop_path
    src_fs = filesystem_path(src)
    if not src_fs.exists():
        return ""

    safe_name = f"{row.get('review_group', 'case')}__{row.get('record_id', pred_id)}__{pred_id}{src.suffix}"
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in safe_name)
    dst = images_dir / safe_name
    shutil.copy2(src_fs, dst)
    return f"images/{safe_name}"


def classify_group(row: dict[str, Any]) -> str:
    gt = row.get("gt_coarse_class", "")
    pred = row.get("pred_vlm_coarse_class", "")
    bucket = row.get("error_bucket", "")

    if bucket == "correct_pipeline_hit":
        return "correct_pipeline_hits"
    if bucket == "detector_miss":
        return "detector_miss"
    if bucket == "bad_crop_from_detector":
        return "bad_crop_from_detector"
    if gt == "insulator_ok" and pred == "defect_flashover":
        return "ok_to_flashover"
    if gt == "insulator_ok" and pred == "defect_broken":
        return "ok_to_broken"
    if gt == "defect_flashover" and pred == "defect_broken":
        return "flashover_to_broken"
    if gt == "defect_flashover" and pred in {"insulator_ok", "unknown", ""}:
        return "flashover_to_ok_or_unknown"
    if gt == "defect_broken" and pred != "defect_broken":
        return "broken_errors"
    if bool_value(row.get("ceiling_correct")) and not bool_value(row.get("vlm_correct_on_good_crop")):
        return "extra_drop_vs_ceiling"
    if bucket == "vlm_error_on_good_pred_crop":
        return "other_vlm_errors"
    return "other_cases"


GROUP_TITLES = {
    "ok_to_flashover": "OK predicted as flashover",
    "ok_to_broken": "OK predicted as broken",
    "flashover_to_broken": "Flashover predicted as broken",
    "flashover_to_ok_or_unknown": "Flashover predicted as OK/unknown",
    "broken_errors": "Broken-class errors",
    "extra_drop_vs_ceiling": "Stage 3 ceiling correct, Stage 4 failed",
    "bad_crop_from_detector": "Bad crop from detector",
    "detector_miss": "Detector misses",
    "other_vlm_errors": "Other VLM errors on good crops",
    "correct_pipeline_hits": "Correct pipeline hits",
    "other_cases": "Other cases",
}


GROUP_ORDER = [
    "ok_to_flashover",
    "ok_to_broken",
    "flashover_to_broken",
    "flashover_to_ok_or_unknown",
    "broken_errors",
    "extra_drop_vs_ceiling",
    "bad_crop_from_detector",
    "detector_miss",
    "other_vlm_errors",
    "correct_pipeline_hits",
    "other_cases",
]


def enrich_rows(
    cases: list[dict[str, str]],
    manifest_by_id: dict[str, dict[str, Any]],
    pred_by_id: dict[str, dict[str, Any]],
    raw_by_id: dict[str, dict[str, Any]],
    pred_crops_dir: Path,
    out_dir: Path,
) -> list[dict[str, Any]]:
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    enriched: list[dict[str, Any]] = []
    for case in cases:
        pred_id = str(case.get("matched_pred_record_id") or "")
        pred = pred_by_id.get(pred_id, {})
        manifest = manifest_by_id.get(pred_id, {})
        raw = raw_by_id.get(pred_id, {})

        row: dict[str, Any] = dict(case)
        row["review_group"] = classify_group(case)
        row["detector_score"] = case.get("matched_pred_score") or manifest.get("detector_score", "")
        row["detector_class"] = case.get("matched_pred_category_name") or manifest.get("detector_class_name", "")
        row["tags"] = "; ".join(pred.get("visual_evidence_tags") or [])
        row["description"] = pred.get("short_canonical_description_en") or pred.get("short_canonical_description") or ""
        row["snippet"] = pred.get("report_snippet_en") or pred.get("report_snippet") or ""
        row["raw_text"] = raw.get("raw_text", "")
        row["crop_relpath"] = manifest.get("crop_path", "")
        row["is_extra_drop_vs_ceiling"] = bool_value(row.get("ceiling_correct")) and not bool_value(
            row.get("vlm_correct_on_good_crop")
        )
        row["image_relpath"] = copy_crop(row, manifest_by_id, pred_crops_dir, images_dir)
        row["match_iou_float"] = float_value(row.get("match_iou"))
        row["detector_score_float"] = float_value(row.get("detector_score"))
        enriched.append(row)

    return enriched


def important_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(row: dict[str, Any]) -> tuple[int, float, float]:
        group_rank = GROUP_ORDER.index(row["review_group"]) if row["review_group"] in GROUP_ORDER else 999
        score = row.get("detector_score_float", 0.0)
        iou = row.get("match_iou_float", 0.0)
        return (group_rank, -score, -iou)

    return sorted(rows, key=sort_key)


def write_html_report(out_dir: Path, rows: list[dict[str, Any]], max_per_group: int) -> None:
    counts = Counter(row["review_group"] for row in rows)
    extra_drop_count = sum(1 for row in rows if row.get("is_extra_drop_vs_ceiling"))
    error_patterns = Counter(
        f"{row.get('gt_coarse_class')} -> {row.get('pred_vlm_coarse_class')}"
        for row in rows
        if row.get("error_bucket") == "vlm_error_on_good_pred_crop"
    )

    style = """
    body { font-family: Georgia, 'Times New Roman', serif; margin: 32px; background: #f5f1e8; color: #1f2722; }
    h1 { font-size: 34px; margin-bottom: 4px; }
    h2 { margin-top: 34px; border-bottom: 2px solid #2e4f3f; padding-bottom: 6px; }
    .lead { max-width: 980px; line-height: 1.55; }
    .summary { display: flex; flex-wrap: wrap; gap: 12px; margin: 22px 0; }
    .pill { background: #fffaf0; border: 1px solid #d8c9a3; border-radius: 999px; padding: 8px 14px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }
    .card { background: #fffaf0; border: 1px solid #d4c6a7; border-radius: 14px; overflow: hidden; box-shadow: 0 8px 20px rgba(47, 53, 45, .08); }
    .card img { width: 100%; height: 210px; object-fit: contain; background: #161815; display: block; }
    .meta { padding: 12px 14px 14px; font-size: 13px; line-height: 1.42; }
    .label { font-weight: 700; color: #294835; }
    .bad { color: #9b2f22; font-weight: 700; }
    .ok { color: #24613e; font-weight: 700; }
    table { border-collapse: collapse; background: #fffaf0; }
    th, td { padding: 8px 11px; border: 1px solid #d8c9a3; text-align: left; }
    th { background: #ece0c3; }
    code { background: #efe5cf; padding: 2px 5px; border-radius: 4px; }
    """

    lines = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>Stage 4 targeted error review</title>",
        f"<style>{style}</style>",
        "<h1>Stage 4 targeted error review</h1>",
        "<p class='lead'>Focused review of the clean v7f detector-to-VLM run. The goal is to see which mistakes are visually plausible, which look like prompt bias, and which need a different input strategy.</p>",
        "<div class='summary'>",
    ]
    for group in GROUP_ORDER:
        if counts.get(group, 0):
            lines.append(f"<div class='pill'><b>{html.escape(GROUP_TITLES[group])}</b>: {counts[group]}</div>")
    lines.append(f"<div class='pill'><b>Ceiling correct but Stage 4 failed</b>: {extra_drop_count}</div>")
    lines.append("</div>")

    lines.extend(["<h2>Main coarse error patterns</h2>", "<table><tr><th>Pattern</th><th>Count</th></tr>"])
    for pattern, count in error_patterns.most_common(12):
        lines.append(f"<tr><td><code>{html.escape(pattern)}</code></td><td>{count}</td></tr>")
    lines.append("</table>")

    for group in GROUP_ORDER:
        group_rows = [row for row in rows if row["review_group"] == group]
        if not group_rows:
            continue
        lines.append(f"<h2>{html.escape(GROUP_TITLES[group])} ({len(group_rows)})</h2>")
        lines.append("<div class='grid'>")
        for row in group_rows[:max_per_group]:
            img_rel = row.get("image_relpath") or ""
            img_html = f"<img src='{html.escape(img_rel)}' alt='crop'>" if img_rel else "<div class='card-missing'>missing crop</div>"
            actual_ok = bool_value(row.get("vlm_correct_on_good_crop")) or row.get("error_bucket") == "correct_pipeline_hit"
            status = "<span class='ok'>correct</span>" if actual_ok else "<span class='bad'>wrong</span>"
            drop_flag = "yes" if row.get("is_extra_drop_vs_ceiling") else "no"
            lines.extend(
                [
                    "<div class='card'>",
                    img_html,
                    "<div class='meta'>",
                    f"<div><span class='label'>GT:</span> {html.escape(str(row.get('gt_coarse_class', '')))} | <span class='label'>VLM:</span> {html.escape(str(row.get('pred_vlm_coarse_class', '')))} | {status}</div>",
                    f"<div><span class='label'>record:</span> <code>{html.escape(str(row.get('record_id', '')))}</code></div>",
                    f"<div><span class='label'>pred crop:</span> <code>{html.escape(str(row.get('matched_pred_record_id', '')))}</code></div>",
                    f"<div><span class='label'>detector:</span> {html.escape(str(row.get('detector_class', '')))} / score {float_value(row.get('detector_score')):.3f} / IoU {float_value(row.get('match_iou')):.3f}</div>",
                    f"<div><span class='label'>ceiling correct, Stage 4 failed:</span> {drop_flag}</div>",
                    f"<div><span class='label'>visibility:</span> GT {html.escape(str(row.get('gt_visibility', '')))} -> VLM {html.escape(str(row.get('pred_vlm_visibility', '')))}</div>",
                    f"<div><span class='label'>tags:</span> {html.escape(short_text(row.get('tags'), 180))}</div>",
                    f"<div><span class='label'>snippet:</span> {html.escape(short_text(row.get('snippet') or row.get('description'), 220))}</div>",
                    "</div></div>",
                ]
            )
        lines.append("</div>")

    lines.extend(
        [
            "<h2>How to use this review</h2>",
            "<p class='lead'>If OK crops frequently contain small dark structures that the model calls flashover, the next useful experiment is not another broad prompt sweep. It is a targeted input or policy ablation: context crop, two-view crop, or a narrow flashover evidence guard.</p>",
        ]
    )
    (out_dir / "targeted_error_review.html").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_report(out_dir: Path, rows: list[dict[str, Any]], max_per_group: int) -> None:
    counts = Counter(row["review_group"] for row in rows)
    extra_drop_count = sum(1 for row in rows if row.get("is_extra_drop_vs_ceiling"))
    error_patterns = Counter(
        f"{row.get('gt_coarse_class')} -> {row.get('pred_vlm_coarse_class')}"
        for row in rows
        if row.get("error_bucket") == "vlm_error_on_good_pred_crop"
    )

    lines = [
        "# Stage 4 Targeted Error Review",
        "",
        "Focused review of the clean v7f detector-to-VLM run.",
        "",
        "## Group Counts",
        "",
        "| Group | Count |",
        "| --- | ---: |",
    ]
    for group in GROUP_ORDER:
        if counts.get(group, 0):
            lines.append(f"| {GROUP_TITLES[group]} | {counts[group]} |")
    lines.append(f"| Ceiling correct but Stage 4 failed | {extra_drop_count} |")

    lines.extend(["", "## Main Coarse Error Patterns", "", "| Pattern | Count |", "| --- | ---: |"])
    for pattern, count in error_patterns.most_common(12):
        lines.append(f"| `{pattern}` | {count} |")

    for group in GROUP_ORDER:
        group_rows = [row for row in rows if row["review_group"] == group]
        if not group_rows:
            continue
        lines.extend(["", f"## {GROUP_TITLES[group]} ({len(group_rows)})", ""])
        for row in group_rows[:max_per_group]:
            img_rel = row.get("image_relpath") or ""
            if img_rel:
                lines.append(f"![{row.get('record_id')}]({img_rel})")
            lines.extend(
                [
                    f"- record: `{row.get('record_id')}`",
                    f"- GT -> VLM: `{row.get('gt_coarse_class')}` -> `{row.get('pred_vlm_coarse_class')}`",
                    f"- detector: `{row.get('detector_class')}`, score `{float_value(row.get('detector_score')):.3f}`, IoU `{float_value(row.get('match_iou')):.3f}`",
                    f"- ceiling correct, Stage 4 failed: `{row.get('is_extra_drop_vs_ceiling')}`",
                    f"- visibility: `{row.get('gt_visibility')}` -> `{row.get('pred_vlm_visibility')}`",
                    f"- tags: {short_text(row.get('tags'), 220)}",
                    f"- snippet: {short_text(row.get('snippet') or row.get('description'), 260)}",
                    "",
                ]
            )

    (out_dir / "targeted_error_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    case_table = Path(args.case_table)
    pred_manifest = Path(args.pred_manifest_jsonl)
    predictions_path = Path(args.predictions_jsonl)
    raw_path = Path(args.raw_responses_jsonl) if args.raw_responses_jsonl else None
    pred_crops_dir = Path(args.pred_crops_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = read_csv(case_table)
    manifest_rows = read_jsonl(pred_manifest)
    prediction_rows = read_jsonl(predictions_path)
    raw_rows = read_jsonl(raw_path) if raw_path and raw_path.exists() else []

    manifest_by_id = {str(row.get("record_id")): row for row in manifest_rows}
    pred_by_id = {str(row.get("record_id")): row for row in prediction_rows}
    raw_by_id = {str(row.get("record_id")): row for row in raw_rows}

    rows = enrich_rows(cases, manifest_by_id, pred_by_id, raw_by_id, pred_crops_dir, out_dir)
    rows = important_rows(rows)

    fieldnames = [
        "review_group",
        "record_id",
        "gt_coarse_class",
        "pred_vlm_coarse_class",
        "gt_visibility",
        "pred_vlm_visibility",
        "detector_class",
        "detector_score",
        "match_iou",
        "error_bucket",
        "ceiling_coarse_class",
        "ceiling_correct",
        "matched_pred_record_id",
        "crop_relpath",
        "image_relpath",
        "is_extra_drop_vs_ceiling",
        "tags",
        "snippet",
        "description",
    ]
    write_csv(out_dir / "targeted_error_review.csv", rows, fieldnames)

    pattern_rows = [
        {"pattern": pattern, "count": count}
        for pattern, count in Counter(
            f"{row.get('gt_coarse_class')} -> {row.get('pred_vlm_coarse_class')}"
            for row in rows
            if row.get("error_bucket") == "vlm_error_on_good_pred_crop"
        ).most_common()
    ]
    write_csv(out_dir / "error_pattern_summary.csv", pattern_rows, ["pattern", "count"])

    write_html_report(out_dir, rows, args.max_per_group)
    write_markdown_report(out_dir, rows, args.max_per_group)

    print(f"Review HTML: {out_dir / 'targeted_error_review.html'}")
    print(f"Review Markdown: {out_dir / 'targeted_error_review.md'}")
    print(f"Review table: {out_dir / 'targeted_error_review.csv'}")


if __name__ == "__main__":
    main()
