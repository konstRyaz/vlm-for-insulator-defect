#!/usr/bin/env python3
from __future__ import annotations

"""
Evaluate Stage 4 detector->VLM pipeline against GT Stage 3 annotations.

Evaluation unit: one GT object (record) from annotated JSONL.
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage4.matching import bbox_iou_xywh, greedy_match_by_iou

ERROR_BUCKETS = [
    "detector_miss",
    "bad_crop_from_detector",
    "vlm_error_on_good_pred_crop",
    "routing_or_filtering_error",
    "correct_pipeline_hit",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage 4 detector->VLM run.")
    parser.add_argument("--gt-jsonl", required=True, type=str, help="Annotated GT Stage 3 JSONL.")
    parser.add_argument("--pred-manifest-jsonl", required=True, type=str, help="Predicted-crop manifest JSONL.")
    parser.add_argument("--pred-vlm-run-dir", required=True, type=str, help="Run dir from run_stage3_vlm_baseline.py on pred manifest.")
    parser.add_argument("--detector-predictions-json", required=True, type=str, help="Detector predictions JSON (COCO detections list).")
    parser.add_argument("--coco-json", default=None, type=str, help="Optional COCO annotations JSON for category names.")
    parser.add_argument(
        "--ceiling-run-dir",
        default=None,
        type=str,
        help="Optional Stage 3 GT-crop run dir with predictions_vlm_labels_v1.jsonl.",
    )
    parser.add_argument(
        "--ceiling-predictions-jsonl",
        default=None,
        type=str,
        help="Optional direct path to ceiling predictions JSONL (overrides --ceiling-run-dir).",
    )
    parser.add_argument("--match-iou-threshold", type=float, default=0.5, help="IoU threshold for GT<->pred match.")
    parser.add_argument(
        "--good-crop-iou-threshold",
        type=float,
        default=0.7,
        help="IoU threshold for good crop quality among matched boxes.",
    )
    parser.add_argument("--output-dir", required=True, type=str, help="Output directory for Stage 4 eval artifacts.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path}:{line_no}, got {type(payload).__name__}")
            rows.append(payload)
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num) / float(den)


def normalize_image_id(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        try:
            return str(int(value))
        except Exception:
            return str(value)
    text = str(value).strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except Exception:
        return text


def normalize_bbox_xywh(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x, y, w, h = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return [x, y, w, h]


def resolve_ceiling_predictions_path(args: argparse.Namespace) -> Path | None:
    if args.ceiling_predictions_jsonl:
        return Path(args.ceiling_predictions_jsonl).resolve()
    if args.ceiling_run_dir:
        return Path(args.ceiling_run_dir).resolve() / "predictions_vlm_labels_v1.jsonl"
    return None


def load_category_name_by_id(coco_json_path: Path | None) -> dict[int, str]:
    if coco_json_path is None:
        return {}
    if not coco_json_path.exists():
        raise FileNotFoundError(f"COCO JSON not found: {coco_json_path}")
    payload = load_json(coco_json_path)
    if not isinstance(payload, dict):
        return {}
    categories = payload.get("categories", [])
    if not isinstance(categories, list):
        return {}

    out: dict[int, str] = {}
    for category in categories:
        if not isinstance(category, dict):
            continue
        try:
            cat_id = int(category.get("id"))
            cat_name = str(category.get("name"))
        except Exception:
            continue
        out[cat_id] = cat_name
    return out


def build_detector_rows(
    detector_predictions: list[dict[str, Any]],
    category_name_by_id: dict[int, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pred in detector_predictions:
        try:
            image_id = normalize_image_id(pred.get("image_id"))
            category_id = int(pred.get("category_id"))
        except Exception:
            continue
        bbox = normalize_bbox_xywh(pred.get("bbox"))
        if bbox is None:
            continue
        score_raw = pred.get("score", 0.0)
        try:
            score = float(score_raw)
        except (TypeError, ValueError):
            score = 0.0

        rows.append(
            {
                "image_id_norm": image_id,
                "bbox_xywh": bbox,
                "score": score,
                "category_id": category_id,
                "category_name": category_name_by_id.get(category_id, f"category_{category_id}"),
            }
        )
    return rows


def main() -> None:
    args = parse_args()

    gt_jsonl = Path(args.gt_jsonl).resolve()
    pred_manifest_jsonl = Path(args.pred_manifest_jsonl).resolve()
    pred_vlm_run_dir = Path(args.pred_vlm_run_dir).resolve()
    detector_predictions_json = Path(args.detector_predictions_json).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not gt_jsonl.exists():
        raise FileNotFoundError(f"GT JSONL not found: {gt_jsonl}")
    if not pred_manifest_jsonl.exists():
        raise FileNotFoundError(f"Predicted manifest JSONL not found: {pred_manifest_jsonl}")
    if not pred_vlm_run_dir.exists():
        raise FileNotFoundError(f"Pred VLM run dir not found: {pred_vlm_run_dir}")
    if not detector_predictions_json.exists():
        raise FileNotFoundError(f"Detector predictions JSON not found: {detector_predictions_json}")

    if args.match_iou_threshold < 0 or args.match_iou_threshold > 1:
        raise ValueError(f"match-iou-threshold must be in [0,1], got {args.match_iou_threshold}")
    if args.good_crop_iou_threshold < 0 or args.good_crop_iou_threshold > 1:
        raise ValueError(f"good-crop-iou-threshold must be in [0,1], got {args.good_crop_iou_threshold}")
    if args.good_crop_iou_threshold < args.match_iou_threshold:
        raise ValueError(
            "good-crop-iou-threshold should be >= match-iou-threshold for meaningful bucketing."
        )

    pred_vlm_predictions_path = pred_vlm_run_dir / "predictions_vlm_labels_v1.jsonl"
    if not pred_vlm_predictions_path.exists():
        raise FileNotFoundError(f"Missing pred VLM predictions: {pred_vlm_predictions_path}")

    ceiling_predictions_path = resolve_ceiling_predictions_path(args)
    if ceiling_predictions_path is not None and not ceiling_predictions_path.exists():
        raise FileNotFoundError(f"Ceiling predictions not found: {ceiling_predictions_path}")

    coco_json_path = Path(args.coco_json).resolve() if isinstance(args.coco_json, str) and args.coco_json else None
    category_name_by_id = load_category_name_by_id(coco_json_path)

    gt_rows_all = load_jsonl(gt_jsonl)
    gt_rows: list[dict[str, Any]] = []
    for row in gt_rows_all:
        record_id = row.get("record_id")
        bbox = normalize_bbox_xywh(row.get("bbox_xywh"))
        image_id_norm = normalize_image_id(row.get("image_id"))
        if isinstance(record_id, str) and record_id.strip() and bbox is not None and image_id_norm:
            gt_rows.append(
                {
                    **row,
                    "record_id": record_id,
                    "image_id_norm": image_id_norm,
                    "bbox_xywh": bbox,
                }
            )

    pred_manifest_rows_all = load_jsonl(pred_manifest_jsonl)
    pred_manifest_rows: list[dict[str, Any]] = []
    for row in pred_manifest_rows_all:
        record_id = row.get("record_id")
        bbox = normalize_bbox_xywh(row.get("bbox_xywh"))
        image_id_norm = normalize_image_id(row.get("image_id") or row.get("parent_image_id"))
        if isinstance(record_id, str) and record_id.strip() and bbox is not None and image_id_norm:
            score_raw = row.get("score", row.get("detector_score", 0.0))
            try:
                score = float(score_raw) if score_raw is not None else 0.0
            except (TypeError, ValueError):
                score = 0.0
            pred_manifest_rows.append(
                {
                    **row,
                    "record_id": record_id,
                    "image_id_norm": image_id_norm,
                    "bbox_xywh": bbox,
                    "score": score,
                }
            )

    pred_vlm_rows = load_jsonl(pred_vlm_predictions_path)
    pred_vlm_by_record_id = {
        str(row["record_id"]): row
        for row in pred_vlm_rows
        if isinstance(row.get("record_id"), str) and row["record_id"].strip()
    }

    ceiling_by_gt_record_id: dict[str, dict[str, Any]] = {}
    if ceiling_predictions_path is not None:
        for row in load_jsonl(ceiling_predictions_path):
            rid = row.get("record_id")
            if isinstance(rid, str) and rid.strip():
                ceiling_by_gt_record_id[rid] = row

    detector_payload = load_json(detector_predictions_json)
    if not isinstance(detector_payload, list):
        raise ValueError(
            f"Expected detector predictions JSON list in {detector_predictions_json}, got {type(detector_payload).__name__}"
        )
    detector_rows = build_detector_rows(
        detector_predictions=[item for item in detector_payload if isinstance(item, dict)],
        category_name_by_id=category_name_by_id,
    )

    gt_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pred_manifest_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    detector_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in gt_rows:
        gt_by_image[row["image_id_norm"]].append(row)
    for row in pred_manifest_rows:
        pred_manifest_by_image[row["image_id_norm"]].append(row)
    for row in detector_rows:
        detector_by_image[row["image_id_norm"]].append(row)

    for image_id in pred_manifest_by_image:
        pred_manifest_by_image[image_id].sort(key=lambda r: (float(r.get("score", 0.0)), r.get("record_id", "")), reverse=True)

    case_rows: list[dict[str, Any]] = []
    error_counter: Counter[str] = Counter()

    detector_found_count = 0
    good_crop_count = 0
    vlm_correct_good_count = 0
    pipeline_correct_count = 0
    ceiling_correct_count = 0

    image_ids = sorted(gt_by_image.keys())
    for image_id in image_ids:
        image_gt_rows = gt_by_image.get(image_id, [])
        image_pred_rows = pred_manifest_by_image.get(image_id, [])
        image_detector_rows = detector_by_image.get(image_id, [])

        match_pairs = greedy_match_by_iou(
            gt_rows=image_gt_rows,
            pred_rows=image_pred_rows,
            min_iou=float(args.match_iou_threshold),
        )
        matched_by_gt_idx = {pair.gt_index: pair for pair in match_pairs}

        raw_has_candidate: dict[int, bool] = {}
        filtered_has_candidate: dict[int, bool] = {}

        for gt_idx, gt_row in enumerate(image_gt_rows):
            gt_bbox = gt_row["bbox_xywh"]
            raw_has_candidate[gt_idx] = any(
                bbox_iou_xywh(gt_bbox, det_row["bbox_xywh"]) >= float(args.match_iou_threshold)
                for det_row in image_detector_rows
            )
            filtered_has_candidate[gt_idx] = any(
                bbox_iou_xywh(gt_bbox, pred_row["bbox_xywh"]) >= float(args.match_iou_threshold)
                for pred_row in image_pred_rows
            )

        for gt_idx, gt_row in enumerate(image_gt_rows):
            gt_record_id = str(gt_row.get("record_id"))
            gt_coarse = gt_row.get("coarse_class")
            gt_visibility = gt_row.get("visibility")

            matched_pair = matched_by_gt_idx.get(gt_idx)
            detector_found = matched_pair is not None
            match_iou = float(matched_pair.iou) if matched_pair is not None else 0.0

            matched_pred_record_id = None
            matched_pred_score = None
            matched_pred_category = None
            pred_vlm_coarse = None
            pred_vlm_visibility = None
            vlm_correct_on_good = False
            is_good_crop = False

            if detector_found:
                detector_found_count += 1
                pred_row = image_pred_rows[matched_pair.pred_index]
                matched_pred_record_id = pred_row.get("record_id")
                matched_pred_score = pred_row.get("score")
                matched_pred_category = pred_row.get("category_name")
                if match_iou >= float(args.good_crop_iou_threshold):
                    is_good_crop = True
                    good_crop_count += 1
                    if isinstance(matched_pred_record_id, str):
                        pred_vlm = pred_vlm_by_record_id.get(matched_pred_record_id)
                    else:
                        pred_vlm = None
                    if isinstance(pred_vlm, dict):
                        pred_vlm_coarse = pred_vlm.get("coarse_class")
                        pred_vlm_visibility = pred_vlm.get("visibility")
                        if isinstance(pred_vlm_coarse, str) and isinstance(gt_coarse, str) and pred_vlm_coarse == gt_coarse:
                            vlm_correct_on_good = True
                            vlm_correct_good_count += 1

            if detector_found and is_good_crop and vlm_correct_on_good:
                error_bucket = "correct_pipeline_hit"
                pipeline_correct_count += 1
            elif not detector_found:
                if raw_has_candidate.get(gt_idx, False) and not filtered_has_candidate.get(gt_idx, False):
                    error_bucket = "routing_or_filtering_error"
                else:
                    error_bucket = "detector_miss"
            elif detector_found and not is_good_crop:
                error_bucket = "bad_crop_from_detector"
            else:
                error_bucket = "vlm_error_on_good_pred_crop"

            error_counter[error_bucket] += 1

            ceiling_coarse = None
            ceiling_correct = False
            ceiling_row = ceiling_by_gt_record_id.get(gt_record_id)
            if isinstance(ceiling_row, dict):
                ceiling_coarse = ceiling_row.get("coarse_class")
                if isinstance(ceiling_coarse, str) and isinstance(gt_coarse, str) and ceiling_coarse == gt_coarse:
                    ceiling_correct = True
                    ceiling_correct_count += 1

            case_rows.append(
                {
                    "record_id": gt_record_id,
                    "image_id": gt_row.get("image_id"),
                    "gt_coarse_class": gt_coarse,
                    "gt_visibility": gt_visibility,
                    "detector_found": detector_found,
                    "matched_pred_record_id": matched_pred_record_id,
                    "matched_pred_score": matched_pred_score,
                    "matched_pred_category_name": matched_pred_category,
                    "match_iou": match_iou,
                    "is_good_crop": is_good_crop,
                    "pred_vlm_coarse_class": pred_vlm_coarse,
                    "pred_vlm_visibility": pred_vlm_visibility,
                    "vlm_correct_on_good_crop": vlm_correct_on_good,
                    "raw_has_match_candidate": raw_has_candidate.get(gt_idx, False),
                    "filtered_has_match_candidate": filtered_has_candidate.get(gt_idx, False),
                    "error_bucket": error_bucket,
                    "ceiling_coarse_class": ceiling_coarse,
                    "ceiling_correct": ceiling_correct,
                }
            )

    total_gt = len(case_rows)

    ceiling_correct_rate = safe_div(ceiling_correct_count, total_gt)
    pipeline_correct_rate = safe_div(pipeline_correct_count, total_gt)

    stage4_metrics = {
        "counts": {
            "gt_objects_total": total_gt,
            "detector_found_total": detector_found_count,
            "good_crop_total": good_crop_count,
            "vlm_correct_on_good_crop_total": vlm_correct_good_count,
            "pipeline_correct_total": pipeline_correct_count,
            "ceiling_correct_total": ceiling_correct_count,
        },
        "rates": {
            "detector_match_rate": safe_div(detector_found_count, total_gt),
            "good_crop_rate_among_matched": safe_div(good_crop_count, detector_found_count),
            "vlm_correct_rate_among_good_pred_crops": safe_div(vlm_correct_good_count, good_crop_count),
            "pipeline_correct_rate": pipeline_correct_rate,
            "ceiling_correct_rate": ceiling_correct_rate,
            "ceiling_vs_actual_gap": ceiling_correct_rate - pipeline_correct_rate,
        },
        "thresholds": {
            "match_iou_threshold": float(args.match_iou_threshold),
            "good_crop_iou_threshold": float(args.good_crop_iou_threshold),
        },
    }

    error_breakdown = {
        "counts": {bucket: int(error_counter.get(bucket, 0)) for bucket in ERROR_BUCKETS},
        "rates": {bucket: safe_div(float(error_counter.get(bucket, 0)), float(total_gt)) for bucket in ERROR_BUCKETS},
    }

    ceiling_vs_actual = {
        "ceiling": {
            "source": str(ceiling_predictions_path) if ceiling_predictions_path is not None else None,
            "correct_rate": ceiling_correct_rate,
            "correct_total": ceiling_correct_count,
        },
        "actual_pred_crop_pipeline": {
            "correct_rate": pipeline_correct_rate,
            "correct_total": pipeline_correct_count,
        },
        "gap": {
            "absolute": ceiling_correct_rate - pipeline_correct_rate,
        },
        "error_bucket_rates": error_breakdown["rates"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    case_table_path = output_dir / "stage4_case_table.csv"
    with case_table_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(case_rows[0].keys()) if case_rows else ["record_id", "error_bucket"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in case_rows:
            writer.writerow(row)

    write_json(output_dir / "stage4_metrics.json", stage4_metrics)
    write_json(output_dir / "stage4_error_breakdown.json", error_breakdown)
    write_json(output_dir / "ceiling_vs_actual.json", ceiling_vs_actual)

    summary_lines = [
        "# Stage 4 Detector -> VLM Summary",
        "",
        f"- GT objects: {total_gt}",
        f"- Detector match rate: {stage4_metrics['rates']['detector_match_rate']:.4f}",
        f"- Good crop rate among matched: {stage4_metrics['rates']['good_crop_rate_among_matched']:.4f}",
        f"- VLM correct rate among good pred crops: {stage4_metrics['rates']['vlm_correct_rate_among_good_pred_crops']:.4f}",
        f"- Pipeline correct rate: {stage4_metrics['rates']['pipeline_correct_rate']:.4f}",
        f"- Ceiling correct rate: {stage4_metrics['rates']['ceiling_correct_rate']:.4f}",
        f"- Ceiling vs actual gap: {stage4_metrics['rates']['ceiling_vs_actual_gap']:.4f}",
        "",
        "## Error buckets",
    ]
    for bucket in ERROR_BUCKETS:
        summary_lines.append(
            f"- {bucket}: {error_breakdown['counts'][bucket]} ({error_breakdown['rates'][bucket]:.4f})"
        )

    summary_lines.extend(
        [
            "",
            "## Artifacts",
            f"- stage4_metrics.json: `{output_dir / 'stage4_metrics.json'}`",
            f"- stage4_error_breakdown.json: `{output_dir / 'stage4_error_breakdown.json'}`",
            f"- stage4_case_table.csv: `{case_table_path}`",
            f"- ceiling_vs_actual.json: `{output_dir / 'ceiling_vs_actual.json'}`",
        ]
    )

    (output_dir / "stage4_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Stage4 eval output: {output_dir}")
    print(f"Metrics: {output_dir / 'stage4_metrics.json'}")
    print(f"Error breakdown: {output_dir / 'stage4_error_breakdown.json'}")
    print(f"Case table: {case_table_path}")


if __name__ == "__main__":
    main()
