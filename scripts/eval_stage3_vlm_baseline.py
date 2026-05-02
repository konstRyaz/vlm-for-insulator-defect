#!/usr/bin/env python3
from __future__ import annotations

"""
Evaluate Stage 3 baseline outputs against annotated GT JSONL.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any


COARSE_CLASS_LABELS = ["insulator_ok", "defect_flashover", "defect_broken", "unknown", "other"]
VISIBILITY_LABELS = ["clear", "partial", "ambiguous"]
MODEL_PREDICTED_CORE_FIELDS = [
    "coarse_class",
    "visual_evidence_tags",
    "visibility",
    "short_canonical_description_en",
    "report_snippet_en",
]
MODEL_OPTIONAL_DEBUG_FIELDS = [
    "annotator_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage 3 baseline run outputs.")
    parser.add_argument("--run-dir", required=True, type=str, help="Run directory from run_stage3_vlm_baseline.py")
    parser.add_argument("--ground-truth-jsonl", required=True, type=str, help="Annotated Stage 3 GT JSONL")
    parser.add_argument(
        "--output-dir",
        default=None,
        type=str,
        help="Evaluation output dir. Default: <run-dir>/eval",
    )
    return parser.parse_args()


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num) / float(den)


def macro_f1(gt: list[str], pred: list[str | None], labels: list[str]) -> tuple[float, dict[str, float]]:
    per_label: dict[str, float] = {}
    f1_values: list[float] = []

    for label in labels:
        tp = 0
        fp = 0
        fn = 0
        for gt_value, pred_value in zip(gt, pred):
            if gt_value == label and pred_value == label:
                tp += 1
            elif gt_value != label and pred_value == label:
                fp += 1
            elif gt_value == label and pred_value != label:
                fn += 1

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        per_label[label] = f1
        f1_values.append(f1)

    macro = safe_div(sum(f1_values), len(f1_values))
    return macro, per_label


def build_confusion(
    gt_values: list[str],
    pred_values: list[str | None],
    gt_labels: list[str],
    base_pred_labels: list[str],
) -> tuple[list[str], dict[str, dict[str, int]]]:
    pred_labels = list(base_pred_labels)
    missing_label = "__missing__"
    if missing_label not in pred_labels:
        pred_labels.append(missing_label)

    matrix: dict[str, dict[str, int]] = {
        gt_label: {pred_label: 0 for pred_label in pred_labels} for gt_label in gt_labels
    }
    for gt_value, pred_value in zip(gt_values, pred_values):
        pred_label = pred_value if isinstance(pred_value, str) else missing_label
        if pred_label not in pred_labels:
            pred_labels.append(pred_label)
            for row in matrix.values():
                row[pred_label] = 0
        if gt_value not in matrix:
            matrix[gt_value] = {label: 0 for label in pred_labels}
        matrix[gt_value][pred_label] += 1

    for gt_label in matrix:
        for pred_label in pred_labels:
            matrix[gt_label].setdefault(pred_label, 0)
    return pred_labels, matrix


def write_confusion_csv(
    path: Path,
    matrix: dict[str, dict[str, int]],
    gt_labels: list[str],
    pred_labels: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gt\\pred"] + pred_labels)
        for gt_label in gt_labels:
            row = [gt_label] + [matrix.get(gt_label, {}).get(pred_label, 0) for pred_label in pred_labels]
            writer.writerow(row)


def normalize_tags(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    out: set[str] = set()
    for item in value:
        if isinstance(item, str) and item.strip():
            out.add(item.strip())
    return out


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    gt_path = Path(args.ground_truth_jsonl).resolve()
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth JSONL not found: {gt_path}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (run_dir / "eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_results_path = run_dir / "sample_results.jsonl"
    parsed_predictions_path = run_dir / "parsed_predictions.jsonl"
    predictions_vlm_labels_v1_path = run_dir / "predictions_vlm_labels_v1.jsonl"

    if not sample_results_path.exists():
        raise FileNotFoundError(f"Missing run artifact: {sample_results_path}")
    if not parsed_predictions_path.exists():
        raise FileNotFoundError(f"Missing run artifact: {parsed_predictions_path}")
    if not predictions_vlm_labels_v1_path.exists():
        raise FileNotFoundError(f"Missing run artifact: {predictions_vlm_labels_v1_path}")

    gt_rows = load_jsonl(gt_path)
    sample_rows = load_jsonl(sample_results_path)
    parsed_rows = load_jsonl(parsed_predictions_path)
    pred_rows = load_jsonl(predictions_vlm_labels_v1_path)

    gt_by_id: dict[str, dict[str, Any]] = {}
    for row in gt_rows:
        rid = row.get("record_id")
        if isinstance(rid, str) and rid.strip():
            gt_by_id[rid] = row

    sample_by_id: dict[str, dict[str, Any]] = {}
    for row in sample_rows:
        rid = row.get("record_id")
        if isinstance(rid, str) and rid.strip():
            sample_by_id[rid] = row

    parsed_by_id: dict[str, dict[str, Any]] = {}
    for row in parsed_rows:
        rid = row.get("record_id")
        if isinstance(rid, str) and rid.strip():
            parsed_by_id[rid] = row

    pred_by_id: dict[str, dict[str, Any]] = {}
    for row in pred_rows:
        rid = row.get("record_id")
        if isinstance(rid, str) and rid.strip():
            pred_by_id[rid] = row

    gt_ids = set(gt_by_id.keys())
    sample_ids = set(sample_by_id.keys())
    pred_ids = set(pred_by_id.keys())
    parsed_ids = set(parsed_by_id.keys())

    eval_ids = sorted(gt_ids.intersection(sample_ids))
    gt_not_evaluated_ids = sorted(gt_ids.difference(sample_ids))
    sample_without_gt_ids = sorted(sample_ids.difference(gt_ids))

    counters = {
        "gt_total": len(gt_by_id),
        "evaluated_total": len(eval_ids),
        "gt_not_evaluated": len(gt_not_evaluated_ids),
        "sample_without_gt": len(sample_without_gt_ids),
        "parse_success": 0,
        "status_ok": 0,
        "schema_valid_true": 0,
        "missing_prediction_record": 0,
    }
    status_counts: dict[str, int] = {}

    coarse_gt: list[str] = []
    coarse_pred: list[str | None] = []
    visibility_gt: list[str] = []
    visibility_pred: list[str | None] = []
    needs_review_gt: list[bool] = []
    needs_review_pred: list[bool | None] = []

    tag_exact_matches = 0
    tag_compared = 0
    tag_jaccard_sum = 0.0

    review_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    field_presence_counts = {field: 0 for field in MODEL_PREDICTED_CORE_FIELDS}
    optional_debug_field_presence_counts = {field: 0 for field in MODEL_OPTIONAL_DEBUG_FIELDS}
    normalized_subset_valid = 0
    parsed_total = 0

    for rid in eval_ids:
        parsed = parsed_by_id.get(rid)
        if not isinstance(parsed, dict):
            continue
        parsed_total += 1
        normalized = parsed.get("normalized_prediction")
        errors = parsed.get("normalization_errors")
        if isinstance(normalized, dict):
            for field in MODEL_PREDICTED_CORE_FIELDS:
                if field in normalized:
                    field_presence_counts[field] += 1
            for field in MODEL_OPTIONAL_DEBUG_FIELDS:
                if field in normalized:
                    optional_debug_field_presence_counts[field] += 1
        if isinstance(errors, list) and len(errors) == 0 and isinstance(normalized, dict):
            normalized_subset_valid += 1

    for rid in eval_ids:
        gt = gt_by_id[rid]
        sample = sample_by_id[rid]
        pred = pred_by_id.get(rid)

        status = str(sample.get("status", "missing"))
        status_counts[status] = status_counts.get(status, 0) + 1

        parse_status = str(sample.get("parse_status", "missing"))
        parse_error = sample.get("parse_error")
        schema_valid = bool(sample.get("schema_valid", False))
        schema_errors = sample.get("schema_errors", [])

        if parse_status == "success":
            counters["parse_success"] += 1
        if status == "ok":
            counters["status_ok"] += 1
        if schema_valid:
            counters["schema_valid_true"] += 1
        if pred is None:
            counters["missing_prediction_record"] += 1

        gt_class = gt.get("coarse_class")
        pred_class = pred.get("coarse_class") if isinstance(pred, dict) else None
        gt_vis = gt.get("visibility")
        pred_vis = pred.get("visibility") if isinstance(pred, dict) else None

        gt_review = bool(gt.get("needs_review", False))
        pred_review = bool(pred.get("needs_review")) if isinstance(pred, dict) and "needs_review" in pred else None

        coarse_gt.append(str(gt_class) if isinstance(gt_class, str) else "other")
        coarse_pred.append(str(pred_class) if isinstance(pred_class, str) else None)

        visibility_gt.append(str(gt_vis) if isinstance(gt_vis, str) else "ambiguous")
        visibility_pred.append(str(pred_vis) if isinstance(pred_vis, str) else None)

        needs_review_gt.append(gt_review)
        needs_review_pred.append(pred_review if isinstance(pred_review, bool) else None)

        class_match = isinstance(pred_class, str) and isinstance(gt_class, str) and pred_class == gt_class
        visibility_match = isinstance(pred_vis, str) and isinstance(gt_vis, str) and pred_vis == gt_vis
        needs_review_match = isinstance(pred_review, bool) and pred_review == gt_review

        gt_tags = normalize_tags(gt.get("visual_evidence_tags"))
        pred_tags = normalize_tags(pred.get("visual_evidence_tags")) if isinstance(pred, dict) else set()
        if isinstance(pred, dict):
            tag_compared += 1
            if gt_tags == pred_tags:
                tag_exact_matches += 1
            union = gt_tags.union(pred_tags)
            if union:
                tag_jaccard_sum += len(gt_tags.intersection(pred_tags)) / len(union)
            else:
                tag_jaccard_sum += 1.0

        pred_ambiguous = pred_vis == "ambiguous"
        overuse_review = pred_review is True and gt_review is False

        review_row = {
            "record_id": rid,
            "image_id": gt.get("image_id"),
            "crop_path": gt.get("crop_path"),
            "status": status,
            "parse_status": parse_status,
            "schema_valid": schema_valid,
            "gt_coarse_class": gt_class,
            "pred_coarse_class": pred_class,
            "coarse_class_match": class_match,
            "gt_visibility": gt_vis,
            "pred_visibility": pred_vis,
            "visibility_match": visibility_match,
            "gt_needs_review": gt_review,
            "pred_needs_review": pred_review,
            "needs_review_match": needs_review_match,
            "pred_ambiguous": pred_ambiguous,
            "overuse_review": overuse_review,
            "parse_error": parse_error,
            "schema_errors": "|".join(schema_errors) if isinstance(schema_errors, list) else schema_errors,
        }
        review_rows.append(review_row)

        is_failure = (
            status != "ok"
            or parse_status != "success"
            or not schema_valid
            or not class_match
            or not visibility_match
            or overuse_review
        )
        if is_failure:
            failure_rows.append(
                {
                    "record_id": rid,
                    "status": status,
                    "parse_status": parse_status,
                    "schema_valid": schema_valid,
                    "gt_coarse_class": gt_class,
                    "pred_coarse_class": pred_class,
                    "gt_visibility": gt_vis,
                    "pred_visibility": pred_vis,
                    "gt_needs_review": gt_review,
                    "pred_needs_review": pred_review,
                    "parse_error": parse_error,
                    "schema_errors": schema_errors,
                }
            )

    total = counters["evaluated_total"]

    coarse_accuracy = safe_div(
        sum(1 for gt_value, pred_value in zip(coarse_gt, coarse_pred) if pred_value is not None and gt_value == pred_value),
        total,
    )
    coarse_macro_f1, coarse_f1_per_label = macro_f1(coarse_gt, coarse_pred, COARSE_CLASS_LABELS)

    visibility_accuracy = safe_div(
        sum(1 for gt_value, pred_value in zip(visibility_gt, visibility_pred) if pred_value is not None and gt_value == pred_value),
        total,
    )
    visibility_macro_f1, visibility_f1_per_label = macro_f1(visibility_gt, visibility_pred, VISIBILITY_LABELS)

    needs_review_accuracy = safe_div(
        sum(1 for gt_value, pred_value in zip(needs_review_gt, needs_review_pred) if isinstance(pred_value, bool) and gt_value == pred_value),
        total,
    )
    pred_ambiguous_count = sum(1 for value in visibility_pred if value == "ambiguous")
    gt_ambiguous_count = sum(1 for value in visibility_gt if value == "ambiguous")

    tag_exact_match_rate = safe_div(tag_exact_matches, tag_compared)
    tag_mean_jaccard = safe_div(tag_jaccard_sum, tag_compared)

    coarse_pred_labels, coarse_confusion = build_confusion(
        gt_values=coarse_gt,
        pred_values=coarse_pred,
        gt_labels=COARSE_CLASS_LABELS,
        base_pred_labels=COARSE_CLASS_LABELS,
    )
    visibility_pred_labels, visibility_confusion = build_confusion(
        gt_values=visibility_gt,
        pred_values=visibility_pred,
        gt_labels=VISIBILITY_LABELS,
        base_pred_labels=VISIBILITY_LABELS,
    )

    write_confusion_csv(
        path=output_dir / "confusion_coarse_class.csv",
        matrix=coarse_confusion,
        gt_labels=COARSE_CLASS_LABELS,
        pred_labels=coarse_pred_labels,
    )
    write_confusion_csv(
        path=output_dir / "confusion_visibility.csv",
        matrix=visibility_confusion,
        gt_labels=VISIBILITY_LABELS,
        pred_labels=visibility_pred_labels,
    )

    review_csv_path = output_dir / "review_table.csv"
    with review_csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(review_rows[0].keys()) if review_rows else [
            "record_id",
            "status",
            "parse_status",
            "schema_valid",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in review_rows:
            writer.writerow(row)

    write_jsonl(output_dir / "failures.jsonl", failure_rows)

    metrics = {
        "counts": {
            "gt_total": counters["gt_total"],
            "evaluated_total": total,
            "gt_not_evaluated": counters["gt_not_evaluated"],
            "sample_without_gt": counters["sample_without_gt"],
            "predictions_available_total": len(pred_by_id),
            "predictions_with_gt": len(pred_ids.intersection(gt_ids)),
            "parsed_with_gt": len(parsed_ids.intersection(gt_ids)),
            "missing_prediction_record": counters["missing_prediction_record"],
            "parse_success": counters["parse_success"],
            "status_ok": counters["status_ok"],
            "schema_valid_true": counters["schema_valid_true"],
            "parsed_total": parsed_total,
            "normalized_subset_valid": normalized_subset_valid,
            "failures_total": len(failure_rows),
            "status_counts": status_counts,
        },
        "rates": {
            "evaluated_coverage_over_gt": safe_div(total, counters["gt_total"]),
            "parse_success_rate": safe_div(counters["parse_success"], total),
            "status_ok_rate": safe_div(counters["status_ok"], total),
            "schema_valid_rate": safe_div(counters["schema_valid_true"], total),
            "coarse_class_accuracy": coarse_accuracy,
            "visibility_accuracy": visibility_accuracy,
            "needs_review_accuracy": needs_review_accuracy,
            "tag_exact_match_rate": tag_exact_match_rate,
            "tag_mean_jaccard": tag_mean_jaccard,
            "pred_ambiguous_rate": safe_div(pred_ambiguous_count, total),
            "gt_ambiguous_rate": safe_div(gt_ambiguous_count, total),
        },
        "f1": {
            "coarse_class_macro_f1": coarse_macro_f1,
            "coarse_class_f1_per_label": coarse_f1_per_label,
            "visibility_macro_f1": visibility_macro_f1,
            "visibility_f1_per_label": visibility_f1_per_label,
        },
        "field_presence_from_parsed_subset": {
            "counts": field_presence_counts,
            "rates": {field: safe_div(count, parsed_total) for field, count in field_presence_counts.items()},
        },
        "optional_debug_field_presence_from_parsed_subset": {
            "counts": optional_debug_field_presence_counts,
            "rates": {
                field: safe_div(count, parsed_total) for field, count in optional_debug_field_presence_counts.items()
            },
        },
        "limitations": {
            "visual_evidence_tags": "Open-ended labels; exact matching is strict. Jaccard is provided as a conservative overlap signal.",
        },
        "artifacts": {
            "review_table_csv": str(review_csv_path),
            "failures_jsonl": str(output_dir / "failures.jsonl"),
            "confusion_coarse_class_csv": str(output_dir / "confusion_coarse_class.csv"),
            "confusion_visibility_csv": str(output_dir / "confusion_visibility.csv"),
        },
    }
    write_json(output_dir / "metrics.json", metrics)

    summary_md = [
        "# Stage 3 Baseline Evaluation",
        "",
        f"- GT records: {counters['gt_total']}",
        f"- Evaluated records: {total}",
        f"- Evaluated coverage over GT: {metrics['rates']['evaluated_coverage_over_gt']:.4f}",
        f"- Parse success rate: {metrics['rates']['parse_success_rate']:.4f}",
        f"- Schema-valid rate: {metrics['rates']['schema_valid_rate']:.4f}",
        f"- Coarse-class accuracy: {metrics['rates']['coarse_class_accuracy']:.4f}",
        f"- Coarse-class macro-F1: {metrics['f1']['coarse_class_macro_f1']:.4f}",
        f"- Visibility accuracy: {metrics['rates']['visibility_accuracy']:.4f}",
        f"- Visibility macro-F1: {metrics['f1']['visibility_macro_f1']:.4f}",
        f"- Needs-review accuracy: {metrics['rates']['needs_review_accuracy']:.4f}",
        f"- Tag exact match: {metrics['rates']['tag_exact_match_rate']:.4f}",
        f"- Tag mean Jaccard: {metrics['rates']['tag_mean_jaccard']:.4f}",
        "",
        "## Artifacts",
        f"- metrics: `{output_dir / 'metrics.json'}`",
        f"- review table: `{review_csv_path}`",
        f"- failures: `{output_dir / 'failures.jsonl'}`",
        f"- confusion (coarse class): `{output_dir / 'confusion_coarse_class.csv'}`",
        f"- confusion (visibility): `{output_dir / 'confusion_visibility.csv'}`",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_md), encoding="utf-8")

    print(f"Eval output dir: {output_dir}")
    print(f"Metrics: {output_dir / 'metrics.json'}")
    print(f"Review table: {review_csv_path}")
    print(f"Failures: {output_dir / 'failures.jsonl'}")


if __name__ == "__main__":
    main()
