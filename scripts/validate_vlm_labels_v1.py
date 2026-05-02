#!/usr/bin/env python3
from __future__ import annotations

"""
Validate `vlm_labels_v1` JSONL records against repo baseline constraints.

This validator is dependency-free (standard library only) and mirrors
the expected structure from `schemas/vlm_labels_v1.schema.json`.

Example:
python scripts/validate_vlm_labels_v1.py \
  --input outputs/stage3_gt_crops/val/vlm_labels_v1_pilot.jsonl
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "record_id",
    "image_id",
    "box_id",
    "source",
    "split",
    "bbox_xywh",
    "coarse_class",
    "visual_evidence_tags",
    "visibility",
    "needs_review",
    "short_canonical_description",
    "report_snippet",
}

ALLOWED_FIELDS = REQUIRED_FIELDS.union(
    {
        "short_canonical_description_ru",
        "short_canonical_description_en",
        "report_snippet_ru",
        "report_snippet_en",
        "crop_path",
        "image_path",
        "score",
        "category_name",
        "annotator_notes",
        "label_version",
    }
)

SOURCE_VALUES = {"gt", "pred"}
SPLIT_VALUES = {"train", "val", "test", "unspecified"}
COARSE_CLASS_VALUES = {"insulator_ok", "defect_flashover", "defect_broken", "unknown", "other"}
VISIBILITY_VALUES = {"clear", "partial", "ambiguous"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate vlm_labels_v1 JSONL records.")
    parser.add_argument("--input", required=True, type=str, help="Path to input JSONL file.")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first validation error.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
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
            rows.append((line_no, payload))
    return rows


def is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_record(record: dict[str, Any], line_no: int) -> list[str]:
    errors: list[str] = []

    missing = sorted(REQUIRED_FIELDS.difference(record.keys()))
    if missing:
        errors.append(f"line {line_no}: missing required fields: {missing}")

    extra = sorted(set(record.keys()).difference(ALLOWED_FIELDS))
    if extra:
        errors.append(f"line {line_no}: unknown fields not allowed by schema: {extra}")

    if "record_id" in record and not is_non_empty_string(record["record_id"]):
        errors.append(f"line {line_no}: record_id must be non-empty string")

    if "image_id" in record:
        image_id = record["image_id"]
        if not (
            isinstance(image_id, int)
            or (isinstance(image_id, str) and bool(image_id.strip()))
        ):
            errors.append(f"line {line_no}: image_id must be int or non-empty string")

    if "box_id" in record and not is_non_empty_string(record["box_id"]):
        errors.append(f"line {line_no}: box_id must be non-empty string")

    if "source" in record and record["source"] not in SOURCE_VALUES:
        errors.append(f"line {line_no}: source must be one of {sorted(SOURCE_VALUES)}")

    if "split" in record and record["split"] not in SPLIT_VALUES:
        errors.append(f"line {line_no}: split must be one of {sorted(SPLIT_VALUES)}")

    if "bbox_xywh" in record:
        bbox = record["bbox_xywh"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(f"line {line_no}: bbox_xywh must be a list of length 4")
        else:
            for idx, value in enumerate(bbox):
                if not is_number(value):
                    errors.append(f"line {line_no}: bbox_xywh[{idx}] must be a finite number")
                elif float(value) < 0:
                    errors.append(f"line {line_no}: bbox_xywh[{idx}] must be >= 0")

    if "coarse_class" in record and record["coarse_class"] not in COARSE_CLASS_VALUES:
        errors.append(f"line {line_no}: coarse_class must be one of {sorted(COARSE_CLASS_VALUES)}")

    if "visual_evidence_tags" in record:
        tags = record["visual_evidence_tags"]
        if not isinstance(tags, list):
            errors.append(f"line {line_no}: visual_evidence_tags must be a list")
        else:
            for idx, tag in enumerate(tags):
                if not is_non_empty_string(tag):
                    errors.append(f"line {line_no}: visual_evidence_tags[{idx}] must be non-empty string")

    if "visibility" in record and record["visibility"] not in VISIBILITY_VALUES:
        errors.append(f"line {line_no}: visibility must be one of {sorted(VISIBILITY_VALUES)}")

    if "needs_review" in record and not isinstance(record["needs_review"], bool):
        errors.append(f"line {line_no}: needs_review must be boolean")

    if "short_canonical_description" in record and not is_non_empty_string(record["short_canonical_description"]):
        errors.append(f"line {line_no}: short_canonical_description must be non-empty string")

    if (
        "short_canonical_description_ru" in record
        and record["short_canonical_description_ru"] is not None
        and not isinstance(record["short_canonical_description_ru"], str)
    ):
        errors.append(f"line {line_no}: short_canonical_description_ru must be string if present")

    if (
        "short_canonical_description_en" in record
        and record["short_canonical_description_en"] is not None
        and not isinstance(record["short_canonical_description_en"], str)
    ):
        errors.append(f"line {line_no}: short_canonical_description_en must be string if present")

    if "report_snippet" in record and not is_non_empty_string(record["report_snippet"]):
        errors.append(f"line {line_no}: report_snippet must be non-empty string")

    if (
        "report_snippet_ru" in record
        and record["report_snippet_ru"] is not None
        and not isinstance(record["report_snippet_ru"], str)
    ):
        errors.append(f"line {line_no}: report_snippet_ru must be string if present")

    if (
        "report_snippet_en" in record
        and record["report_snippet_en"] is not None
        and not isinstance(record["report_snippet_en"], str)
    ):
        errors.append(f"line {line_no}: report_snippet_en must be string if present")

    if "crop_path" in record and record["crop_path"] is not None and not isinstance(record["crop_path"], str):
        errors.append(f"line {line_no}: crop_path must be string if present")

    if "image_path" in record and record["image_path"] is not None and not isinstance(record["image_path"], str):
        errors.append(f"line {line_no}: image_path must be string if present")

    if "score" in record and record["score"] is not None and not is_number(record["score"]):
        errors.append(f"line {line_no}: score must be finite number or null")

    if "category_name" in record and record["category_name"] is not None and not isinstance(record["category_name"], str):
        errors.append(f"line {line_no}: category_name must be string if present")

    if "annotator_notes" in record and record["annotator_notes"] is not None and not isinstance(record["annotator_notes"], str):
        errors.append(f"line {line_no}: annotator_notes must be string if present")

    if "label_version" in record and record["label_version"] is not None and not isinstance(record["label_version"], str):
        errors.append(f"line {line_no}: label_version must be string if present")

    return errors


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows = load_jsonl(input_path)
    if not rows:
        print(f"No records found in {input_path}")
        sys.exit(1)

    all_errors: list[str] = []
    for line_no, record in rows:
        errors = validate_record(record=record, line_no=line_no)
        if errors:
            all_errors.extend(errors)
            if args.fail_fast:
                break

    if all_errors:
        print(f"Validation failed: {len(all_errors)} issue(s)")
        for msg in all_errors[:50]:
            print(f"- {msg}")
        if len(all_errors) > 50:
            print(f"... and {len(all_errors) - 50} more")
        sys.exit(1)

    print(f"Validation passed: {len(rows)} record(s)")


if __name__ == "__main__":
    main()
