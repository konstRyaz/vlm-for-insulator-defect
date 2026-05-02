#!/usr/bin/env python3
from __future__ import annotations

"""
Bootstrap a draft `vlm_labels_v1` JSONL file from GT crop manifest.

Typical usage:
python scripts/bootstrap_vlm_labels_pilot.py \
  --manifest outputs/stage3_gt_crops/val/manifest.jsonl \
  --output outputs/stage3_gt_crops/val/vlm_labels_v1_pilot.jsonl \
  --limit 50
"""

import argparse
import json
from pathlib import Path
from typing import Any


ALLOWED_COARSE_CLASSES = {
    "insulator_ok",
    "defect_flashover",
    "defect_broken",
    "unknown",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create draft vlm_labels_v1 records from a crop manifest JSONL."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=str,
        help="Path to manifest JSONL produced by scripts/export_vlm_crops.py",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output JSONL path for draft vlm_labels_v1 records.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of records to export.",
    )
    parser.add_argument(
        "--include-categories",
        nargs="+",
        default=None,
        help="Optional category_name filter.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="gt",
        choices=["gt", "pred"],
        help="Source value to set when manifest record has no source (default: gt).",
    )
    parser.add_argument(
        "--label-version",
        type=str,
        default="vlm_labels_v1",
        help="Label version tag (default: vlm_labels_v1).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting output file if it exists.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{idx}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in {path}:{idx}, got {type(payload).__name__}")
            records.append(payload)
    return records


def coarse_class_from_category(category_name: str | None) -> str:
    if category_name is None:
        return "other"
    name = str(category_name).strip()
    return name if name in ALLOWED_COARSE_CLASSES else "other"


def class_default_text(coarse_class: str) -> tuple[str, str]:
    if coarse_class == "defect_flashover":
        return (
            "Visible signs consistent with flashover-like damage on the insulator surface.",
            "Selected insulator region shows flashover-like visible damage.",
        )
    if coarse_class == "defect_broken":
        return (
            "Visible structural discontinuity suggests a broken insulator segment.",
            "Selected insulator segment appears broken with visible structural discontinuity.",
        )
    if coarse_class == "insulator_ok":
        return (
            "No obvious defect is visible in the selected insulator region.",
            "No clear defect is observed in the selected insulator region.",
        )
    if coarse_class == "unknown":
        return (
            "Region appearance is unclear and cannot be confidently classified.",
            "One selected region is uncertain and should be reviewed manually.",
        )
    return (
        "Region appearance requires manual review before final class assignment.",
        "Selected region requires manual review before final reporting.",
    )


def build_label_record(
    manifest_record: dict[str, Any],
    default_source: str,
    label_version: str,
) -> dict[str, Any]:
    record_id = str(manifest_record.get("record_id", "")).strip()
    if not record_id:
        image_id_fallback = manifest_record.get("image_id", "unknown")
        box_id_fallback = manifest_record.get("box_id", "unknown")
        record_id = f"auto_img{image_id_fallback}_{box_id_fallback}"

    image_id = manifest_record.get("image_id")
    if image_id is None:
        image_id = "unknown"

    ann_id = manifest_record.get("ann_id")
    box_id = manifest_record.get("box_id")
    if box_id is None:
        box_id = f"ann{ann_id}" if ann_id is not None else "unknown_box"

    source = manifest_record.get("source", default_source)
    source = str(source) if source in {"gt", "pred"} else default_source

    split = str(manifest_record.get("split", "unspecified"))
    if split not in {"train", "val", "test", "unspecified"}:
        split = "unspecified"

    bbox_xywh = manifest_record.get("bbox_xywh")
    if not isinstance(bbox_xywh, list) or len(bbox_xywh) != 4:
        bbox_xywh = [0.0, 0.0, 0.0, 0.0]

    category_name = manifest_record.get("category_name")
    coarse_class = coarse_class_from_category(category_name if isinstance(category_name, str) else None)
    needs_review = coarse_class in {"unknown", "other"}
    visibility = "ambiguous" if needs_review else "clear"
    canonical_desc, report_snippet = class_default_text(coarse_class)

    record: dict[str, Any] = {
        "record_id": record_id,
        "image_id": image_id,
        "box_id": str(box_id),
        "source": source,
        "split": split,
        "bbox_xywh": [float(v) for v in bbox_xywh],
        "coarse_class": coarse_class,
        "visual_evidence_tags": [],
        "visibility": visibility,
        "needs_review": needs_review,
        "short_canonical_description": canonical_desc,
        "report_snippet": report_snippet,
        "crop_path": manifest_record.get("crop_path"),
        "image_path": manifest_record.get("image_path"),
        "score": manifest_record.get("score"),
        "category_name": str(category_name) if category_name is not None else coarse_class,
        "annotator_notes": "Draft created from manifest; refine manually.",
        "label_version": label_version,
    }
    return record


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path} (use --overwrite to replace)")
    if args.limit is not None and args.limit <= 0:
        raise ValueError(f"limit must be positive, got {args.limit}")

    include_categories = set(args.include_categories or [])

    manifest_records = load_jsonl(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for record in manifest_records:
            category_name = record.get("category_name")
            if include_categories and category_name not in include_categories:
                skipped += 1
                continue

            label_record = build_label_record(
                manifest_record=record,
                default_source=args.source,
                label_version=args.label_version,
            )
            out_f.write(json.dumps(label_record, ensure_ascii=False) + "\n")
            selected += 1

            if args.limit is not None and selected >= args.limit:
                break

    print(f"Draft records written: {selected}")
    print(f"Skipped by category filter: {skipped}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()

