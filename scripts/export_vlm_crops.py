#!/usr/bin/env python3
from __future__ import annotations

"""
Export bbox crops and JSONL manifest for Stage 3/4 pipelines.

Supports:
- GT boxes from COCO annotations (`--bbox-source gt`)
- Predicted boxes from detector predictions JSON (`--bbox-source pred`)

Examples:
python scripts/export_vlm_crops.py \
  --bbox-source gt \
  --coco-json data/processed/val/annotations.json \
  --images-dir data/processed/val/images \
  --output-dir outputs/stage3_gt_crops/val \
  --split val \
  --padding-ratio 0.15

python scripts/export_vlm_crops.py \
  --bbox-source pred \
  --coco-json data/processed/val/annotations.json \
  --images-dir data/processed/val/images \
  --predictions-json outputs/infer/detector_baseline/predictions.json \
  --score-threshold 0.30 \
  --max-detections-per-image 50 \
  --output-dir outputs/stage4/demo/02_pred_crops \
  --split val \
  --padding-ratio 0.15
"""

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export padded crops and build a JSONL manifest from GT or predicted bboxes."
    )
    parser.add_argument("--bbox-source", choices=["gt", "pred"], default="gt", help="Bounding-box source.")
    parser.add_argument("--coco-json", required=True, type=str, help="Path to COCO annotations JSON.")
    parser.add_argument("--images-dir", required=True, type=str, help="Directory containing source images.")
    parser.add_argument("--output-dir", required=True, type=str, help="Output directory for crops and manifests.")
    parser.add_argument("--split", required=True, type=str, help="Split name to write into manifest.")
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.15,
        help="Symmetric bbox padding ratio relative to bbox width/height (default: 0.15).",
    )
    parser.add_argument(
        "--include-categories",
        nargs="+",
        default=None,
        help="Optional category names to export. If omitted, all categories are included.",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default="manifest.jsonl",
        help="Manifest file name under output-dir (default: manifest.jsonl).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of exported crops after filtering.",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="summary.json",
        help="Summary file name under output-dir (default: summary.json).",
    )

    parser.add_argument(
        "--predictions-json",
        type=str,
        default=None,
        help="Required when --bbox-source pred. Path to detector predictions JSON (COCO detections list).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="For --bbox-source pred: keep predictions with score >= threshold.",
    )
    parser.add_argument(
        "--max-detections-per-image",
        type=int,
        default=None,
        help="For --bbox-source pred: keep top-K detections per image by score.",
    )

    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_category_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    return normalized or "unknown_category"


def as_float_bbox(raw_bbox: Any) -> list[float] | None:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None
    try:
        x, y, w, h = [float(v) for v in raw_bbox]
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return [x, y, w, h]


def xyxy_from_xywh(bbox_xywh: list[float]) -> list[float]:
    x, y, w, h = bbox_xywh
    return [x, y, x + w, y + h]


def compute_clipped_xyxy(
    bbox_xywh: list[float],
    image_w: int,
    image_h: int,
    padding_ratio: float,
) -> list[int] | None:
    x1, y1, x2, y2 = xyxy_from_xywh(bbox_xywh)

    pad_x = bbox_xywh[2] * padding_ratio
    pad_y = bbox_xywh[3] * padding_ratio

    x1_p = int(math.floor(x1 - pad_x))
    y1_p = int(math.floor(y1 - pad_y))
    x2_p = int(math.ceil(x2 + pad_x))
    y2_p = int(math.ceil(y2 + pad_y))

    x1_c = max(0, min(image_w, x1_p))
    y1_c = max(0, min(image_h, y1_p))
    x2_c = max(0, min(image_w, x2_p))
    y2_c = max(0, min(image_h, y2_p))

    if x2_c <= x1_c or y2_c <= y1_c:
        return None
    return [x1_c, y1_c, x2_c, y2_c]


def clip_xyxy_to_image(xyxy: list[float], image_w: int, image_h: int) -> list[float] | None:
    x1, y1, x2, y2 = xyxy
    x1_c = float(max(0, min(image_w, x1)))
    y1_c = float(max(0, min(image_h, y1)))
    x2_c = float(max(0, min(image_w, x2)))
    y2_c = float(max(0, min(image_h, y2)))
    if x2_c <= x1_c or y2_c <= y1_c:
        return None
    return [x1_c, y1_c, x2_c, y2_c]


def normalize_image_path(file_name: str, images_dir: Path) -> Path:
    raw = Path(file_name)
    if raw.is_absolute():
        return raw
    return images_dir / raw


def to_posix_relative(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def build_gt_rows(
    annotations: list[dict[str, Any]],
    category_name_by_id: dict[int, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sorted_annotations = sorted(
        annotations,
        key=lambda a: (int(a.get("image_id", -1)), int(a.get("id", -1))),
    )
    for ann in sorted_annotations:
        try:
            category_id = int(ann.get("category_id"))
        except Exception:
            category_id = -1
        rows.append(
            {
                "image_id": ann.get("image_id"),
                "ann_id": ann.get("id"),
                "category_id": category_id,
                "bbox": ann.get("bbox"),
                "score": None,
                "area": ann.get("area", None),
                "iscrowd": ann.get("iscrowd", 0),
                "category_name": category_name_by_id.get(category_id, ""),
                "source_bbox_type": "gt",
            }
        )
    return rows


def build_pred_rows(
    predictions: list[dict[str, Any]],
    category_name_by_id: dict[int, str],
    score_threshold: float,
    max_detections_per_image: int | None,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for pred in predictions:
        try:
            image_id = int(pred.get("image_id"))
            category_id = int(pred.get("category_id"))
        except Exception:
            continue

        bbox = as_float_bbox(pred.get("bbox"))
        if bbox is None:
            continue

        score_raw = pred.get("score", None)
        score = None
        if score_raw is not None:
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                score = None
        if score is None:
            score = 0.0

        if score < score_threshold:
            continue

        grouped[image_id].append(
            {
                "image_id": image_id,
                "ann_id": None,
                "category_id": category_id,
                "bbox": bbox,
                "score": score,
                "area": None,
                "iscrowd": 0,
                "category_name": category_name_by_id.get(category_id, f"category_{category_id}"),
                "source_bbox_type": "pred",
            }
        )

    rows: list[dict[str, Any]] = []
    for image_id in sorted(grouped.keys()):
        items = grouped[image_id]
        items.sort(key=lambda p: (float(p.get("score", 0.0)), int(p.get("category_id", 0))), reverse=True)
        if max_detections_per_image is not None:
            items = items[:max_detections_per_image]
        rows.extend(items)

    return rows


def main() -> None:
    args = parse_args()

    coco_json = Path(args.coco_json)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    if not coco_json.exists():
        raise FileNotFoundError(f"COCO JSON not found: {coco_json}")
    if not images_dir.exists():
        raise FileNotFoundError(f"images-dir not found: {images_dir}")
    if args.padding_ratio < 0:
        raise ValueError(f"padding-ratio must be >= 0, got {args.padding_ratio}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError(f"limit must be positive, got {args.limit}")
    if args.max_detections_per_image is not None and args.max_detections_per_image <= 0:
        raise ValueError(f"max-detections-per-image must be positive, got {args.max_detections_per_image}")
    if args.score_threshold < 0:
        raise ValueError(f"score-threshold must be >= 0, got {args.score_threshold}")

    payload = load_json(coco_json)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {coco_json}, got {type(payload).__name__}")

    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = payload.get("categories", [])

    if not isinstance(images, list) or not isinstance(annotations, list) or not isinstance(categories, list):
        raise ValueError("COCO JSON must contain list fields: images, annotations, categories")

    image_by_id: dict[int, dict[str, Any]] = {}
    for image_info in images:
        try:
            image_id = int(image_info["id"])
        except Exception:
            continue
        image_by_id[image_id] = image_info

    category_name_by_id: dict[int, str] = {}
    for cat in categories:
        try:
            cat_id = int(cat["id"])
            cat_name = str(cat["name"])
        except Exception:
            continue
        category_name_by_id[cat_id] = cat_name

    include_categories = set(args.include_categories or [])
    if include_categories:
        known_names = set(category_name_by_id.values())
        unknown_requested = sorted(include_categories.difference(known_names))
        if unknown_requested:
            raise ValueError(
                "Unknown --include-categories names: "
                + ", ".join(unknown_requested)
                + f". Known categories: {sorted(known_names)}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    crops_root = output_dir / "crops" / str(args.split)
    crops_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / str(args.manifest_name)
    summary_path = output_dir / str(args.summary_name)

    counters: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()

    rows: list[dict[str, Any]]
    predictions_path: Path | None = None

    if args.bbox_source == "gt":
        rows = build_gt_rows(annotations=annotations, category_name_by_id=category_name_by_id)
    else:
        if not args.predictions_json:
            raise ValueError("--predictions-json is required when --bbox-source pred")
        predictions_path = Path(args.predictions_json)
        if not predictions_path.exists():
            raise FileNotFoundError(f"Predictions JSON not found: {predictions_path}")
        preds_payload = load_json(predictions_path)
        if not isinstance(preds_payload, list):
            raise ValueError(
                f"Expected JSON list in predictions file {predictions_path}, got {type(preds_payload).__name__}"
            )
        rows = build_pred_rows(
            predictions=[item for item in preds_payload if isinstance(item, dict)],
            category_name_by_id=category_name_by_id,
            score_threshold=float(args.score_threshold),
            max_detections_per_image=args.max_detections_per_image,
        )

    # deterministic local index for predicted boxes per image
    pred_local_counter: dict[int, int] = defaultdict(int)

    exported = 0
    with manifest_path.open("w", encoding="utf-8") as manifest_f:
        for row in rows:
            if args.limit is not None and exported >= args.limit:
                break

            counters["rows_seen"] += 1

            try:
                image_id = int(row.get("image_id"))
            except Exception:
                counters["skipped_invalid_image_id"] += 1
                continue

            image_info = image_by_id.get(image_id)
            if image_info is None:
                counters["skipped_missing_image_info"] += 1
                continue

            try:
                category_id = int(row.get("category_id"))
            except Exception:
                counters["skipped_invalid_category_id"] += 1
                continue

            category_name = str(row.get("category_name") or category_name_by_id.get(category_id, f"category_{category_id}"))
            if include_categories and category_name not in include_categories:
                counters["skipped_by_category_filter"] += 1
                continue

            bbox_xywh = as_float_bbox(row.get("bbox"))
            if bbox_xywh is None:
                counters["skipped_invalid_bbox"] += 1
                continue

            file_name = image_info.get("file_name")
            if not isinstance(file_name, str) or not file_name.strip():
                counters["skipped_missing_file_name"] += 1
                continue

            image_path = normalize_image_path(file_name=file_name, images_dir=images_dir)
            if not image_path.exists():
                counters["skipped_missing_image_file"] += 1
                continue

            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    image_w, image_h = image.size

                    bbox_xyxy_raw = xyxy_from_xywh(bbox_xywh)
                    bbox_xyxy_clipped = clip_xyxy_to_image(bbox_xyxy_raw, image_w=image_w, image_h=image_h)
                    if bbox_xyxy_clipped is None:
                        counters["skipped_invalid_bbox_after_clip"] += 1
                        continue

                    bbox_xyxy_for_crop = compute_clipped_xyxy(
                        bbox_xywh=bbox_xywh,
                        image_w=image_w,
                        image_h=image_h,
                        padding_ratio=float(args.padding_ratio),
                    )
                    if bbox_xyxy_for_crop is None:
                        counters["skipped_empty_crop_after_clip"] += 1
                        continue

                    x1, y1, x2, y2 = bbox_xyxy_for_crop
                    crop = image.crop((x1, y1, x2, y2))
                    crop_w, crop_h = crop.size

                    if args.bbox_source == "gt":
                        ann_id = row.get("ann_id")
                        ann_id_text = str(ann_id) if ann_id is not None else f"noid_{counters['rows_seen']}"
                        box_id = f"ann{ann_id_text}"
                        record_id = f"{args.split}_img{image_id}_{box_id}"
                    else:
                        pred_local_counter[image_id] += 1
                        pred_idx = pred_local_counter[image_id]
                        box_id = f"pred{pred_idx}"
                        record_id = f"{args.split}_img{image_id}_{box_id}"
                        ann_id = None

                    category_dir = crops_root / safe_category_name(category_name)
                    category_dir.mkdir(parents=True, exist_ok=True)
                    crop_name = f"{record_id}.jpg"
                    crop_path = category_dir / crop_name
                    crop.save(crop_path, format="JPEG", quality=95)

            except OSError:
                counters["skipped_unreadable_image"] += 1
                continue

            source = "gt" if args.bbox_source == "gt" else "pred"
            score_value = row.get("score", None)
            score_out = float(score_value) if isinstance(score_value, (int, float)) else None
            area = float(row.get("area")) if isinstance(row.get("area"), (int, float)) else float(bbox_xywh[2] * bbox_xywh[3])
            iscrowd = int(row.get("iscrowd", 0)) if args.bbox_source == "gt" else 0

            record: dict[str, Any] = {
                "record_id": record_id,
                "source": source,
                "split": str(args.split),
                "image_id": image_id,
                "image_path": to_posix_relative(image_path, images_dir),
                "ann_id": ann_id,
                "box_id": box_id,
                "bbox_xywh": [float(v) for v in bbox_xywh],
                "bbox_xyxy": [float(v) for v in bbox_xyxy_clipped],
                "category_id": category_id,
                "category_name": category_name,
                "crop_path": to_posix_relative(crop_path, output_dir),
                "padding_ratio": float(args.padding_ratio),
                "width": int(crop_w),
                "height": int(crop_h),
                "area": area,
                "iscrowd": iscrowd,
                "score": score_out,
                "label_version": "vlm_labels_v1",
                "vlm_labels_v1": None,
            }

            if args.bbox_source == "pred":
                record.update(
                    {
                        "source_bbox_type": "pred",
                        "parent_image_id": image_id,
                        "pred_box_id": box_id,
                        "pred_box_xyxy": [float(v) for v in bbox_xyxy_clipped],
                        "detector_score": score_out,
                        "detector_class_name": category_name,
                    }
                )

            manifest_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            counters["exported"] += 1
            exported += 1
            category_counter[category_name] += 1

    summary = {
        "input": {
            "bbox_source": args.bbox_source,
            "coco_json": str(coco_json),
            "images_dir": str(images_dir),
            "predictions_json": str(predictions_path) if predictions_path is not None else None,
            "output_dir": str(output_dir),
            "split": str(args.split),
            "padding_ratio": float(args.padding_ratio),
            "score_threshold": float(args.score_threshold),
            "max_detections_per_image": args.max_detections_per_image,
            "include_categories": sorted(include_categories) if include_categories else None,
            "limit": args.limit,
            "manifest_name": str(args.manifest_name),
        },
        "totals": {
            "exported_crops": int(counters.get("exported", 0)),
            "rows_seen": int(counters.get("rows_seen", 0)),
        },
        "by_category": {k: int(v) for k, v in sorted(category_counter.items())},
        "counters": {k: int(v) for k, v in sorted(counters.items())},
        "artifacts": {
            "manifest_jsonl": str(manifest_path),
            "summary_json": str(summary_path),
            "crops_root": str(crops_root),
        },
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Export complete: {counters.get('exported', 0)} crops")
    print(f"Manifest: {manifest_path}")
    print(f"Summary:  {summary_path}")


if __name__ == "__main__":
    main()
