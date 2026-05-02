#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


CLASS_MAPPING: dict[tuple[str, str], str] = {
    ("No issues", "No issues"): "insulator_ok",
    ("glaze", "Flashover damage"): "defect_flashover",
    ("shell", "Broken"): "defect_broken",
    ("notbroken-notflashed", "notbroken-notflashed"): "unknown",
}

CATEGORY_NAMES = [
    "insulator_ok",
    "defect_flashover",
    "defect_broken",
    "unknown",
]


@dataclass(frozen=True)
class ImageSample:
    image_id: int
    file_name: str
    source_path: Path
    width: int
    height: int
    objects: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert IDID labels JSON to COCO train/val splits")
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Path to IDID labels JSON (e.g. labels_v1.2.json).",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory with source images referenced by filename in labels JSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for COCO train/val splits.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of images to place into val split (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic train/val split (default: 42).",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into output split directories (disabled by default).",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help="Optional path for conversion summary JSON. "
        "Default: <out-dir>/reports/conversion_summary.json",
    )
    return parser.parse_args()


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected top-level JSON list in {path}, got {type(payload).__name__}")
    return payload


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def aggregate_by_filename(
    items: list[dict[str, Any]],
    counters: Counter[str],
) -> dict[str, list[dict[str, Any]]]:
    by_filename: dict[str, list[dict[str, Any]]] = {}

    for item in items:
        if not isinstance(item, dict):
            counters["invalid_top_level_items"] += 1
            continue

        filename_raw = item.get("filename")
        if not isinstance(filename_raw, str) or not filename_raw.strip():
            counters["skipped_items_missing_filename"] += 1
            continue
        filename = filename_raw.strip()

        labels = item.get("Labels")
        if labels is None:
            counters["items_missing_labels"] += 1
            objects_raw = []
        elif not isinstance(labels, dict):
            counters["items_invalid_labels_type"] += 1
            objects_raw = []
        else:
            objects_raw = labels.get("objects", [])

        if not isinstance(objects_raw, list):
            counters["items_invalid_objects_type"] += 1
            continue

        valid_objects: list[dict[str, Any]] = []
        for obj in objects_raw:
            if isinstance(obj, dict):
                valid_objects.append(obj)
            else:
                counters["skipped_non_dict_objects"] += 1

        by_filename.setdefault(filename, []).extend(valid_objects)

    return by_filename


def build_image_samples(
    objects_by_filename: dict[str, list[dict[str, Any]]],
    images_dir: Path,
    counters: Counter[str],
) -> list[ImageSample]:
    samples: list[ImageSample] = []
    image_id = 1

    for filename in sorted(objects_by_filename):
        source_path = images_dir / filename
        objects = objects_by_filename[filename]

        if not source_path.exists():
            counters["skipped_images_missing_file"] += 1
            counters["skipped_objects_missing_image_file"] += len(objects)
            continue

        try:
            with Image.open(source_path) as img:
                width, height = img.size
        except Exception:
            counters["skipped_images_unreadable_file"] += 1
            counters["skipped_objects_unreadable_image_file"] += len(objects)
            continue

        if width <= 0 or height <= 0:
            counters["skipped_images_invalid_size"] += 1
            counters["skipped_objects_invalid_image_size"] += len(objects)
            continue

        samples.append(
            ImageSample(
                image_id=image_id,
                file_name=filename,
                source_path=source_path,
                width=int(width),
                height=int(height),
                objects=objects,
            )
        )
        image_id += 1

    return samples


def split_samples(
    samples: list[ImageSample],
    val_ratio: float,
    seed: int,
) -> tuple[list[ImageSample], list[ImageSample]]:
    if not (0.0 <= val_ratio <= 1.0):
        raise ValueError(f"val_ratio must be in [0, 1], got {val_ratio}")

    if not samples:
        return [], []

    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    if val_ratio == 0.0:
        n_val = 0
    elif val_ratio == 1.0:
        n_val = len(indices)
    else:
        n_val = int(round(len(indices) * val_ratio))
        if len(indices) >= 2:
            n_val = max(1, min(len(indices) - 1, n_val))

    val_index_set = set(indices[:n_val])
    train_samples = [samples[i] for i in range(len(samples)) if i not in val_index_set]
    val_samples = [samples[i] for i in range(len(samples)) if i in val_index_set]
    return train_samples, val_samples


def parse_bbox(raw_bbox: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None
    try:
        x, y, w, h = (float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3]))
    except (TypeError, ValueError):
        return None
    return x, y, w, h


def clamp_bbox_to_image(
    x: float,
    y: float,
    w: float,
    h: float,
    image_w: int,
    image_h: int,
) -> tuple[list[float] | None, bool]:
    x1 = max(0.0, min(float(image_w), x))
    y1 = max(0.0, min(float(image_h), y))
    x2 = max(0.0, min(float(image_w), x + w))
    y2 = max(0.0, min(float(image_h), y + h))

    clamped = (
        abs(x1 - x) > 1e-9
        or abs(y1 - y) > 1e-9
        or abs(x2 - (x + w)) > 1e-9
        or abs(y2 - (y + h)) > 1e-9
    )

    new_w = x2 - x1
    new_h = y2 - y1
    if new_w <= 0 or new_h <= 0:
        return None, clamped

    return [float(x1), float(y1), float(new_w), float(new_h)], clamped


def map_condition_to_category(
    obj: dict[str, Any],
    counters: Counter[str],
    unmapped_pairs: Counter[str],
) -> str | None:
    conditions = obj.get("conditions")
    if conditions is None:
        counters["skipped_objects_missing_conditions"] += 1
        return None

    if not isinstance(conditions, dict) or len(conditions) != 1:
        counters["skipped_objects_invalid_conditions_format"] += 1
        return None

    raw_pair = next(iter(conditions.items()))
    pair = (str(raw_pair[0]), str(raw_pair[1]))
    category_name = CLASS_MAPPING.get(pair)
    if category_name is None:
        counters["skipped_objects_unmapped_conditions"] += 1
        unmapped_pairs[f"{pair[0]} -> {pair[1]}"] += 1
        return None
    return category_name


def build_coco_categories() -> list[dict[str, Any]]:
    categories: list[dict[str, Any]] = []
    for idx, name in enumerate(CATEGORY_NAMES, start=1):
        categories.append({"id": idx, "name": name, "supercategory": "insulator"})
    return categories


def to_distribution(counter: Counter[str]) -> dict[str, int]:
    return {name: int(counter[name]) for name in sorted(counter)}


def convert_split(
    split_name: str,
    samples: list[ImageSample],
    out_dir: Path,
    copy_images: bool,
    categories: list[dict[str, Any]],
    category_id_by_name: dict[str, int],
    ann_id_start: int,
    global_counters: Counter[str],
    global_unmapped_pairs: Counter[str],
) -> tuple[int, dict[str, Any]]:
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    images_out_dir = split_dir / "images"
    if copy_images:
        images_out_dir.mkdir(parents=True, exist_ok=True)

    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    split_class_counter: Counter[str] = Counter()
    split_counters: Counter[str] = Counter()

    ann_id = ann_id_start

    for sample in samples:
        if copy_images:
            dst_image_path = images_out_dir / sample.file_name
            dst_image_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sample.source_path, dst_image_path)
            file_name_for_coco = sample.file_name
        else:
            file_name_for_coco = str(sample.source_path.resolve())

        coco_images.append(
            {
                "id": sample.image_id,
                "file_name": file_name_for_coco,
                "width": sample.width,
                "height": sample.height,
            }
        )
        split_counters["images_written"] += 1

        for obj in sample.objects:
            split_counters["objects_seen"] += 1

            category_name = map_condition_to_category(obj, split_counters, global_unmapped_pairs)
            if category_name is None:
                continue

            parsed_bbox = parse_bbox(obj.get("bbox"))
            if parsed_bbox is None:
                split_counters["skipped_objects_invalid_bbox_format"] += 1
                continue

            x, y, w, h = parsed_bbox
            if w <= 0 or h <= 0:
                split_counters["skipped_objects_non_positive_bbox"] += 1
                continue

            clamped_bbox, was_clamped = clamp_bbox_to_image(
                x=x,
                y=y,
                w=w,
                h=h,
                image_w=sample.width,
                image_h=sample.height,
            )
            if clamped_bbox is None:
                split_counters["skipped_objects_empty_bbox_after_clamp"] += 1
                continue

            if was_clamped:
                split_counters["clamped_bboxes"] += 1

            bbox_w = float(clamped_bbox[2])
            bbox_h = float(clamped_bbox[3])

            coco_annotations.append(
                {
                    "id": ann_id,
                    "image_id": sample.image_id,
                    "category_id": category_id_by_name[category_name],
                    "bbox": clamped_bbox,
                    "area": float(bbox_w * bbox_h),
                    "iscrowd": 0,
                }
            )
            split_class_counter[category_name] += 1
            split_counters["annotations_written"] += 1
            ann_id += 1

    coco_payload = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    save_json(coco_payload, split_dir / "annotations.json")

    for key, value in split_counters.items():
        global_counters[key] += value

    split_summary = {
        "num_images": len(coco_images),
        "num_annotations": len(coco_annotations),
        "class_distribution": to_distribution(split_class_counter),
        "counters": to_distribution(split_counters),
    }
    return ann_id, split_summary


def main() -> None:
    args = parse_args()

    input_json = Path(args.input_json)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    summary_path = (
        Path(args.summary_path)
        if args.summary_path is not None
        else (out_dir / "reports" / "conversion_summary.json")
    )

    if not input_json.exists():
        raise FileNotFoundError(f"Input JSON does not exist: {input_json}")
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir does not exist: {images_dir}")

    raw_items = load_json_list(input_json)

    parse_counters: Counter[str] = Counter()
    parse_counters["top_level_items"] = len(raw_items)

    objects_by_filename = aggregate_by_filename(raw_items, counters=parse_counters)
    parse_counters["unique_filenames_in_json"] = len(objects_by_filename)
    parse_counters["objects_after_json_parse"] = sum(len(v) for v in objects_by_filename.values())

    samples = build_image_samples(
        objects_by_filename=objects_by_filename,
        images_dir=images_dir,
        counters=parse_counters,
    )
    parse_counters["images_loaded"] = len(samples)
    parse_counters["objects_attached_to_loaded_images"] = sum(len(sample.objects) for sample in samples)

    train_samples, val_samples = split_samples(samples, val_ratio=args.val_ratio, seed=args.seed)

    categories = build_coco_categories()
    category_id_by_name = {cat["name"]: int(cat["id"]) for cat in categories}

    out_dir.mkdir(parents=True, exist_ok=True)
    global_counters: Counter[str] = Counter()
    global_unmapped_pairs: Counter[str] = Counter()

    ann_id = 1
    ann_id, train_summary = convert_split(
        split_name="train",
        samples=train_samples,
        out_dir=out_dir,
        copy_images=args.copy_images,
        categories=categories,
        category_id_by_name=category_id_by_name,
        ann_id_start=ann_id,
        global_counters=global_counters,
        global_unmapped_pairs=global_unmapped_pairs,
    )
    ann_id, val_summary = convert_split(
        split_name="val",
        samples=val_samples,
        out_dir=out_dir,
        copy_images=args.copy_images,
        categories=categories,
        category_id_by_name=category_id_by_name,
        ann_id_start=ann_id,
        global_counters=global_counters,
        global_unmapped_pairs=global_unmapped_pairs,
    )

    overall_class_counter: Counter[str] = Counter()
    for name, value in train_summary["class_distribution"].items():
        overall_class_counter[name] += int(value)
    for name, value in val_summary["class_distribution"].items():
        overall_class_counter[name] += int(value)

    summary: dict[str, Any] = {
        "input": {
            "input_json": str(input_json),
            "images_dir": str(images_dir),
            "out_dir": str(out_dir),
            "val_ratio": float(args.val_ratio),
            "seed": int(args.seed),
            "copy_images": bool(args.copy_images),
        },
        "totals": {
            "annotations_written": int(global_counters.get("annotations_written", 0)),
            "images_written": int(global_counters.get("images_written", 0)),
            "objects_seen": int(global_counters.get("objects_seen", 0)),
            "clamped_bboxes": int(global_counters.get("clamped_bboxes", 0)),
            "class_distribution": to_distribution(overall_class_counter),
        },
        "parse_counters": to_distribution(parse_counters),
        "conversion_counters": to_distribution(global_counters),
        "unmapped_condition_pairs": to_distribution(global_unmapped_pairs),
        "splits": {
            "train": train_summary,
            "val": val_summary,
        },
    }

    save_json(summary, summary_path)

    print(f"COCO conversion complete: {out_dir}")
    print(f"  train images: {train_summary['num_images']}, annotations: {train_summary['num_annotations']}")
    print(f"  val images:   {val_summary['num_images']}, annotations: {val_summary['num_annotations']}")
    print(f"  clamped bboxes: {summary['totals']['clamped_bboxes']}")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
