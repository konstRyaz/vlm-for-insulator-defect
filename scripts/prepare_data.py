#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

SMALL_THRESH = 32 * 32
MEDIUM_THRESH = 96 * 96


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare detection dataset in COCO format")
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    return parser.parse_args()


def convert_voc(raw_dir: Path, out_dir: Path) -> None:
    raise NotImplementedError("VOC converter is a stub. Implement convert_voc() for your dataset.")


def convert_yolo(raw_dir: Path, out_dir: Path) -> None:
    raise NotImplementedError("YOLO converter is a stub. Implement convert_yolo() for your dataset.")


def convert_dataset(raw_dir: Path, out_dir: Path, dataset_name: str) -> None:
    raise NotImplementedError(
        f"Converter for dataset='{dataset_name}' is a stub. "
        f"Implement convert_{dataset_name}() and call it here."
    )


def load_coco(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def area_bucket(area: float) -> str:
    if area < SMALL_THRESH:
        return "small"
    if area <= MEDIUM_THRESH:
        return "medium"
    return "large"


def validate_and_collect_stats(split_dir: Path) -> dict[str, Any]:
    images_dir = split_dir / "images"
    ann_path = split_dir / "annotations.json"

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {ann_path}")

    coco = load_coco(ann_path)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    categories_dict = {int(cat["id"]): str(cat["name"]) for cat in categories}
    if not categories_dict:
        raise ValueError(f"No categories found in {ann_path}")

    image_by_id = {}
    for image in images:
        image_id = int(image["id"])
        if image_id in image_by_id:
            raise ValueError(f"Duplicate image_id={image_id} in {ann_path}")

        file_name = str(image["file_name"])
        width = float(image["width"])
        height = float(image["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size for image_id={image_id}: width={width}, height={height}")

        image_path = images_dir / file_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image file listed in COCO not found: {image_path}")

        image_by_id[image_id] = image

    ann_ids = set()
    class_counter: Counter[str] = Counter()
    area_counter: Counter[str] = Counter()

    eps = 1e-6
    for ann in annotations:
        ann_id = int(ann["id"])
        if ann_id in ann_ids:
            raise ValueError(f"Duplicate annotation id={ann_id} in {ann_path}")
        ann_ids.add(ann_id)

        image_id = int(ann["image_id"])
        if image_id not in image_by_id:
            raise ValueError(f"ann_id={ann_id}: image_id={image_id} is missing in images[]")

        category_id = int(ann["category_id"])
        if category_id not in categories_dict:
            raise ValueError(f"ann_id={ann_id}: category_id={category_id} is absent in categories[]")

        bbox = ann.get("bbox", None)
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"ann_id={ann_id}: bbox must be [x,y,w,h]")

        x, y, w, h = [float(v) for v in bbox]
        if w <= 0 or h <= 0:
            raise ValueError(f"ann_id={ann_id}: bbox must have positive width/height, got w={w}, h={h}")

        img_w = float(image_by_id[image_id]["width"])
        img_h = float(image_by_id[image_id]["height"])
        if x < -eps or y < -eps or (x + w) > (img_w + eps) or (y + h) > (img_h + eps):
            raise ValueError(
                f"ann_id={ann_id}: bbox out of bounds for image_id={image_id}. "
                f"bbox=[{x},{y},{w},{h}], image_size=[{img_w},{img_h}]"
            )

        area = float(ann.get("area", w * h))
        if area <= 0:
            raise ValueError(f"ann_id={ann_id}: area must be > 0, got area={area}")

        class_counter[categories_dict[category_id]] += 1
        area_counter[area_bucket(area)] += 1

    stats = {
        "split": split_dir.name,
        "num_images": len(images),
        "num_annotations": len(annotations),
        "class_distribution": dict(sorted(class_counter.items())),
        "bbox_area_distribution": {
            "small": int(area_counter.get("small", 0)),
            "medium": int(area_counter.get("medium", 0)),
            "large": int(area_counter.get("large", 0)),
        },
    }
    return stats


def process_coco(raw_dir: Path, out_dir: Path) -> None:
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    validation_report: dict[str, Any] = {
        "dataset": "coco",
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "splits": {},
        "status": "ok",
    }

    for split in ["train", "val", "test"]:
        split_raw = raw_dir / split
        if not split_raw.exists():
            if split == "test":
                continue
            raise FileNotFoundError(f"Required split is missing: {split_raw}")

        stats = validate_and_collect_stats(split_raw)
        save_json(stats, reports_dir / f"{split}_stats.json")

        split_out = out_dir / split
        shutil.copytree(split_raw, split_out, dirs_exist_ok=True)

        validation_report["splits"][split] = {
            "validated": True,
            "copied_to": str(split_out),
            "stats_file": str(reports_dir / f"{split}_stats.json"),
        }

    save_json(validation_report, reports_dir / "validation_report.json")


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    dataset_name = args.dataset.lower().strip()

    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir does not exist: {raw_dir}")

    if dataset_name == "coco":
        process_coco(raw_dir, out_dir)
    elif dataset_name == "voc":
        convert_voc(raw_dir, out_dir)
    elif dataset_name == "yolo":
        convert_yolo(raw_dir, out_dir)
    else:
        convert_dataset(raw_dir, out_dir, dataset_name)

    print(f"Data preparation finished. Processed data: {out_dir}")


if __name__ == "__main__":
    main()
