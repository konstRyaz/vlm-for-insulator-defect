#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


CATEGORY_TO_ID = {
    "insulator_ok": 1,
    "defect_flashover": 2,
    "defect_broken": 3,
    "unknown": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Stage 4 detector-assets bundle from Stage 3 regrouped GT labels."
    )
    parser.add_argument(
        "--gt-jsonl",
        type=str,
        default="outputs/stage3_regrouped_v2/val/vlm_labels_v1_val_v2.annotated.jsonl",
        help="Annotated Stage 3 JSONL used as the source of boxes and labels.",
    )
    parser.add_argument(
        "--image-roots",
        nargs="+",
        default=["data/raw/idid_mini/train/images"],
        help="Candidate directories that contain the original full-size images.",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default="outputs/train/detector_baseline/best.pth",
        help="Detector checkpoint to include in the bundle.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="kaggle_upload/idid-detector-assets",
        help="Output directory for the detector-assets bundle.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(payload)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_image_path(file_name: str, image_roots: list[Path]) -> Path:
    for root in image_roots:
        candidate = root / file_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found: {file_name}")


def main() -> None:
    args = parse_args()

    gt_jsonl = Path(args.gt_jsonl)
    image_roots = [Path(root) for root in args.image_roots]
    weights_path = Path(args.weights_path)
    output_dir = Path(args.output_dir)

    if not gt_jsonl.exists():
        raise FileNotFoundError(f"GT JSONL not found: {gt_jsonl}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    for root in image_roots:
        if not root.exists():
            raise FileNotFoundError(f"Image root not found: {root}")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    rows = load_jsonl(gt_jsonl)
    if not rows:
        raise ValueError(f"No rows found in {gt_jsonl}")

    image_names = sorted({str(row["image_path"]) for row in rows if row.get("image_path")})
    if not image_names:
        raise ValueError("No image_path values found in GT JSONL")

    image_id_by_name = {name: idx + 1 for idx, name in enumerate(image_names)}

    bundle_images_dir = output_dir / "data/processed/val/images"
    bundle_images_dir.mkdir(parents=True, exist_ok=True)

    images_payload: list[dict[str, Any]] = []
    name_to_source: dict[str, str] = {}
    for name in image_names:
        src_path = resolve_image_path(name, image_roots)
        dst_path = bundle_images_dir / name
        shutil.copy2(src_path, dst_path)
        with Image.open(src_path) as image:
            width, height = image.size
        images_payload.append(
            {
                "id": image_id_by_name[name],
                "file_name": name,
                "width": width,
                "height": height,
            }
        )
        name_to_source[name] = str(src_path)

    annotations_payload: list[dict[str, Any]] = []
    remapped_rows: list[dict[str, Any]] = []
    category_counter: Counter[str] = Counter()
    for ann_id, row in enumerate(rows, start=1):
        category_name = str(row.get("category_name", "")).strip()
        if category_name not in CATEGORY_TO_ID:
            raise ValueError(f"Unsupported category_name: {category_name!r}")

        image_name = str(row["image_path"])
        bbox = row.get("bbox_xywh")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Invalid bbox_xywh in record_id={row.get('record_id')}")
        x, y, w, h = [float(v) for v in bbox]
        if w <= 0 or h <= 0:
            raise ValueError(f"Non-positive bbox in record_id={row.get('record_id')}")

        annotations_payload.append(
            {
                "id": ann_id,
                "image_id": image_id_by_name[image_name],
                "category_id": CATEGORY_TO_ID[category_name],
                "bbox": [x, y, w, h],
                "area": float(w * h),
                "iscrowd": 0,
            }
        )

        remapped = dict(row)
        remapped["image_id"] = image_id_by_name[image_name]
        remapped_rows.append(remapped)
        category_counter[category_name] += 1

    coco_payload = {
        "images": images_payload,
        "annotations": annotations_payload,
        "categories": [
            {"id": category_id, "name": category_name}
            for category_name, category_id in CATEGORY_TO_ID.items()
        ],
    }
    write_json(output_dir / "data/processed/val/annotations.json", coco_payload)

    weights_out = output_dir / "outputs/train/detector_baseline/best.pth"
    weights_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights_path, weights_out)

    append_jsonl(output_dir / "analysis/stage4_gt_remapped.jsonl", remapped_rows)
    write_json(
        output_dir / "analysis/image_id_mapping.json",
        {
            "image_id_by_name": image_id_by_name,
            "source_image_path_by_name": name_to_source,
        },
    )
    write_json(
        output_dir / "summary.json",
        {
            "gt_jsonl": str(gt_jsonl),
            "weights_path": str(weights_path),
            "image_roots": [str(root) for root in image_roots],
            "num_images": len(images_payload),
            "num_annotations": len(annotations_payload),
            "category_distribution": dict(sorted(category_counter.items())),
            "artifacts": {
                "coco_json": str(output_dir / "data/processed/val/annotations.json"),
                "images_dir": str(bundle_images_dir),
                "weights_path": str(weights_out),
                "stage4_gt_remapped_jsonl": str(output_dir / "analysis/stage4_gt_remapped.jsonl"),
            },
        },
    )

    print(f"Built detector-assets bundle: {output_dir}")
    print(f"Images: {len(images_payload)} | Annotations: {len(annotations_payload)}")
    print(f"COCO: {output_dir / 'data/processed/val/annotations.json'}")
    print(f"Remapped GT: {output_dir / 'analysis/stage4_gt_remapped.jsonl'}")


if __name__ == "__main__":
    main()
