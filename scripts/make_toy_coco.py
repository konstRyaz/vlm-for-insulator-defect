#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a toy COCO detection dataset")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--train_images", type=int, default=24)
    parser.add_argument("--val_images", type=int, default=8)
    parser.add_argument("--test_images", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_split(
    split_dir: Path,
    n_images: int,
    image_size: int,
    rng: random.Random,
    image_id_start: int,
    ann_id_start: int,
) -> tuple[int, int]:
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []

    image_id = image_id_start
    ann_id = ann_id_start

    for i in range(n_images):
        file_name = f"{split_dir.name}_{i:05d}.jpg"
        image_path = images_dir / file_name

        bg_color = (
            rng.randint(80, 180),
            rng.randint(80, 180),
            rng.randint(80, 180),
        )
        image = Image.new("RGB", (image_size, image_size), color=bg_color)
        draw = ImageDraw.Draw(image)

        n_boxes = rng.randint(1, 4)
        for _ in range(n_boxes):
            w = rng.randint(image_size // 20, image_size // 5)
            h = rng.randint(image_size // 20, image_size // 5)
            x = rng.randint(0, image_size - w - 1)
            y = rng.randint(0, image_size - h - 1)

            draw.rectangle([x, y, x + w, y + h], outline=(240, 20, 20), width=3)

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": float(w * h),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        image.save(image_path, quality=95)
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": image_size,
                "height": image_size,
            }
        )
        image_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "defect"}],
    }

    with (split_dir / "annotations.json").open("w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    return image_id, ann_id


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    image_id, ann_id = 1, 1
    split_sizes = {
        "train": args.train_images,
        "val": args.val_images,
        "test": args.test_images,
    }

    for split, n_images in split_sizes.items():
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        image_id, ann_id = make_split(
            split_dir=split_dir,
            n_images=n_images,
            image_size=args.image_size,
            rng=rng,
            image_id_start=image_id,
            ann_id_start=ann_id,
        )

    print(f"Toy COCO dataset created at: {out_dir}")


if __name__ == "__main__":
    main()
