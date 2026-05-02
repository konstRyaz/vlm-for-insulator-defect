from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class CocoDetectionDataset(Dataset):
    def __init__(
        self,
        split_dir: str | Path,
        image_size: int = 512,
        resize: bool = True,
        resize_mode: str = "pad",
    ) -> None:
        self.split_dir = Path(split_dir)
        self.images_dir = self.split_dir / "images"
        self.annotation_path = self.split_dir / "annotations.json"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Missing images dir: {self.images_dir}")
        if not self.annotation_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {self.annotation_path}")

        with self.annotation_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.images = sorted(data.get("images", []), key=lambda x: int(x["id"]))
        self.annotations = data.get("annotations", [])

        categories = data.get("categories", [])
        if not categories:
            category_ids = sorted({int(ann["category_id"]) for ann in self.annotations})
            categories = [{"id": cid, "name": str(cid)} for cid in category_ids]

        categories = sorted(categories, key=lambda x: int(x["id"]))
        self.category_id_to_name = {int(cat["id"]): str(cat["name"]) for cat in categories}
        self.category_id_to_label = {
            category_id: idx + 1 for idx, category_id in enumerate(self.category_id_to_name.keys())
        }
        self.label_to_category_id = {
            label: category_id for category_id, label in self.category_id_to_label.items()
        }

        self.anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in self.annotations:
            self.anns_by_image[int(ann["image_id"])].append(ann)

        self.image_size = int(image_size)
        self.resize = bool(resize)
        self.resize_mode = str(resize_mode)
        if self.resize_mode not in {"pad", "stretch"}:
            raise ValueError("resize_mode must be one of: 'pad', 'stretch'")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        image_info = self.images[index]
        image_id = int(image_info["id"])
        image_path = self.images_dir / image_info["file_name"]

        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size

        anns = self.anns_by_image.get(image_id, [])
        boxes_xyxy = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            x1 = float(x)
            y1 = float(y)
            x2 = float(x + w)
            y2 = float(y + h)
            boxes_xyxy.append([x1, y1, x2, y2])

            category_id = int(ann["category_id"])
            labels.append(self.category_id_to_label[category_id])
            iscrowd.append(int(ann.get("iscrowd", 0)))
            areas.append(float(ann.get("area", w * h)))

        if boxes_xyxy:
            boxes_np = np.asarray(boxes_xyxy, dtype=np.float32)
        else:
            boxes_np = np.zeros((0, 4), dtype=np.float32)

        image, boxes_np, resize_meta = _resize_image_and_boxes(
            image=image,
            boxes_xyxy=boxes_np,
            image_size=self.image_size,
            resize=self.resize,
            resize_mode=self.resize_mode,
        )

        if boxes_np.size:
            widths = boxes_np[:, 2] - boxes_np[:, 0]
            heights = boxes_np[:, 3] - boxes_np[:, 1]
            areas = (widths * heights).tolist()

        boxes_tensor = torch.as_tensor(boxes_np, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        area_tensor = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd_tensor = torch.as_tensor(iscrowd, dtype=torch.int64)

        target: dict[str, Any] = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": area_tensor,
            "iscrowd": iscrowd_tensor,
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.float32),
            "resized_size": torch.tensor([resize_meta["resized_h"], resize_meta["resized_w"]], dtype=torch.float32),
            "scale_factors": torch.tensor([resize_meta["scale_x"], resize_meta["scale_y"]], dtype=torch.float32),
            "pad": torch.tensor([resize_meta["pad_x"], resize_meta["pad_y"]], dtype=torch.float32),
            "file_name": image_info["file_name"],
            "image_path": str(image_path),
        }

        image_tensor = F.to_tensor(image)
        return image_tensor, target

    @property
    def image_id_to_path(self) -> dict[int, Path]:
        return {
            int(info["id"]): self.images_dir / str(info["file_name"])
            for info in self.images
        }


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        input_dir: str | Path,
        image_size: int = 512,
        resize: bool = True,
        resize_mode: str = "pad",
    ) -> None:
        self.input_dir = Path(input_dir)
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Missing input dir: {self.input_dir}")

        self.image_paths = sorted(
            [p for p in self.input_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS]
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {self.input_dir}")

        self.image_size = int(image_size)
        self.resize = bool(resize)
        self.resize_mode = str(resize_mode)
        if self.resize_mode not in {"pad", "stretch"}:
            raise ValueError("resize_mode must be one of: 'pad', 'stretch'")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")

        resized_image, _, resize_meta = _resize_image_and_boxes(
            image=image,
            boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
            image_size=self.image_size,
            resize=self.resize,
            resize_mode=self.resize_mode,
        )

        image_tensor = F.to_tensor(resized_image)
        orig_w, orig_h = image.size

        image_id = index + 1
        meta = {
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.float32),
            "resized_size": torch.tensor([resize_meta["resized_h"], resize_meta["resized_w"]], dtype=torch.float32),
            "scale_factors": torch.tensor([resize_meta["scale_x"], resize_meta["scale_y"]], dtype=torch.float32),
            "pad": torch.tensor([resize_meta["pad_x"], resize_meta["pad_y"]], dtype=torch.float32),
            "file_name": image_path.name,
            "image_path": str(image_path),
        }

        return image_tensor, meta

    @property
    def image_id_to_path(self) -> dict[int, Path]:
        return {idx + 1: p for idx, p in enumerate(self.image_paths)}


def _resize_image_and_boxes(
    image: Image.Image,
    boxes_xyxy: np.ndarray,
    image_size: int,
    resize: bool,
    resize_mode: str,
) -> tuple[Image.Image, np.ndarray, dict[str, float]]:
    orig_w, orig_h = image.size

    if not resize:
        return image, boxes_xyxy, {
            "scale_x": 1.0,
            "scale_y": 1.0,
            "pad_x": 0.0,
            "pad_y": 0.0,
            "resized_w": float(orig_w),
            "resized_h": float(orig_h),
        }

    if resize_mode == "stretch":
        scale_x = image_size / max(orig_w, 1)
        scale_y = image_size / max(orig_h, 1)

        resized_image = image.resize((image_size, image_size), Image.BILINEAR)
        resized_boxes = boxes_xyxy.copy()
        if resized_boxes.size:
            resized_boxes[:, [0, 2]] *= scale_x
            resized_boxes[:, [1, 3]] *= scale_y

        return resized_image, resized_boxes, {
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
            "pad_x": 0.0,
            "pad_y": 0.0,
            "resized_w": float(image_size),
            "resized_h": float(image_size),
        }

    # letterbox-like resize: keep aspect ratio + pad to square
    scale = min(image_size / max(orig_w, 1), image_size / max(orig_h, 1))
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    resized = image.resize((new_w, new_h), Image.BILINEAR)
    padded = Image.new("RGB", (image_size, image_size), color=(0, 0, 0))

    pad_x = int((image_size - new_w) / 2)
    pad_y = int((image_size - new_h) / 2)
    padded.paste(resized, (pad_x, pad_y))

    resized_boxes = boxes_xyxy.copy()
    if resized_boxes.size:
        resized_boxes[:, [0, 2]] = resized_boxes[:, [0, 2]] * scale + pad_x
        resized_boxes[:, [1, 3]] = resized_boxes[:, [1, 3]] * scale + pad_y

    return padded, resized_boxes, {
        "scale_x": float(scale),
        "scale_y": float(scale),
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "resized_w": float(image_size),
        "resized_h": float(image_size),
    }
