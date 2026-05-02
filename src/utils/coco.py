from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _pair(value: Any, default: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    if value is None:
        return default
    arr = _to_numpy(value).reshape(-1)
    if arr.size >= 2:
        return float(arr[0]), float(arr[1])
    return default


def _image_id(value: Any) -> int:
    arr = _to_numpy(value).reshape(-1)
    return int(arr[0])


def _undo_resize_boxes(boxes_xyxy: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    if boxes_xyxy.size == 0:
        return boxes_xyxy

    scale_x, scale_y = _pair(meta.get("scale_factors"), (1.0, 1.0))
    pad_x, pad_y = _pair(meta.get("pad"), (0.0, 0.0))
    orig_h, orig_w = _pair(meta.get("orig_size"), (0.0, 0.0))

    boxes = boxes_xyxy.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / max(scale_x, 1e-8)
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / max(scale_y, 1e-8)

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0.0, max(orig_w - 1.0, 0.0))
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0.0, max(orig_h - 1.0, 0.0))
    return boxes


def _xyxy_to_xywh(box: np.ndarray) -> list[float]:
    x1, y1, x2, y2 = box.tolist()
    return [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))]


def predictions_to_coco(
    predictions: list[dict[str, torch.Tensor]],
    metas: list[dict[str, Any]],
    label_to_category_id: dict[int, int] | None = None,
    score_threshold: float = 0.05,
) -> list[dict[str, Any]]:
    label_to_category_id = label_to_category_id or {}

    results: list[dict[str, Any]] = []
    for pred, meta in zip(predictions, metas):
        boxes = _to_numpy(pred["boxes"]).astype(np.float32)
        scores = _to_numpy(pred["scores"]).astype(np.float32)
        labels = _to_numpy(pred["labels"]).astype(np.int64)

        boxes = _undo_resize_boxes(boxes, meta)
        image_id = _image_id(meta["image_id"])

        keep = np.where(scores >= score_threshold)[0]
        for idx in keep.tolist():
            label = int(labels[idx])
            category_id = int(label_to_category_id.get(label, label))
            results.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": _xyxy_to_xywh(boxes[idx]),
                    "score": float(scores[idx]),
                }
            )

    return results


def save_predictions_json(predictions: list[dict[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    return output_path
