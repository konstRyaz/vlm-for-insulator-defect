from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MatchPair:
    gt_index: int
    pred_index: int
    iou: float
    pred_score: float


def xyxy_from_xywh(bbox_xywh: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = bbox_xywh
    return x, y, x + w, y + h


def bbox_iou_xywh(a_xywh: list[float], b_xywh: list[float]) -> float:
    ax1, ay1, ax2, ay2 = xyxy_from_xywh(a_xywh)
    bx1, by1, bx2, by2 = xyxy_from_xywh(b_xywh)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def greedy_match_by_iou(
    gt_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    min_iou: float,
) -> list[MatchPair]:
    candidates: list[MatchPair] = []
    for gt_idx, gt_row in enumerate(gt_rows):
        gt_bbox = gt_row.get("bbox_xywh")
        if not isinstance(gt_bbox, list) or len(gt_bbox) != 4:
            continue
        try:
            gt_box = [float(v) for v in gt_bbox]
        except (TypeError, ValueError):
            continue

        for pred_idx, pred_row in enumerate(pred_rows):
            pred_bbox = pred_row.get("bbox_xywh")
            if not isinstance(pred_bbox, list) or len(pred_bbox) != 4:
                continue
            try:
                pred_box = [float(v) for v in pred_bbox]
            except (TypeError, ValueError):
                continue

            iou = bbox_iou_xywh(gt_box, pred_box)
            if iou < min_iou:
                continue

            pred_score_raw = pred_row.get("score", 0.0)
            try:
                pred_score = float(pred_score_raw)
            except (TypeError, ValueError):
                pred_score = 0.0
            candidates.append(MatchPair(gt_index=gt_idx, pred_index=pred_idx, iou=iou, pred_score=pred_score))

    candidates.sort(key=lambda pair: (-pair.iou, -pair.pred_score, pair.gt_index, pair.pred_index))

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[MatchPair] = []

    for pair in candidates:
        if pair.gt_index in used_gt:
            continue
        if pair.pred_index in used_pred:
            continue
        used_gt.add(pair.gt_index)
        used_pred.add(pair.pred_index)
        matches.append(pair)

    matches.sort(key=lambda pair: pair.gt_index)
    return matches
