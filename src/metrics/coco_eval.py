from __future__ import annotations

import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

COCO_STATS_NAMES = [
    "map_50_95",
    "map_50",
    "map_75",
    "ap_small",
    "ap_medium",
    "ap_large",
    "ar_1",
    "ar_10",
    "ar_100",
    "ar_small",
    "ar_medium",
    "ar_large",
]


def evaluate_coco(gt_json_path: str | Path, pred_json_path: str | Path) -> dict[str, float]:
    gt_path = Path(gt_json_path)
    pred_path = Path(pred_json_path)

    coco_gt = COCO(str(gt_path))

    if not pred_path.exists() or pred_path.stat().st_size == 0:
        return {name: 0.0 for name in COCO_STATS_NAMES}

    with pred_path.open("r", encoding="utf-8") as f:
        predictions = json.load(f)
    if not predictions:
        return {name: 0.0 for name in COCO_STATS_NAMES}

    coco_dt = coco_gt.loadRes(str(pred_path))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        name: float(value)
        for name, value in zip(COCO_STATS_NAMES, coco_eval.stats.tolist())
    }
