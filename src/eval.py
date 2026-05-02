#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets import CocoDetectionDataset, detection_collate_fn
from src.metrics import evaluate_coco
from src.model import build_detector
from src.utils.checkpoint import load_checkpoint
from src.utils.coco import predictions_to_coco, save_predictions_json
from src.utils.init_utils import set_random_seed, set_worker_seed, setup_saving_and_logging
from src.utils.vis import save_detection_visualizations


def resolve_device(device_cfg: str) -> str:
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg):
    set_random_seed(int(cfg.seed))

    logger, run_dir = setup_saving_and_logging(cfg)
    project_config = OmegaConf.to_container(cfg, resolve=True)
    writer = instantiate(
        cfg.writer,
        logger=logger,
        project_config=project_config,
        run_name=cfg.get("run_name", None),
    )

    device = resolve_device(str(cfg.device))

    test_path = Path(str(cfg.test_dir))
    split_dir = test_path if test_path.exists() and (test_path / "annotations.json").exists() else Path(str(cfg.val_dir))

    dataset = CocoDetectionDataset(
        split_dir=split_dir,
        image_size=int(cfg.image_size),
        resize=bool(cfg.resize),
        resize_mode=str(cfg.resize_mode),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        collate_fn=detection_collate_fn,
        worker_init_fn=set_worker_seed,
    )

    model = build_detector(num_classes=int(cfg.num_classes), pretrained=bool(cfg.pretrained)).to(device)
    load_checkpoint(cfg.checkpoint_path, model=model, optimizer=None, map_location=device)
    model.eval()

    detections = []
    with torch.no_grad():
        for images, metas in tqdm(dataloader, desc="eval"):
            images = [img.to(device) for img in images]
            outputs = model(images)
            detections.extend(
                predictions_to_coco(
                    predictions=outputs,
                    metas=metas,
                    label_to_category_id=dataset.label_to_category_id,
                    score_threshold=float(cfg.score_threshold),
                )
            )

    predictions_path = run_dir / "predictions.json"
    save_predictions_json(detections, predictions_path)

    metrics = evaluate_coco(dataset.annotation_path, predictions_path)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_detection_visualizations(
        image_id_to_path={k: str(v) for k, v in dataset.image_id_to_path.items()},
        detections=detections,
        output_dir=run_dir,
        class_names=OmegaConf.to_container(cfg.get("class_names", {}), resolve=True),
        score_threshold=float(cfg.score_threshold),
        max_images=int(cfg.vis_samples),
    )

    writer.set_step(0, mode="eval")
    for key, value in metrics.items():
        writer.add_scalar(key, float(value))
    writer.close()

    print(f"mAP@[.5:.95]: {metrics['map_50_95']:.4f}")
    print(f"AP_small: {metrics['ap_small']:.4f}")
    print(f"AR_small: {metrics['ar_small']:.4f}")


if __name__ == "__main__":
    main()
