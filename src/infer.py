#!/usr/bin/env python3
from __future__ import annotations

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

from src.datasets import ImageFolderDataset, inference_collate_fn
from src.model import build_detector
from src.utils.checkpoint import load_checkpoint
from src.utils.coco import predictions_to_coco, save_predictions_json
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.utils.vis import save_detection_visualizations


def resolve_device(device_cfg: str) -> str:
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def class_names_to_category_map(class_names_cfg) -> dict[int, int]:
    class_names = OmegaConf.to_container(class_names_cfg, resolve=True) if class_names_cfg is not None else {}
    mapping = {}
    for key in class_names.keys():
        category_id = int(key)
        mapping[category_id] = category_id
    return mapping


@hydra.main(version_base=None, config_path="configs", config_name="infer")
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

    dataset = ImageFolderDataset(
        input_dir=cfg.input_dir,
        image_size=int(cfg.image_size),
        resize=bool(cfg.resize),
        resize_mode=str(cfg.resize_mode),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=inference_collate_fn,
    )

    model = build_detector(num_classes=int(cfg.num_classes), pretrained=bool(cfg.pretrained)).to(device)
    load_checkpoint(cfg.checkpoint_path, model=model, optimizer=None, map_location=device)
    model.eval()

    label_to_category_id = class_names_to_category_map(cfg.get("class_names"))

    detections = []
    with torch.no_grad():
        for images, metas in tqdm(dataloader, desc="infer"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            detections.extend(
                predictions_to_coco(
                    predictions=outputs,
                    metas=metas,
                    label_to_category_id=label_to_category_id,
                    score_threshold=float(cfg.score_threshold),
                )
            )

    predictions_path = run_dir / "predictions.json"
    save_predictions_json(detections, predictions_path)

    save_detection_visualizations(
        image_id_to_path={k: str(v) for k, v in dataset.image_id_to_path.items()},
        detections=detections,
        output_dir=run_dir,
        class_names=OmegaConf.to_container(cfg.get("class_names", {}), resolve=True),
        score_threshold=float(cfg.score_threshold),
        max_images=int(cfg.vis_samples),
    )

    writer.close()

    print(f"Saved predictions: {predictions_path}")
    print(f"Saved visualizations: {run_dir / 'vis'}")


if __name__ == "__main__":
    main()
