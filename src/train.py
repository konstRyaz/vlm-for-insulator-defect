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

from src.datasets import CocoDetectionDataset, detection_collate_fn
from src.metrics import evaluate_coco
from src.model import build_detector
from src.utils.checkpoint import save_checkpoint
from src.utils.coco import predictions_to_coco, save_predictions_json
from src.utils.init_utils import set_random_seed, set_worker_seed, setup_saving_and_logging


def resolve_device(device_cfg: str) -> str:
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def move_targets_to_device(targets, device: str):
    moved = []
    for target in targets:
        moved_target = {}
        for key, value in target.items():
            if torch.is_tensor(value):
                moved_target[key] = value.to(device)
            else:
                moved_target[key] = value
        moved.append(moved_target)
    return moved


def run_validation(model, dataloader, device, dataset, score_threshold, pred_path: Path):
    model.eval()
    detections = []

    with torch.no_grad():
        for images, metas in tqdm(dataloader, desc="val", leave=False):
            images = [img.to(device) for img in images]
            outputs = model(images)
            detections.extend(
                predictions_to_coco(
                    predictions=outputs,
                    metas=metas,
                    label_to_category_id=dataset.label_to_category_id,
                    score_threshold=float(score_threshold),
                )
            )

    save_predictions_json(detections, pred_path)
    metrics = evaluate_coco(dataset.annotation_path, pred_path)
    return metrics


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg):
    set_random_seed(int(cfg.seed))

    logger, run_dir = setup_saving_and_logging(cfg)
    logger.info("Run dir: %s", run_dir)

    project_config = OmegaConf.to_container(cfg, resolve=True)
    writer = instantiate(
        cfg.writer,
        logger=logger,
        project_config=project_config,
        run_name=cfg.run_name,
    )

    device = resolve_device(str(cfg.device))
    logger.info("Device: %s", device)

    train_dataset = CocoDetectionDataset(
        split_dir=cfg.train_dir,
        image_size=int(cfg.image_size),
        resize=bool(cfg.resize),
        resize_mode=str(cfg.resize_mode),
    )
    val_dataset = CocoDetectionDataset(
        split_dir=cfg.val_dir,
        image_size=int(cfg.image_size),
        resize=bool(cfg.resize),
        resize_mode=str(cfg.resize_mode),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        collate_fn=detection_collate_fn,
        worker_init_fn=set_worker_seed,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        collate_fn=detection_collate_fn,
        worker_init_fn=set_worker_seed,
    )

    model = build_detector(num_classes=int(cfg.num_classes), pretrained=bool(cfg.pretrained)).to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.learning_rate),
        momentum=0.9,
        weight_decay=float(cfg.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(cfg.lr_step_size),
        gamma=float(cfg.lr_gamma),
    )

    best_map = float("-inf")
    global_step = 0

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for images, targets in tqdm(train_loader, desc=f"train epoch {epoch}"):
            images = [img.to(device) for img in images]
            targets_device = move_targets_to_device(targets, device)

            loss_dict = model(images, targets_device)
            total_loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            epoch_loss += float(total_loss.item())
            epoch_steps += 1

            writer.set_step(global_step, mode="train")
            writer.add_scalar("loss_total", float(total_loss.item()))
            global_step += 1

        scheduler.step()

        avg_loss = epoch_loss / max(epoch_steps, 1)
        pred_path = run_dir / f"val_predictions_epoch_{epoch:03d}.json"
        val_metrics = run_validation(
            model=model,
            dataloader=val_loader,
            device=device,
            dataset=val_dataset,
            score_threshold=float(cfg.score_threshold),
            pred_path=pred_path,
        )

        writer.set_step(epoch, mode="val")
        writer.add_scalar("map_50_95", float(val_metrics["map_50_95"]))
        writer.add_scalar("ap_small", float(val_metrics["ap_small"]))
        writer.add_scalar("ar_small", float(val_metrics["ar_small"]))

        logger.info(
            "Epoch %d | loss=%.5f | mAP@[.5:.95]=%.4f | AP_small=%.4f | AR_small=%.4f",
            epoch,
            avg_loss,
            val_metrics["map_50_95"],
            val_metrics["ap_small"],
            val_metrics["ar_small"],
        )

        config_dict = OmegaConf.to_container(cfg, resolve=True)
        save_checkpoint(
            path=run_dir / "last.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric=max(best_map, float(val_metrics["map_50_95"])),
            config_dict=config_dict,
        )

        if epoch % int(cfg.save_every) == 0:
            save_checkpoint(
                path=run_dir / f"epoch_{epoch:03d}.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=max(best_map, float(val_metrics["map_50_95"])),
                config_dict=config_dict,
            )

        if float(val_metrics["map_50_95"]) >= best_map:
            best_map = float(val_metrics["map_50_95"])
            save_checkpoint(
                path=run_dir / "best.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_map,
                config_dict=config_dict,
            )

    writer.close()


if __name__ == "__main__":
    main()
