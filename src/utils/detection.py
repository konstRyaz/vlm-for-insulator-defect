from pathlib import Path

import torch
from tqdm.auto import tqdm


def move_targets_to_device(targets, device):
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


def collect_predictions(model, dataloader, device, dataset, desc="eval"):
    model.eval()
    all_predictions = []
    all_targets = []
    all_image_paths = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc=desc):
            images = [image.to(device) for image in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                prediction = {
                    key: value.detach().cpu() if torch.is_tensor(value) else value
                    for key, value in output.items()
                }
                all_predictions.append(prediction)

                target_cpu = {
                    key: value.detach().cpu() if torch.is_tensor(value) else value
                    for key, value in target.items()
                }
                all_targets.append(target_cpu)

                image_id = int(target_cpu["image_id"].item())
                all_image_paths.append(Path(dataset.get_image_path_by_id(image_id)))

    return all_predictions, all_targets, all_image_paths
