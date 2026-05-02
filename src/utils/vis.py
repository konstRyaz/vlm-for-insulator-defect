from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def save_detection_visualizations(
    image_id_to_path: dict[int, str | Path],
    detections: list[dict[str, Any]],
    output_dir: str | Path,
    class_names: dict[str, str] | dict[int, str] | None = None,
    score_threshold: float = 0.3,
    max_images: int = 8,
) -> Path:
    vis_dir = Path(output_dir) / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for det in detections:
        if float(det.get("score", 0.0)) >= score_threshold:
            grouped[int(det["image_id"])].append(det)

    image_ids = list(grouped.keys())
    if not image_ids:
        image_ids = list(image_id_to_path.keys())
    image_ids = image_ids[:max_images]

    class_names = class_names or {}

    for image_id in image_ids:
        image_path = Path(image_id_to_path[image_id])
        if not image_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")

        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(image)
        ax.axis("off")

        for det in grouped.get(image_id, []):
            x, y, w, h = det["bbox"]
            category_id = det["category_id"]
            score = float(det["score"])

            rect = patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
            ax.add_patch(rect)

            label = class_names.get(str(category_id), class_names.get(category_id, str(category_id)))
            ax.text(
                x,
                max(0.0, y - 2.0),
                f"{label}: {score:.2f}",
                color="black",
                fontsize=9,
                bbox={"facecolor": "lime", "alpha": 0.7, "pad": 1},
            )

        out_path = vis_dir / f"{image_path.stem}_pred.jpg"
        fig.tight_layout(pad=0)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

    return vis_dir
