#!/usr/bin/env bash
set -euo pipefail

python scripts/make_toy_coco.py --out_dir data/raw/toy_coco
python scripts/prepare_data.py --raw_dir data/raw/toy_coco --out_dir data/processed --dataset coco
python src/train.py +experiment=detector_smoke epochs=1 image_size=320
python src/eval.py +experiment=detector_smoke
