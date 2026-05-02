.PHONY: install smoke toy

install:
	python -m pip install -r requirements.txt

smoke:
	python scripts/smoke_imports.py

toy:
	python scripts/make_toy_coco.py --out_dir data/raw/toy_coco
	python scripts/prepare_data.py --dataset coco --raw_dir data/raw/toy_coco --out_dir data/processed/toy_coco
	bash scripts/smoke_run_toy.sh
