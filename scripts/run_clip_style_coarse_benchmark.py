#!/usr/bin/env python3
"""Run CLIP/TL-CLIP/DINO-style coarse classifier benchmark.

Skeleton/spec for scripts/run_clip_style_coarse_benchmark.py.
This must stay separate from full generative VLM JSON reporting.
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLIP-style coarse benchmark.")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--registry", default="configs/vlm_models_registry.yaml")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--mode", choices=["zeroshot", "linear_probe"], default="linear_probe")
    parser.add_argument("--output-dir", default="outputs/clip_style_coarse")
    parser.add_argument("--report-dir", default="reports/vlm_comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO:
    # 1. load crops and labels without leaking labels into inputs;
    # 2. load model/preprocess;
    # 3. zeroshot ranking or feature extraction;
    # 4. train logistic regression on train only if mode=linear_probe;
    # 5. evaluate on val;
    # 6. write predictions.csv, summary.csv, confusion_matrix.csv.
    print("TODO skeleton: CLIP-style benchmark not implemented yet")


if __name__ == "__main__":
    main()
