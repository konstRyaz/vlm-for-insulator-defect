#!/usr/bin/env python3
"""Run leakage-free Stage 3/Stage 4 VLM backbone sweeps.

This is a skeleton/spec file for Codex to implement in the repo.
It should be copied to scripts/run_vlm_backbone_sweep.py and completed.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VLM backbone sweep.")
    parser.add_argument("--registry", default="configs/vlm_models_registry.yaml")
    parser.add_argument("--stage", choices=["stage3", "stage4"], default="stage3")
    parser.add_argument("--models", default="", help="Comma-separated model keys. Empty means all enabled for stage.")
    parser.add_argument("--base-config", default="configs/pipeline/stage3_vlm_gt_baseline.yaml")
    parser.add_argument("--ground-truth-jsonl", default="outputs/stage3_regrouped_v2/val/vlm_labels_v1_val_v2.annotated.jsonl")
    parser.add_argument("--output-root", default="outputs/vlm_backbone_sweeps")
    parser.add_argument("--report-dir", default="reports/vlm_comparison")
    parser.add_argument("--sweep-id", default="auto")
    parser.add_argument("--preflight-samples", type=int, default=1)
    parser.add_argument("--full-run", action="store_true", help="Run full test-split evaluation after successful preflight.")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object: {path}")
    return data


def run_cmd(cmd: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def main() -> None:
    args = parse_args()
    registry = load_yaml((ROOT / args.registry).resolve())
    models = registry.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("registry.models must be an object")

    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    if not selected:
        selected = [k for k, v in models.items() if isinstance(v, dict) and v.get(f"{args.stage}_enabled", False)]

    report_dir = (ROOT / args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    output_root = (ROOT / args.output_root / args.stage).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []

    for model_key in selected:
        cfg = models.get(model_key)
        if not isinstance(cfg, dict):
            raise KeyError(f"Unknown model key: {model_key}")

        # TODO: implement effective config generation.
        # TODO: support qwen_hf/generic_hf/internvl/llava/openai backends.
        # TODO: preflight via scripts/run_stage3_vlm_baseline.py --max-samples.
        # TODO: full run and eval if preflight passes and --full-run is set.
        row = {
            "model_key": model_key,
            "model_id": cfg.get("model_id", ""),
            "stage": args.stage,
            "preflight_ok": False,
            "full_ok": False,
            "error": "TODO implementation skeleton",
        }
        manifest_rows.append(row)

    manifest_path = output_root / "sweep_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({key for row in manifest_rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
