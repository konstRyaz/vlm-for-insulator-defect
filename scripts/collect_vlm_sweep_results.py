#!/usr/bin/env python3
"""Collect VLM sweep results into comparison CSV/Markdown.

Skeleton/spec for scripts/collect_vlm_sweep_results.py.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect VLM sweep results.")
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--report-dir", default="reports/vlm_comparison")
    parser.add_argument("--stage", choices=["stage3", "stage4"], default="stage3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # TODO: inspect run dirs and eval summary JSON/CSV.
    # TODO: normalize columns:
    # model_key, model_id, parse_success, schema_valid, accuracy, correct, total,
    # macro_f1_3class, per-class recalls, paths, error, verdict.

    out_csv = report_dir / f"{args.stage}_vlm_backbone_comparison.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model_key", "model_id", "stage", "verdict", "notes"])
        writer.writeheader()
    print(f"Wrote placeholder: {out_csv}")


if __name__ == "__main__":
    main()
