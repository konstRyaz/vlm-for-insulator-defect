#!/usr/bin/env python3
from __future__ import annotations

"""
Run Stage 4 baseline pipeline:
val images -> detector predictions -> predicted crops/manifest -> frozen Stage 3 VLM -> Stage 4 eval.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 4 detector->VLM baseline pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage4_detector_to_vlm_pred_val.yaml",
        help="Path to Stage 4 YAML config.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object in {path}, got {type(payload).__name__}")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def resolve_path(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty path string, got: {value!r}")
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def run_command(command: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write("COMMAND:\n")
        log_f.write(" ".join(command) + "\n\n")
        log_f.flush()

        process = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {process.returncode}: {' '.join(command)}\n"
            f"See log: {log_path}"
        )


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_yaml(config_path)

    stage4_cfg = cfg.get("stage4", {})
    detector_cfg = cfg.get("detector", {})
    crop_cfg = cfg.get("crop_export", {})
    vlm_cfg = cfg.get("vlm", {})
    analysis_cfg = cfg.get("analysis", {})

    run_name = str(stage4_cfg.get("run_name", "stage4_detector_to_vlm")).strip()
    split = str(stage4_cfg.get("split", "val")).strip() or "val"
    output_root = resolve_path(stage4_cfg.get("output_root", "outputs/stage4"))
    run_dir = output_root / run_name

    detector_dir = run_dir / "01_detector"
    pred_crops_dir = run_dir / "02_pred_crops"
    vlm_root_dir = run_dir / "03_vlm_pred"
    eval_dir = run_dir / "04_eval"
    compare_dir = run_dir / "05_compare"

    for directory in [detector_dir, pred_crops_dir, vlm_root_dir, eval_dir, compare_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    write_yaml(run_dir / "stage4_config_snapshot.yaml", cfg)

    detector_input_dir = resolve_path(detector_cfg.get("input_dir"))
    detector_config_path = resolve_path(detector_cfg.get("config_path", "src/configs/infer.yaml"))
    detector_weights = resolve_path(detector_cfg.get("weights_path"))
    detector_conf_threshold = float(detector_cfg.get("conf_threshold", 0.3))
    detector_vis_samples = int(detector_cfg.get("vis_samples", 8))
    detector_device = str(detector_cfg.get("device", "auto"))
    detector_experiment = str(detector_cfg.get("experiment", "detector_baseline")).strip() or "detector_baseline"

    if not detector_input_dir.exists():
        raise FileNotFoundError(f"Detector input_dir not found: {detector_input_dir}")
    if not detector_config_path.exists():
        raise FileNotFoundError(f"Detector config_path not found: {detector_config_path}")
    if not detector_weights.exists():
        raise FileNotFoundError(f"Detector weights not found: {detector_weights}")

    detector_cmd = [
        sys.executable,
        str((ROOT / "src/infer.py").resolve()),
        f"+experiment={detector_experiment}",
        f"input_dir={detector_input_dir.as_posix()}",
        f"output_dir={detector_dir.as_posix()}",
        f"checkpoint_path={detector_weights.as_posix()}",
        f"score_threshold={detector_conf_threshold}",
        f"vis_samples={detector_vis_samples}",
        f"device={detector_device}",
    ]
    run_command(detector_cmd, cwd=ROOT, log_path=detector_dir / "run_detector.log")

    detector_predictions_path = detector_dir / "predictions.json"
    if not detector_predictions_path.exists():
        raise FileNotFoundError(f"Detector predictions not found after infer: {detector_predictions_path}")

    coco_json = resolve_path(crop_cfg.get("coco_json"))
    images_dir = resolve_path(crop_cfg.get("images_dir"))
    padding_ratio = float(crop_cfg.get("padding_ratio", 0.15))
    manifest_name = str(crop_cfg.get("manifest_name", "pred_manifest.jsonl"))
    summary_name = str(crop_cfg.get("summary_name", "pred_manifest_summary.json"))
    include_categories = crop_cfg.get("include_categories", [])
    limit = crop_cfg.get("limit", None)

    if not coco_json.exists():
        raise FileNotFoundError(f"crop_export.coco_json not found: {coco_json}")
    if not images_dir.exists():
        raise FileNotFoundError(f"crop_export.images_dir not found: {images_dir}")

    export_cmd = [
        sys.executable,
        str((ROOT / "scripts/export_vlm_crops.py").resolve()),
        "--bbox-source",
        "pred",
        "--coco-json",
        str(coco_json),
        "--images-dir",
        str(images_dir),
        "--predictions-json",
        str(detector_predictions_path),
        "--output-dir",
        str(pred_crops_dir),
        "--split",
        split,
        "--padding-ratio",
        str(padding_ratio),
        "--manifest-name",
        manifest_name,
        "--summary-name",
        summary_name,
        "--score-threshold",
        str(detector_conf_threshold),
        "--max-detections-per-image",
        str(int(detector_cfg.get("max_detections_per_image", 100))),
    ]

    if isinstance(include_categories, list) and include_categories:
        export_cmd.extend(["--include-categories", *[str(item) for item in include_categories]])
    if limit is not None:
        export_cmd.extend(["--limit", str(int(limit))])

    run_command(export_cmd, cwd=ROOT, log_path=pred_crops_dir / "run_export_pred_crops.log")

    pred_manifest_path = pred_crops_dir / manifest_name
    if not pred_manifest_path.exists():
        raise FileNotFoundError(f"Predicted manifest not found: {pred_manifest_path}")

    stage3_base_config = resolve_path(vlm_cfg.get("stage3_runner_config", "configs/pipeline/stage3_vlm_gt_baseline.yaml"))
    if not stage3_base_config.exists():
        raise FileNotFoundError(f"Stage 3 config not found: {stage3_base_config}")

    stage3_effective_cfg = load_yaml(stage3_base_config)
    stage3_effective_cfg.setdefault("input", {})
    stage3_effective_cfg.setdefault("output", {})
    stage3_effective_cfg.setdefault("run", {})
    stage3_effective_cfg.setdefault("backend", {})
    stage3_effective_cfg.setdefault("prompt", {})

    stage3_effective_cfg["input"]["dataset_jsonl"] = str(pred_manifest_path)
    stage3_effective_cfg["output"]["root_dir"] = str(vlm_root_dir)

    vlm_run_id = str(vlm_cfg.get("run_id", f"{run_name}_pred_vlm")).strip() or f"{run_name}_pred_vlm"
    stage3_effective_cfg["run"]["run_id"] = vlm_run_id
    stage3_effective_cfg["run"]["resume"] = False

    backend_mode = str(vlm_cfg.get("backend_mode", stage3_effective_cfg["backend"].get("mode", "qwen_hf"))).strip()
    stage3_effective_cfg["backend"]["mode"] = backend_mode

    model_id = vlm_cfg.get("model_id")
    if isinstance(model_id, str) and model_id.strip():
        stage3_effective_cfg["backend"].setdefault("qwen_hf", {})
        stage3_effective_cfg["backend"]["qwen_hf"]["model_id"] = model_id.strip()

    qwen_overrides = vlm_cfg.get("qwen_hf", {})
    if isinstance(qwen_overrides, dict) and qwen_overrides:
        stage3_effective_cfg["backend"].setdefault("qwen_hf", {})
        stage3_effective_cfg["backend"]["qwen_hf"].update(qwen_overrides)

    prompt_version = vlm_cfg.get("prompt_version")
    if isinstance(prompt_version, str) and prompt_version.strip():
        stage3_effective_cfg["prompt"]["version"] = prompt_version.strip()

    effective_stage3_config_path = vlm_root_dir / "stage3_effective_config.yaml"
    write_yaml(effective_stage3_config_path, stage3_effective_cfg)

    stage3_cmd = [
        sys.executable,
        str((ROOT / "scripts/run_stage3_vlm_baseline.py").resolve()),
        "--config",
        str(effective_stage3_config_path),
        "--run-id",
        vlm_run_id,
        "--no-resume",
    ]
    run_command(stage3_cmd, cwd=ROOT, log_path=vlm_root_dir / "run_stage3_pred.log")

    pred_vlm_run_dir = vlm_root_dir / vlm_run_id
    if not pred_vlm_run_dir.exists():
        raise FileNotFoundError(f"Pred VLM run dir not found: {pred_vlm_run_dir}")

    gt_jsonl = resolve_path(analysis_cfg.get("ground_truth_jsonl"))
    if not gt_jsonl.exists():
        raise FileNotFoundError(f"analysis.ground_truth_jsonl not found: {gt_jsonl}")

    eval_cmd = [
        sys.executable,
        str((ROOT / "scripts/eval_stage4_detector_to_vlm.py").resolve()),
        "--gt-jsonl",
        str(gt_jsonl),
        "--pred-manifest-jsonl",
        str(pred_manifest_path),
        "--pred-vlm-run-dir",
        str(pred_vlm_run_dir),
        "--detector-predictions-json",
        str(detector_predictions_path),
        "--coco-json",
        str(coco_json),
        "--match-iou-threshold",
        str(float(analysis_cfg.get("match_iou_threshold", 0.5))),
        "--good-crop-iou-threshold",
        str(float(analysis_cfg.get("good_crop_iou_threshold", 0.7))),
        "--output-dir",
        str(eval_dir),
    ]

    ceiling_predictions_jsonl = analysis_cfg.get("ceiling_predictions_jsonl")
    ceiling_run_dir = analysis_cfg.get("ceiling_run_dir")
    if isinstance(ceiling_predictions_jsonl, str) and ceiling_predictions_jsonl.strip():
        eval_cmd.extend(["--ceiling-predictions-jsonl", str(resolve_path(ceiling_predictions_jsonl))])
    elif isinstance(ceiling_run_dir, str) and ceiling_run_dir.strip():
        eval_cmd.extend(["--ceiling-run-dir", str(resolve_path(ceiling_run_dir))])

    run_command(eval_cmd, cwd=ROOT, log_path=eval_dir / "run_eval_stage4.log")

    ceiling_vs_actual_path = eval_dir / "ceiling_vs_actual.json"
    if ceiling_vs_actual_path.exists():
        shutil.copy2(ceiling_vs_actual_path, compare_dir / "ceiling_vs_actual.json")

    eval_summary_path = eval_dir / "stage4_summary.md"
    if eval_summary_path.exists():
        shutil.copy2(eval_summary_path, compare_dir / "stage4_summary.md")

    stage4_run_summary = {
        "run_name": run_name,
        "split": split,
        "config_path": str(config_path),
        "artifact_dirs": {
            "detector": str(detector_dir),
            "pred_crops": str(pred_crops_dir),
            "vlm_pred": str(pred_vlm_run_dir),
            "eval": str(eval_dir),
            "compare": str(compare_dir),
        },
        "artifacts": {
            "detector_predictions": str(detector_predictions_path),
            "pred_manifest": str(pred_manifest_path),
            "pred_vlm_predictions": str(pred_vlm_run_dir / "predictions_vlm_labels_v1.jsonl"),
            "stage4_metrics": str(eval_dir / "stage4_metrics.json"),
            "stage4_error_breakdown": str(eval_dir / "stage4_error_breakdown.json"),
            "stage4_case_table": str(eval_dir / "stage4_case_table.csv"),
            "stage4_summary": str(eval_dir / "stage4_summary.md"),
            "ceiling_vs_actual": str(compare_dir / "ceiling_vs_actual.json"),
        },
        "frozen_baselines": {
            "detector_baseline": detector_cfg.get("experiment", "detector_baseline"),
            "detector_config_path": str(detector_config_path),
            "detector_conf_threshold": detector_conf_threshold,
            "detector_iou_threshold_configured": detector_cfg.get("iou_threshold"),
            "detector_max_detections_per_image": detector_cfg.get("max_detections_per_image"),
            "vlm_model_id": stage3_effective_cfg.get("backend", {}).get("qwen_hf", {}).get("model_id"),
            "vlm_prompt_version": stage3_effective_cfg.get("prompt", {}).get("version"),
        },
    }
    write_json(run_dir / "stage4_run_summary.json", stage4_run_summary)

    print(f"Stage 4 run directory: {run_dir}")
    print(f"Detector predictions: {detector_predictions_path}")
    print(f"Predicted manifest: {pred_manifest_path}")
    print(f"Pred VLM run dir: {pred_vlm_run_dir}")
    print(f"Stage4 metrics: {eval_dir / 'stage4_metrics.json'}")


if __name__ == "__main__":
    main()
