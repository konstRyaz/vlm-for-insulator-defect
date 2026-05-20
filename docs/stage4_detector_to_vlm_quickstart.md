# Stage 4 Quickstart: Detector -> Pred Crop -> VLM

Stage 4 measures real pipeline quality after detector errors.

Flow:

`val images -> detector_baseline_v1 -> predicted boxes/crops -> Stage 3 VLM (frozen) -> Stage 4 evaluation`

Current clean Stage 4 default:

- VLM model: `Qwen/Qwen2.5-VL-3B-Instruct`
- prompt version: `qwen_vlm_labels_v1_prompt_v7f_flashover_unclear_to_unknown_nocroppath`
- best current input-strategy candidate: context padding `0.30` with `max_pixels=401408`

The Stage 4 prompt removes `crop_path` from prompt-visible metadata so the VLM cannot read detector class names from crop folder paths.
The current clean reference uses the final leakage-free Stage 3 prompt listed above.

## 1) Required inputs

Check these paths in `configs/stage4_detector_to_vlm_pred_val.yaml`:

- detector images: `detector.input_dir`
- detector weights: `detector.weights_path`
- COCO annotations for image/category lookup: `crop_export.coco_json`
- Stage 3 GT labels for object-level evaluation: `analysis.ground_truth_jsonl`
- clean Stage 3 GT-crop ceiling run (optional but recommended): `analysis.ceiling_run_dir`

The default config is a template; adjust paths to your local/Kaggle layout.

Recommended clean ceiling run id:

- `outputs/stage3_vlm_baseline_runs/stage3_qwen_val_v2_clean_final`

Notebook entry point for Kaggle:

- `notebooks/stage4_detector_to_vlm_kaggle_run.ipynb`

Context-padding ablation notebook:

- `notebooks/stage4_context_pad050_kaggle_run.ipynb`
- `notebooks/stage4_context_pad030_kaggle_run.ipynb`

Current context comparison report:

- `reports/stage4_context_comparison_final/report.md`

## 2) Run Stage 4 in one command

```bash
python scripts/run_stage4_detector_to_vlm.py \
  --config configs/stage4_detector_to_vlm_pred_val.yaml
```

This command runs all stages and writes outputs to:

- `outputs/stage4/<run_name>/01_detector/`
- `outputs/stage4/<run_name>/02_pred_crops/`
- `outputs/stage4/<run_name>/03_vlm_pred/`
- `outputs/stage4/<run_name>/04_eval/`
- `outputs/stage4/<run_name>/05_compare/`

## 3) Key artifacts

Main artifacts after run:

- detector predictions: `01_detector/predictions.json`
- predicted-crop manifest: `02_pred_crops/pred_manifest.jsonl`
- VLM predictions on predicted crops: `03_vlm_pred/<vlm_run_id>/predictions_vlm_labels_v1.jsonl`
- Stage 4 metrics: `04_eval/stage4_metrics.json`
- error buckets: `04_eval/stage4_error_breakdown.json`
- per-GT case table: `04_eval/stage4_case_table.csv`
- Stage 4 summary: `04_eval/stage4_summary.md`
- ceiling vs actual summary: `05_compare/ceiling_vs_actual.json`

Optional targeted review from an existing Stage 4 eval:

```bash
python scripts/build_stage4_targeted_error_review.py \
  --case-table outputs/stage4/<run_name>/04_eval/stage4_case_table.csv \
  --pred-manifest-jsonl outputs/stage4/<run_name>/02_pred_crops/pred_manifest.jsonl \
  --predictions-jsonl outputs/stage4/<run_name>/03_vlm_pred/<vlm_run_id>/predictions_vlm_labels_v1.jsonl \
  --raw-responses-jsonl outputs/stage4/<run_name>/03_vlm_pred/<vlm_run_id>/raw_responses.jsonl \
  --pred-crops-dir outputs/stage4/<run_name>/02_pred_crops \
  --out-dir outputs/stage4/<run_name>/04_eval/targeted_review
```

## 4) What Stage 4 metrics mean

- `detector_match_rate`: GT objects with a matched predicted box (`IoU >= match_iou_threshold`).
- `good_crop_rate_among_matched`: matched boxes with `IoU >= good_crop_iou_threshold`.
- `vlm_correct_rate_among_good_pred_crops`: coarse-class correctness on good predicted crops.
- `pipeline_correct_rate`: end-to-end correctness over all GT objects.
- `ceiling_correct_rate`: GT-crop Stage 3 correctness (upper bound).
- `ceiling_vs_actual_gap`: ceiling minus actual pipeline rate.

Error buckets are stored in `stage4_error_breakdown.json`:

- `detector_miss`
- `bad_crop_from_detector`
- `vlm_error_on_good_pred_crop`
- `routing_or_filtering_error`
- `correct_pipeline_hit`
