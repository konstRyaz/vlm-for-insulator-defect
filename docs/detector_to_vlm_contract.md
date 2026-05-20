# Detector to VLM Contract (Baseline v1)

## Purpose

This contract defines how detector outputs are filtered, routed, and handed off to Stage 3 VLM processing.
It is the operating policy for baseline integration and should be treated as explicit system behavior, not implicit assumptions.

## Scope

- Detector source: `detector_baseline_v1` (Faster R-CNN baseline checkpoint).
- Stage focus: transition from detector-only evaluation to controlled region-level VLM input.
- This contract is intentionally conservative to reduce noise and scope creep.

## 1. Eval vs Pipeline Threshold

- `eval_threshold: 0.05`
  - used only for COCO-style detector evaluation/export.
- `pipeline_threshold_default: 0.5`
  - default runtime threshold for `detector -> VLM` handoff.
- `pipeline_threshold_alt: 0.3`
  - optional alternative for recall-oriented ablations.

## 2. Routing Policy

Default routing to VLM is defect-first:

- send to VLM:
  - `defect_flashover`
  - `defect_broken`
- do not send all detections blindly.

Reason: keep region count manageable and avoid propagating detector noise to the VLM stage.

## 3. Unknown Policy

`unknown` is not a central descriptive target class in baseline v1.

Policy:

- treat `unknown` as `review` / uncertain finding;
- allow suppression in default pipeline mode;
- keep this behavior explicit in config so it is auditable and ablatable later.

## 4. Insulator_OK Policy

`insulator_ok` is not mass-routed into VLM by default.

Policy:

- suppress most `insulator_ok` regions from VLM input;
- use as fallback/no-defect signal only when no defect regions remain after filtering.

## 5. Top-k Policy

To control downstream VLM load:

- `max_regions_per_image: 5`
- sort by score descending
- defect categories have higher routing priority than `insulator_ok` and `unknown`

Recommended priority order:

1. `defect_flashover`
2. `defect_broken`
3. `insulator_ok`
4. `unknown`

## 6. Crop Policy

- `crop_padding_ratio: 0.15`
- apply symmetric context around bbox before crop extraction
- always clip expanded bbox to image boundaries

This is the initial baseline crop policy; context ablation can be done later.

## 7. Output / Handoff Contract

Each routed region should produce a structured record with at least:

- `image_id`
- `image_path`
- `source` (`pred` now, `gt` supported for Stage 3 calibration)
- `bbox_xywh`
- `score`
- `category_id`
- `category_name`
- `crop_path` (if crop file exists)
- `needs_review`
- `routing_decision`

Recommended `routing_decision` values:

- `send_to_vlm`
- `review`
- `suppress`
- `no_defect_signal`

## 8. Non-Goals for This Baseline

- no YOLO migration in this step;
- no taxonomy rebuild;
- no new detector training loop;
- no GPU dependency for policy artifacts.
