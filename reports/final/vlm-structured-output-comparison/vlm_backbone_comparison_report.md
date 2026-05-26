# VLM Backbone Comparison Report

This report is generated from the frozen VLM comparison tables. It keeps the clean Stage 3 protocol fixed and summarizes which models are safe to promote.

## Stage 3 Structured Reporter Runs

| model | parse | schema | acc | macro-F1 | visibility macro-F1 | tag Jaccard |
|---|---:|---:|---:|---:|---:|---:|
| `qwen25vl_3b_control` | 1.0000 | 1.0000 | 0.4828 | 0.2946 | 0.5218 | 0.1977 |
| `internvl3_2b_base` | 1.0000 | 1.0000 | 0.5517 | 0.2853 | 0.2949 | 0.0330 |
| `internvl3_2b_defect_recall` | 1.0000 | 1.0000 | 0.3966 | 0.2255 | 0.2949 | 0.0517 |
| `internvl3_2b_balanced_defect` | 1.0000 | 1.0000 | 0.5000 | 0.2316 | 0.2949 | 0.1580 |
| `llava_onevision_qwen2_0_5b` | 0.7931 | 0.2414 | 0.1207 | 0.0609 | 0.1021 | 0.0000 |
| `smolvlm2_2b_instruct` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0896 | 0.0000 |
| `smolvlm2_500m_video_instruct` | 0.6034 | 0.6034 | 0.3276 | 0.1134 | 0.0526 | 0.0000 |
| `phi35_vision_instruct` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Domain-Specific and Coarse-Only Candidates

| candidate | status | eval mode | blocker |
|---|---|---|---|
| `TL-CLIP` | pending_weights_code_confirmation | coarse_classifier_only | public weights/code not confirmed locally |
| `PowerGPT` | related_work_until_runnable_release_confirmed | structured_reporter_if_runnable | public runnable inference path not confirmed |
| `Power-LLaVA` | related_work_until_runnable_release_confirmed | structured_reporter_if_runnable | public runnable inference path not confirmed |
| `PLVLDet` | related_work_or_detector_baseline | detector_baseline | not a structured VLM reporter |

## Decision

No new frozen VLM is promoted to Stage 4 from this pass. InternVL3-2B improves raw accuracy, but it does not improve macro-F1 and loses visibility/tag quality. The next branch should be domain adaptation or a hybrid discriminative coarse classifier plus structured Qwen reporter.

Stage 3 rows: 8
Stage 4 rows: 2
Domain status rows: 4
