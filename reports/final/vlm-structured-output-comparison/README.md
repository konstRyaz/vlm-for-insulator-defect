# VLM comparison workspace

This folder tracks the next Stage 3/Stage 4 model-comparison pass. The goal is not to change the detector, schema, evaluator, or crop protocol. The goal is to test whether another frozen VLM backbone can improve clean Stage 3 semantics under the same leakage-free GT-crop protocol.

The comparison starts from the current clean baseline:

| item | value |
|---|---|
| canonical Stage 3 prompt | `qwen_vlm_labels_v1_prompt_v7f_flashover_unclear_to_unknown_nocroppath` |
| baseline model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| test split | `stage3_regrouped_v2/val/vlm_labels_v1_val_v2.annotated.jsonl` |
| contract | `vlm_labels_v1` JSON |
| leakage rule | do not pass `crop_path`, class-like folder names, annotation text, or label tokens to the VLM prompt |

The first pass is a triage sweep. A model must pass one-sample preflight and a small clean micro-run before it is considered for full Stage 3 or Stage 4.
