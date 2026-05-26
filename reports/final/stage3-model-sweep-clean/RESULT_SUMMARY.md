# Stage 3 Clean Qwen Model Sweep Result

This run tested frozen Qwen-family models under the clean no-crop-path Stage 3 protocol. The dataset, prompt, output schema, and evaluator were fixed. Only `model_id` changed.

Run source: Kaggle kernel `kostyaryazanov/notebookd64e91cba0`, version 16.

## Setup

- Dataset: clean test split (historical `val_v2` path), 58 GT crops
- Prompt: `qwen_vlm_labels_v1_prompt_v7f_flashover_unclear_to_unknown_nocroppath`
- Max pixels: `401408`
- Output contract: `vlm_labels_v1`
- GPU: Kaggle T4

## Results

| model | full run | coarse acc | correct | coarse macro-F1 | visibility macro-F1 | tag Jaccard | OK recall | flashover recall | broken recall | verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen2.5-VL-3B-Instruct | yes | 0.4828 | 28/58 | 0.2946 | 0.5218 | 0.1977 | 0.3125 | 0.7500 | 0.5000 | control |
| Qwen2.5-VL-7B-Instruct | yes | 0.5000 | 29/58 | 0.1556 | 0.5593 | 0.4066 | 0.8750 | 0.0500 | 0.0000 | weak signal, not useful as-is |
| Qwen3-VL-4B-Instruct | yes | 0.5345 | 31/58 | 0.2748 | 0.5577 | 0.2876 | 0.8438 | 0.1000 | 0.3333 | weak signal, class trade-off |
| Qwen2.5-VL-7B-Instruct-AWQ | no | - | - | - | - | - | - | - | - | preflight failed |
| Qwen3-VL-2B-Instruct | no | - | - | - | - | - | - | - | - | full run failed at schema validation |

## Interpretation

The sweep does not produce a strong replacement for the clean Qwen2.5-VL-3B baseline. Qwen3-VL-4B and Qwen2.5-VL-7B improve raw accuracy by predicting many more normal insulators correctly, but both largely lose the flashover class. This is not the improvement we need: the core research bottleneck is `insulator_ok` versus `defect_flashover`, and these candidates shift the boundary too far toward `insulator_ok`.

The same-family 7B model is especially informative. It reaches 29/58 correct, but flashover recall drops to 1/20 and broken recall drops to 0/6. The larger model therefore does not solve the semantic defect problem under the current prompt; it changes the error profile.

Qwen3-VL-4B is the best raw-accuracy candidate at 31/58, but its macro-F1 is lower than the 3B control and flashover recall is only 2/20. It should not be promoted to Stage 4 as a final model without class-boundary repair.

## Decision

Do not replace the clean Stage 3 baseline with 7B or Qwen3-4B as-is.

The next useful branch is not another broad frozen-model sweep. The next branch should be one of:

1. a targeted prompt adaptation for Qwen3-VL-4B focused on recovering flashover recall;
2. a hybrid coarse classifier plus Qwen reporter;
3. Qwen2.5-VL-3B LoRA/QLoRA on clean training-split labels.

Given the current results, the most promising research path is supervised adaptation or a hybrid coarse classifier, not simply increasing frozen model size.
