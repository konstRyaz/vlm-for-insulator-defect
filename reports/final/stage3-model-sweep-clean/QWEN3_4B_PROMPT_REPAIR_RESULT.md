# Qwen3-4B Prompt Repair Result

This run tested whether Qwen3-VL-4B can keep its strong `insulator_ok` behavior while recovering `defect_flashover` recall through small prompt-only repairs.

Run source: Kaggle kernel `kostyaryazanov/notebookd64e91cba0`, version 18.

## Setup

- Model: `Qwen/Qwen3-VL-4B-Instruct`
- Dataset: clean test split (historical `val_v2` path), 58 GT crops
- Output contract: `vlm_labels_v1`
- Max pixels: `401408`
- GPU: Kaggle T4
- Control prompt: `qwen_vlm_labels_v1_prompt_v7f_flashover_unclear_to_unknown_nocroppath`
- Candidate prompts: `v8a`, `v8b`, `v8c`

## Results

| prompt | correct | acc | macro-F1 | OK recall | flashover recall | broken recall | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v7f control | 31/58 | 0.5345 | 0.2748 | 27/32 | 2/20 | 2/6 | control |
| v8a flashover recall | 30/58 | 0.5172 | 0.1998 | 26/32 | 4/20 | 0/6 | weak repair |
| v8b balanced | 29/58 | 0.5000 | 0.1732 | 27/32 | 2/20 | 0/6 | no repair |
| v8c positive evidence | 28/58 | 0.4828 | 0.1532 | 27/32 | 1/20 | 0/6 | no repair |

All runs had parse success and schema validity equal to `1.0`.

## Interpretation

The targeted prompt repair did not produce a useful Qwen3-4B variant. The only variant that increased flashover recall was `v8a`, but the gain was small: `2/20 -> 4/20`. It also reduced overall correctness and lost all `defect_broken` hits.

This supports the earlier conclusion: the Qwen3-4B frozen model has a strong normal-insulator bias under the current task framing. Prompt-only repair can move the boundary a little, but not enough to make Qwen3-4B a better Stage 3 model than the clean Qwen2.5-VL-3B baseline.

## Decision

Do not promote Qwen3-4B prompt variants to Stage 4.

Stop broad prompt-only work for frozen VLMs. The next useful branch should be one of:

1. a supervised Qwen2.5-VL-3B LoRA/QLoRA experiment;
2. a hybrid coarse classifier plus Qwen structured reporter;
3. a coarse-only TL-CLIP / CLIP-style benchmark if usable weights are available.

The current result is still scientifically useful: larger/newer frozen VLMs and prompt repair did not solve the `insulator_ok` vs `defect_flashover` bottleneck, so the next research step should involve adaptation or a separate discriminative coarse-class component.
