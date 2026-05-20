# VLM Model Availability Audit

This is the first checkpoint for the post-Stage-4 model expansion branch. It separates runnable candidates from paper-only or classifier-only candidates so that the next experiments stay comparable with the clean Stage 3/4 baseline.

## Baseline To Beat

Reference model:

- `Qwen/Qwen2.5-VL-3B-Instruct`
- prompt: `qwen_vlm_labels_v1_prompt_v7f_flashover_unclear_to_unknown_nocroppath`
- clean Stage 3 GT-crop coarse accuracy: `0.4655`
- clean Stage 4 best actual rate: `0.3966`

Any new model must use the no-crop-path protocol.

## Candidate Status

| Candidate | Type | Current status | Recommended action |
| --- | --- | --- | --- |
| Qwen2.5-VL-7B-Instruct | generative VLM | likely runnable through the existing Qwen backend if GPU memory permits | first same-family scale test |
| Qwen2.5-VL-7B-Instruct-AWQ | quantized generative VLM | useful if full 7B is memory-limited | run after full 7B preflight, or keep as fallback |
| Qwen3-VL-2B-Instruct | generative VLM | newer small Qwen candidate; requires recent Transformers support | frozen generalization check |
| Qwen3-VL-4B-Instruct | generative VLM | newer mid-small Qwen candidate; requires recent Transformers support | frozen generalization check |
| PowerGPT | power-domain generative MLLM | paper exists; public runnable weights/code not confirmed from quick search | keep as related work until weights/code are found |
| TL-CLIP | CLIP-style power-domain model | paper exists; contrastive/classification-style, not a JSON-generating VLM | evaluate only as coarse classifier if weights/code are found |
| Qwen2.5-VL-3B LoRA/QLoRA | fine-tuned generative VLM | not implemented yet | do after frozen model checks |

## Notes From Public Sources

PowerGPT is described as a multimodal foundation model for power inspection, with a Power Security Instruction Dataset and PowerBENCH benchmark. That makes it highly relevant scientifically, but it is not automatically runnable in this repo unless the authors release usable weights or inference code.

TL-CLIP is described as a power-specific contrastive language-image pre-training framework for transmission line defect recognition. It is relevant to domain adaptation, but its natural evaluation is coarse classification or image-text ranking, not full `vlm_labels_v1` JSON generation.

## Immediate Experiment

Run the clean Qwen scale sweep notebook:

`notebooks/stage3_model_sweep_qwen_clean.ipynb`

It keeps the prompt and dataset fixed, changes only `model_id`, runs a one-sample preflight per model, and runs full test-split evaluation only for candidates whose preflight succeeds.

Current frozen sweep candidates:

- `Qwen/Qwen2.5-VL-3B-Instruct`
- `Qwen/Qwen2.5-VL-7B-Instruct`
- `Qwen/Qwen2.5-VL-7B-Instruct-AWQ`
- `Qwen/Qwen3-VL-2B-Instruct`
- `Qwen/Qwen3-VL-4B-Instruct`

## Acceptance Signal

A stronger model is worth keeping only if it improves coarse classification without breaking structured output:

- parse success = `1.0`
- schema valid = `1.0`
- coarse accuracy above the clean 3B baseline
- coarse macro-F1 above the clean 3B baseline
- visibility does not collapse

Treat tiny gains cautiously. On the 58-object test split, a one-object change is about 1.7 percentage points. A model should preferably gain at least five objects and improve macro-F1 before it is treated as a strong candidate.

If Qwen2.5-VL-7B fails memory on Kaggle T4, do not force it by changing the evaluation protocol too much. Record it as environment-limited and move to either a quantized model path or Qwen3B LoRA.
