# Specialized VLM Availability Audit - 2026-05-02

This audit covers the three domain-specific models named in the comparison plan: TL-CLIP, PowerGPT, and Power-LLaVA. The goal was to determine whether each can be benchmarked reproducibly in the clean Stage 3/Stage 4 protocol.

## Audit Method

Sources checked:

- paper pages / abstracts;
- Hugging Face model search and model metadata API;
- GitHub repository search;
- direct checks for candidate HF repositories discovered by search.

The criterion for a runnable experiment is strict: public weights or a public API, a plausible inference path, and enough provenance to say we are testing the named model rather than an unrelated checkpoint.

## Summary

| candidate | type | runnable status | decision |
|---|---|---|---|
| TL-CLIP | CLIP-style contrastive model | not runnable: weights/code not found | related work / future coarse-only benchmark |
| PowerGPT | generative power-inspection MLLM | not runnable: public weights/API/code not found | related work only |
| Power-LLaVA | generative transmission-line VLM | not runnable: official release not found | related work only |

## TL-CLIP

Paper: `https://arxiv.org/abs/2411.11370`

The paper describes a transmission-line-oriented contrastive language-image pretraining framework for defect recognition. This is naturally a `crop -> coarse_class` or embedding/ranking baseline, not a `vlm_labels_v1` JSON reporter.

Audit result:

- Hugging Face search did not reveal a clear TL-CLIP checkpoint.
- GitHub search for exact `TL-CLIP` returned no relevant transmission-line model repo.
- No runnable weights/code were found.

Decision: do not run. Keep TL-CLIP as related work and as a future coarse-only benchmark if the authors release weights/code.

## PowerGPT

Paper: `https://www.sciencedirect.com/science/article/pii/S1568494625012529`

The paper describes PowerGPT, PSID, and PowerBench for power inspection. This would be highly relevant as a structured VLM reporter if a public runnable model were available.

Audit result:

- Hugging Face search for `PowerGPT` / `PowerGPT power inspection` did not reveal a usable public model.
- GitHub search for `PowerGPT` plus `power inspection` did not reveal a usable repo.
- No public inference API or checkpoint was found.

Decision: do not run. Cite as related work only; do not count as failed experimental benchmark.

## Power-LLaVA

Paper: `https://arxiv.org/abs/2407.19178`

The paper states that code shall be released. A reproducible benchmark requires an official or provenance-clear checkpoint.

Audit result:

- Hugging Face search found `xjtulmx/powerllava`, but its model repo exposes only `.gitattributes`; no usable model files or README were available.
- Hugging Face search also found `power0341/llava-v1_5-mlp2x-336px-qwen1_8b`, but its README points to a generic LLaVA/Qwen fork and does not establish that it is the Power-LLaVA paper model.
- GitHub search did not reveal an official Power-LLaVA release with weights.

Decision: do not run the unverified checkpoints as Power-LLaVA. Keep Power-LLaVA as related work until an official/provenance-clear release appears.

## Consequence for the Research Plan

The three domain-specific models cannot currently be included as executed Stage 3/Stage 4 baselines. This is not a negative model result; it is an availability result. The current experimental path should remain:

1. frozen general VLM comparison: closed;
2. Qwen LoRA/SFT repair: negative adaptation checkpoint;
3. DINOv2 + Qwen hybrid: current best reproducible Stage 4 result;
4. domain-specific models: related work / future work unless runnable releases appear.
