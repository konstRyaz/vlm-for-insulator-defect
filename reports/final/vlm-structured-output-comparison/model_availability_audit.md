# Model availability and run audit

This audit follows the archived VLM comparison plan. Models are separated by role: full structured VLM reporters, coarse-only encoders, and related-work/domain models without a confirmed runnable path.

## Completed structured-reporter checks

| model key | model id | status | result |
|---|---|---|---|
| `qwen25vl_3b_control` | `Qwen/Qwen2.5-VL-3B-Instruct` | runnable existing | kept as clean structured baseline |
| `internvl3_2b_base` | `OpenGVLab/InternVL3-2B-hf` | completed | parse/schema stable, higher raw acc, not promoted due macro-F1/tag/visibility regression |
| `internvl3_2b_defect_recall` | `OpenGVLab/InternVL3-2B-hf` + recall addendum | completed | defect overcall; not promoted |
| `internvl3_2b_balanced_defect` | `OpenGVLab/InternVL3-2B-hf` + balanced addendum | completed | defect recall too low; not promoted |
| `llava_onevision_qwen2_0_5b` | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | completed | schema/semantic failure under current JSON reporter protocol |
| `smolvlm2_2b_instruct` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | completed | parse failure |
| `smolvlm2_500m_video_instruct` | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | completed | partial parse but class collapse |
| `phi35_vision_instruct` | `microsoft/Phi-3.5-vision-instruct` | preflight failed | generic image-text-to-text pipeline incompatible; not counted as benchmark failure |

## Not promoted to Stage 4

No new frozen structured reporter passes the promotion gate. The only useful challenger, InternVL3-2B base, improves raw accuracy but not macro-F1 and degrades visibility/tag quality. The archived plan therefore points away from more broad frozen-VLM sweeps.

## Domain-specific model status

| candidate | current status | evaluation role |
|---|---|---|
| TL-CLIP | public runnable path still needs confirmation | coarse-only benchmark, not JSON reporter |
| PowerGPT | related work unless runnable weights/API/code are confirmed | possible structured reporter only if runnable |
| Power-LLaVA | related work unless runnable weights/API/code are confirmed | possible structured reporter only if runnable |
| PLVLDet | detector-related work | not a VLM reporter |

## Next valid branch

The next comparison branch should be domain adaptation or hybridization, not another broad frozen reporter sweep. The two concrete options are Qwen2.5-VL-3B LoRA/QLoRA on clean training-split data, or a hybrid discriminative coarse classifier plus Qwen structured reporting.
