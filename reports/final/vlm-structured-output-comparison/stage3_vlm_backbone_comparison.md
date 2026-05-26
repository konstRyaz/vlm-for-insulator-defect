# Stage 3 VLM backbone comparison final checkpoint

This checkpoint follows the archived VLM comparison plan: keep the clean Stage 3 dataset/schema/evaluator fixed, compare runnable frozen VLM backbones, and promote to Stage 4 only if Stage 3 improves beyond noise without parse/schema collapse.

| model | parse | schema | acc | macro-F1 | visibility macro-F1 | tag Jaccard | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `qwen25vl_3b_control` | 1.0000 | 1.0000 | 0.4828 | 0.2946 | 0.5218 | 0.1977 | BASELINE_ANCHOR |
| `internvl3_2b_base` | 1.0000 | 1.0000 | 0.5517 | 0.2853 | 0.2949 | 0.0330 | BEST_RAW_ACC_NOT_PROMOTED |
| `internvl3_2b_defect_recall` | 1.0000 | 1.0000 | 0.3966 | 0.2255 | 0.2949 | 0.0517 | FAILED_OVERCALL_DEFECTS |
| `internvl3_2b_balanced_defect` | 1.0000 | 1.0000 | 0.5000 | 0.2316 | 0.2949 | 0.1580 | FAILED_LOW_DEFECT_RECALL |
| `llava_onevision_qwen2_0_5b` | 0.7931 | 0.2414 | 0.1207 | 0.0609 | 0.1021 | 0.0000 | FAILED_SCHEMA_OR_SEMANTICS |
| `smolvlm2_2b_instruct` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0896 | 0.0000 | FAILED_PARSE |
| `smolvlm2_500m_video_instruct` | 0.6034 | 0.6034 | 0.3276 | 0.1134 | 0.0526 | 0.0000 | FAILED_CLASS_COLLAPSE |
| `phi35_vision_instruct` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NOT_RUN_GENERIC_PIPELINE_INCOMPATIBLE |

## Decision

`InternVL3-2B` is the only new frozen VLM with full parse/schema stability and higher raw accuracy than the Qwen2.5-VL-3B control. It is not promoted to Stage 4 because macro-F1 does not improve, visibility/tag quality is weaker, and prompt-repair variants either overcall defects or lose defect recall.

The frozen VLM comparison is therefore closed for now. The next improvement path should not be another broad VLM sweep; it should be domain adaptation or a hybrid discriminative classifier plus structured reporter.

## Notes

- `Phi-3.5-Vision` is not counted as a failed benchmark because it was incompatible with the current generic HF image-text-to-text pipeline path.
- PowerGPT/Power-LLaVA/TL-CLIP remain availability/coarse-only related-work items unless a runnable public inference path is confirmed.
- Stage 4 is not launched for InternVL because the Stage 3 promotion gate was not met.
