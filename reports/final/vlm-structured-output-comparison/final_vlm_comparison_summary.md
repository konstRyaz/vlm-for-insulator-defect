# Final VLM comparison summary

This report closes the archived frozen-VLM comparison pass for clean Stage 3. The protocol was fixed: same clean `val_v2` GT crops, same `vlm_labels_v1` output contract, same evaluator, no `crop_path` or class-like file-path hints in the prompt.

## Main result

No new frozen VLM is promoted to Stage 4. InternVL3-2B is the only tested non-Qwen model that kept `parse_success=1.0` and `schema_valid=1.0` while improving raw accuracy, but it did not improve macro-F1 and it lost too much visibility/tag quality. The repair variants moved the defect boundary, but one overcalled defects and the other undercalled them.

| model | parse | schema | acc | macro-F1 | visibility macro-F1 | tag Jaccard | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `qwen25vl_3b_control` | 1.0000 | 1.0000 | 0.4828 | 0.2946 | 0.5218 | 0.1977 | baseline anchor |
| `internvl3_2b_base` | 1.0000 | 1.0000 | 0.5517 | 0.2853 | 0.2949 | 0.0330 | not promoted |
| `internvl3_2b_defect_recall` | 1.0000 | 1.0000 | 0.3966 | 0.2255 | 0.2949 | 0.0517 | failed: defect overcall |
| `internvl3_2b_balanced_defect` | 1.0000 | 1.0000 | 0.5000 | 0.2316 | 0.2949 | 0.1580 | failed: low defect recall |
| `llava_onevision_qwen2_0_5b` | 0.7931 | 0.2414 | 0.1207 | 0.0609 | 0.1021 | 0.0000 | failed schema/semantics |
| `smolvlm2_2b_instruct` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0896 | 0.0000 | failed parse |
| `smolvlm2_500m_video_instruct` | 0.6034 | 0.6034 | 0.3276 | 0.1134 | 0.0526 | 0.0000 | failed class collapse |
| `phi35_vision_instruct` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | not run: generic pipeline incompatible |

## Paired behavior against Qwen control

The paired analysis confirms why InternVL3-2B is not a clean replacement. The base model fixes 16 Qwen mistakes, but breaks 12 Qwen-correct cases. Its raw accuracy gain is therefore real but narrow, and the macro-F1/visibility/tag regressions make it unsafe to promote as the new structured reporter.

| candidate | helped wrong->right | hurt right->wrong | both right | both wrong same | both wrong different |
|---|---:|---:|---:|---:|---:|
| `internvl3_2b_base` | 16 | 12 | 16 | 7 | 7 |
| `internvl3_2b_defect_recall` | 7 | 12 | 16 | 15 | 8 |
| `internvl3_2b_balanced_defect` | 15 | 14 | 14 | 7 | 8 |

## Interpretation

The clean Stage 3 bottleneck remains semantic discrimination, especially normal insulators versus flashover-like surface evidence. Bigger or different frozen open VLMs did not remove that bottleneck under the current protocol. The results support the earlier research direction: keep Qwen2.5-VL-3B as the structured reporter baseline, and move the next improvement attempt to domain adaptation or a hybrid discriminative coarse classifier plus structured reporter.

## Domain-specific models

PowerGPT and Power-LLaVA remain relevant related work, but they are not counted as failed experiments because no runnable public inference path was confirmed in this pass. TL-CLIP remains a coarse-only candidate, not a `vlm_labels_v1` JSON reporter. PLVLDet is detector-related work and should not be mixed into the VLM reporter table.

## Limitations

The test split has 58 GT objects, so one object changes accuracy by about 1.7 percentage points. These results are good enough for a reproducible baseline and error-decomposition story, but not enough for broad deployment claims.

## Decision

Freeze this frozen-VLM comparison checkpoint. Do not launch Stage 4 for InternVL/LLaVA/SmolVLM/Phi from these runs. The next operation should be one of two branches: Qwen LoRA/QLoRA on the clean split, or a clean hybrid system where a discriminative coarse classifier supplies `coarse_class` and Qwen remains the structured reporter.

## Specialized Model Audit Update

A follow-up audit on 2026-05-02 checked TL-CLIP, PowerGPT, and Power-LLaVA. No public runnable, provenance-clear checkpoint/API was found for these three models. They remain related-work or future-work candidates, not executed baselines. See `reports/vlm_comparison/specialized_models_deep_audit_2026-05-02.md`.

