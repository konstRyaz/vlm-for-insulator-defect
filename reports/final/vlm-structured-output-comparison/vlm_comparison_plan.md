# VLM comparison plan

The next operation compares frozen VLM backbones under the clean Stage 3 protocol.

The comparison rule is simple. A candidate sees the crop image and the clean task prompt only. It must output the same `vlm_labels_v1` fields. It does not receive the crop path, folder name, detector class, ground-truth class, or annotation text.

The first two non-Qwen candidates are:

| notebook | candidate | purpose |
|---|---|---|
| `stage3_vlm_triage_internvl3_2b_clean` | `OpenGVLab/InternVL3-2B-hf` | small InternVL family preflight and clean test test |
| `stage3_vlm_triage_llava_onevision_0_5b_clean` | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | small LLaVA-OneVision compatibility and clean test test |

A model is worth promoting only if it passes parse/schema checks and improves more than noise. On 58 objects, one object is about 1.7 percentage points, so tiny changes should not be overinterpreted.

Promotion gate:

| metric | desired signal |
|---|---|
| parse success | 1.0 or near 1.0 with easy formatting repair |
| schema valid | 1.0 or near 1.0 with easy formatting repair |
| coarse accuracy | better than Qwen3B by more than one object |
| macro-F1 | improves class balance, not only normal-class accuracy |
| flashover vs ok | fewer destructive trade-offs than Qwen prompt sweeps |

If no frozen VLM improves clean Stage 3, the next serious path remains domain adaptation or a hybrid discriminative coarse classifier plus Qwen reporter.
