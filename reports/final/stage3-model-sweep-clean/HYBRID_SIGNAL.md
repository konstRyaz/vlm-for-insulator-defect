# Stage 3 Model Sweep And Hybrid Signal

This note records the local analysis done after the clean frozen Qwen model sweep.

## Completed Frozen Runs

The completed models were:

- `Qwen/Qwen2.5-VL-3B-Instruct`
- `Qwen/Qwen2.5-VL-7B-Instruct`
- `Qwen/Qwen3-VL-4B-Instruct`

`Qwen/Qwen2.5-VL-7B-Instruct-AWQ` failed at preflight. `Qwen/Qwen3-VL-2B-Instruct` passed preflight but failed at validation because the produced predictions did not pass the existing `vlm_labels_v1` validation path.

## Main Result

The larger/newer frozen models are not drop-in replacements for the clean 3B baseline.

| model | correct | accuracy | macro-F1 | OK recall | flashover recall | broken recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-3B | 28/58 | 0.4828 | 0.2946 | 0.3125 | 0.7500 | 0.5000 |
| Qwen2.5-VL-7B | 29/58 | 0.5000 | 0.1556 | 0.8750 | 0.0500 | 0.0000 |
| Qwen3-VL-4B | 31/58 | 0.5345 | 0.2748 | 0.8438 | 0.1000 | 0.3333 |

Qwen3-VL-4B has the best raw accuracy, but it mostly gets there by predicting normal insulators well while losing flashover recall. The result is useful, but not a direct improvement for the project bottleneck.

## Hybrid / Oracle Check

A simple local analysis compared 3B, 7B, and Qwen3-4B predictions on the same 58 objects.

| rule | correct | accuracy | macro-F1 | OK correct | flashover correct | broken correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3B | 28 | 0.4828 | 0.2946 | 10/32 | 15/20 | 3/6 |
| 7B | 29 | 0.5000 | 0.1556 | 28/32 | 1/20 | 0/6 |
| Qwen3-4B | 31 | 0.5345 | 0.2748 | 27/32 | 2/20 | 2/6 |
| majority vote, defect tie | 33 | 0.5690 | 0.2715 | 27/32 | 5/20 | 1/6 |
| 3B defects else Qwen3-4B | 30 | 0.5172 | 0.3057 | 12/32 | 15/20 | 3/6 |
| oracle any model correct | 47 | 0.8103 | 0.4888 | 28/32 | 15/20 | 4/6 |

The oracle is high, so the models contain complementary signal. But simple rule-based ensembles do not yet provide a clean win: majority voting improves raw accuracy while hurting macro-F1 and flashover recall.

## Decision

Do not promote 7B or Qwen3-4B as-is.

The next experiment is a targeted Qwen3-4B prompt repair sweep:

`notebooks/stage3_qwen3_4b_prompt_repair_clean.ipynb`

The goal is not broad prompt tuning. It tests whether Qwen3-4B can keep its strong `insulator_ok` behavior while recovering enough `defect_flashover` recall to become useful.

Acceptance signal:

- parse/schema remains 1.0;
- flashover recall improves substantially over Qwen3-4B control;
- OK recall remains reasonable;
- coarse macro-F1 beats the clean 3B baseline or at least becomes competitive.
