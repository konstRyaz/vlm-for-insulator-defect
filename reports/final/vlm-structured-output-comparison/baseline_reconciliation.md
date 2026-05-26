# Baseline reconciliation

The clean Stage 3 Qwen baseline appears in two close but not identical contexts.

| source | run | coarse accuracy | note |
|---|---|---:|---|
| final clean report | `stage3_qwen_val_v2_clean_final` | 0.4655 | final v7f no-crop-path reportable checkpoint |
| Qwen model sweep control | `stage3_qwen_val_v2_model_sweep_clean_qwen25vl_3b_control` | 0.4828 | same family control inside model sweep |

For model-comparison decisions, treat this as a reproducibility band rather than a contradiction. A new frozen VLM should beat the 3B baseline by more than one object on the 58-object test split, and should improve macro-F1 or the flashover/ok confusion profile rather than only raw accuracy.
