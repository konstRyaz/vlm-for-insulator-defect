# Таблица основных запусков

| run_id | Этап | Тип | Accuracy | Macro-F1 | OK recall | Flashover recall | Broken recall | Комментарий |
|---|---|---|---:|---:|---:|---:|---:|---|
| stage3_qwen_val_v2_clean_final | Stage 3 | Qwen baseline | 0.4655 | 0.4804 по 3 классам | 0.3438 | 0.6500 | 0.5000 | clean GT-crop baseline |
| stage4_context_pad030_maxpix401k | Stage 4 | Qwen baseline | 0.3966 | — | — | — | — | лучший Qwen-only context crop |
| stage3_clip_train_selected_clean | Stage 3 | CLIP coarse-only | 0.5345 | 0.3713 | 0.5938 | 0.6000 | 0.0000 | полезный coarse-сигнал, но нет broken recall |
| stage3_qwen25vl_3b_lora_masked_smoke_clean | Stage 3 | LoRA smoke | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | structured generation не прошла preflight |
| stage4_hybrid_dinov2_qwen_light_pad030 | Stage 4 | DINOv2+Qwen hybrid | 0.4828 | 0.4671 | 0.1875 | 0.9500 | 0.5000 | сильный flashover recall, но перекос в flashover |
| stage4_hybrid_dinov2_qwen_traincv_policy_pad030_secondbest035 | Stage 4 | DINOv2+Qwen hybrid | 0.5862 | 0.5922 | 0.4688 | 0.7000 | 0.8333 | текущий лучший промежуточный результат |
| stage3_dinov2_traincv_policy_clean_secondbest035 | Stage 3 | DINOv2 coarse-only | 0.5517 | 0.5766 | 0.4062 | 0.6500 | 1.0000 | coarse-only контроль для DINOv2 |
