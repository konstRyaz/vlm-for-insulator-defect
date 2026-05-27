# Отчёт о сравнении VLM-моделей

Этот отчёт построен по таблицам сравнения frozen VLM. Протокол Stage 3 оставался фиксированным: модели получали один и тот же crop, один и тот же prompt и должны были выдать структурированный ответ в формате `vlm_labels_v1`.

## Stage 3: VLM как генератор структурированного описания

| model | parse | schema | acc | macro-F1 | visibility macro-F1 | tag Jaccard |
|---|---:|---:|---:|---:|---:|---:|
| `qwen25vl_3b_control` | 1.0000 | 1.0000 | 0.4828 | 0.2946 | 0.5218 | 0.1977 |
| `internvl3_2b_base` | 1.0000 | 1.0000 | 0.5517 | 0.2853 | 0.2949 | 0.0330 |
| `internvl3_2b_defect_recall` | 1.0000 | 1.0000 | 0.3966 | 0.2255 | 0.2949 | 0.0517 |
| `internvl3_2b_balanced_defect` | 1.0000 | 1.0000 | 0.5000 | 0.2316 | 0.2949 | 0.1580 |
| `llava_onevision_qwen2_0_5b` | 0.7931 | 0.2414 | 0.1207 | 0.0609 | 0.1021 | 0.0000 |
| `smolvlm2_2b_instruct` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0896 | 0.0000 |
| `smolvlm2_500m_video_instruct` | 0.6034 | 0.6034 | 0.3276 | 0.1134 | 0.0526 | 0.0000 |
| `phi35_vision_instruct` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Кандидаты из энергетической области и coarse-only модели

| candidate | status | eval mode | blocker |
|---|---|---|---|
| `TL-CLIP` | ожидается подтверждение доступности весов/кода | только coarse-классификатор | публичные веса/код локально не подтверждены |
| `PowerGPT` | related work до подтверждения runnable release | structured reporter, если получится запустить | публичный runnable inference path не подтверждён |
| `Power-LLaVA` | related work до подтверждения runnable release | structured reporter, если получится запустить | публичный runnable inference path не подтверждён |
| `PLVLDet` | related work или detector baseline | detector baseline | не является VLM-генератором структурированного описания |

## Интерпретация

Новая frozen VLM из этого прохода не была продвинута в Stage 4. `InternVL3-2B` улучшает обычную accuracy, но не улучшает macro-F1 и сильно проигрывает по качеству structured output: хуже visibility и evidence tags.

Иными словами, модель с лучшей обычной accuracy не обязательно лучше как генератор структурированного описания. В нашем случае `Qwen2.5-VL-3B` остаётся более стабильным structured reporter, потому что у него выше `visibility macro-F1` и `tag Jaccard`, хотя accuracy класса ниже, чем у `InternVL3-2B`.

