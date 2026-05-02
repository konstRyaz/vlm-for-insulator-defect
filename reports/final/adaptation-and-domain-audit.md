# Адаптация и доменные VLM

## Qwen LoRA/SFT repair

LoRA/SFT repair был нужен, чтобы проверить, можно ли улучшить Stage 3 без смены архитектуры и без утечки. Запуск исправил раннюю проблему с невалидным выводом: full validation прошёл с `parse_success = 1.0` и `schema_valid = 1.0`.

| metric | value |
|---|---:|
| train samples | 96 |
| eval objects | 58 |
| coarse accuracy | 0.5172 |
| coarse macro-F1 | 0.1579 |
| visibility accuracy | 0.7931 |
| visibility macro-F1 | 0.2949 |
| tag mean Jaccard | 0.3261 |
| pred ambiguous rate | 0.0000 |

Главная проблема: модель схлопнулась к `insulator_ok`. На `defect_flashover` она корректно нашла только `1/20`, на `defect_broken` — `0/6`. Поэтому adapter не продвигается в Stage 4.

Этот результат полезен как отрицательный checkpoint: простая next-token LoRA/SFT учит стабильный JSON, но не чинит нужную coarse boundary. Если возвращаться к адаптации, нужна class-balanced цель или вспомогательный discriminative loss.

## TL-CLIP, PowerGPT, Power-LLaVA

Проверка доменных моделей была проведена как availability audit, а не как GPU benchmark. Причина простая: для этих кандидатов не найден воспроизводимый публичный inference path с проверяемыми весами.

| candidate | статус | решение |
|---|---|---|
| TL-CLIP | публичные runnable weights/code не найдены | оставить как related work; если появится release, тестировать coarse-only |
| PowerGPT | публичные weights/API/code не найдены | related work, не считать failed benchmark |
| Power-LLaVA | официальный runnable release не найден | related work, не подменять непроверенными fork/checkpoint |

TL-CLIP концептуально подходит для `crop -> coarse_class`, но не для полного `vlm_labels_v1` JSON reporter. PowerGPT и Power-LLaVA релевантны как доменные VLM, но пока не могут быть честно включены в воспроизводимое сравнение.

Таблицы: `tables/lora_sft_repair_metrics.csv`, `tables/lora_sft_repair_confusion.csv`, `tables/domain_specific_models_status.csv`.
