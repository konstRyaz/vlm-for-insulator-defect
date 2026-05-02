# Итог текущего clean-среза

Исследование дошло до воспроизводимого checkpoint: Stage 2 detector baseline заморожен, Stage 3/Stage 4 очищены от prompt-visible утечки, а последние ветки сравнивались на одном clean `val_v2` протоколе.

| Блок | Лучший/ключевой результат | Решение |
|---|---:|---|
| Stage 3 Qwen clean baseline | `27/58 = 0.4655` | оставить как VLM ceiling anchor |
| Stage 3 Qwen model-sweep control | `28/58 = 0.4828` | считать допустимой воспроизводимой вариацией |
| Stage 4 Qwen context pad 0.30 | `23/58 = 0.3966` | лучший pure-Qwen detector-to-VLM baseline |
| Stage 4 DINOv2+Qwen champion | `34/58 = 0.5862` | текущий основной результат |
| Frozen VLM sweep | InternVL3-2B acc `0.5517`, macro-F1 `0.2853` | не продвигать из-за macro/visibility/tag regressions |
| Qwen LoRA/SFT repair | acc `0.5172`, macro-F1 `0.1579` | не продвигать, class collapse |

Основной вывод остался прежним, но стал лучше подтверждён: на текущем срезе геометрия детектора не является главным узким местом. Основная ошибка живёт в семантическом различении `insulator_ok`, `defect_flashover` и `defect_broken` на crop-уровне.

Гибридная ветка полезна именно потому, что отделяет `coarse_class` от генерации отчётных полей. DINOv2-классификатор улучшает coarse decision, а Qwen сохраняет структурированный `vlm_labels_v1`-совместимый вывод. Это не production-ready система: текстовые поля Qwen могут быть не полностью согласованы с переопределённым `coarse_class`. Для исследовательской декомпозиции это приемлемо.
