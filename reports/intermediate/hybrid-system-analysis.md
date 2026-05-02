# Анализ гибридной системы DINOv2 + Qwen

## Сравниваемые варианты

1. `qwen_baseline` — Qwen-only Stage 4 на context crop.
2. `hard_dinov2` — жёсткая замена `coarse_class` на DINOv2-classifier.
3. `champ_secondbest_cv035` — DINOv2 classifier с train-CV политикой second-best fallback.

## Метрики

| run | correct | total | acc | macro3 | recall_insulator_ok | recall_defect_flashover | recall_defect_broken |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen_baseline | 23 | 58 | 0.3966 | 0.3749 | 0.3438 | 0.5000 | 0.3333 |
| hard_dinov2 | 28 | 58 | 0.4828 | 0.4671 | 0.1875 | 0.9500 | 0.5000 |
| qwen_veto_cv035 | 27 | 58 | 0.4655 | 0.4687 | 0.2500 | 0.8000 | 0.5000 |
| champ_secondbest_cv035 | 34 | 58 | 0.5862 | 0.5922 | 0.4688 | 0.7000 | 0.8333 |

## Парная разница с Qwen-only baseline

| candidate | baseline_correct | candidate_correct | delta_correct | helped | hurt | unchanged_correct | unchanged_wrong | sign_test_p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hard_dinov2 | 23 | 28 | 5 | 14 | 9 | 14 | 21 | 0.4049 |
| qwen_veto_cv035 | 23 | 27 | 4 | 11 | 7 | 16 | 24 | 0.4807 |
| champ_secondbest_cv035 | 23 | 34 | 11 | 21 | 10 | 13 | 14 | 0.0708 |

## Confusion matrix лучшего варианта

| gt | defect_broken | defect_flashover | insulator_ok | empty |
|---|---:|---:|---:|---:|
| defect_broken | 5 | 0 | 1 | 0 |
| defect_flashover | 1 | 14 | 5 | 0 |
| insulator_ok | 5 | 11 | 15 | 1 |

## Интерпретация

`hard_dinov2` сильно повышает flashover recall, но начинает слишком часто называть нормальные изоляторы flashover. Политика second-best fallback уменьшает этот перекос и улучшает баланс классов. Поэтому именно `champ_secondbest_cv035` является текущим лучшим промежуточным вариантом.

Ограничение: результат проверен на 58 объектах, поэтому для утверждения устойчивости нужен повтор на расширенной валидации.
