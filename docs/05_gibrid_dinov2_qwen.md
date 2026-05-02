# Гибридная ветка DINOv2 + Qwen

## Идея

Qwen-only pipeline хорошо формирует структурированный JSON, но слабее различает близкие визуальные классы. Поэтому проверена гибридная схема:

```text
crop -> DINOv2 features -> LogisticRegression -> coarse_class
crop -> Qwen2.5-VL -> visibility/tags/text
merge -> vlm_labels_v1
```

В этой схеме DINOv2 отвечает за `coarse_class`, а Qwen остаётся structured reporter для остальных полей.

## Лучший промежуточный вариант

`stage4_dinov2_packfix_secondbest035`

Параметры:

- backbone: `facebook/dinov2-base`;
- classifier: `LogisticRegression(C=0.03, class_weight=balanced)`;
- политика: если top-class — низкоуверенный `defect_flashover`, заменить на second-best class;
- threshold: `0.35`, выбран по train OOF-CV;
- crop: detector predicted boxes + context padding `0.30`;
- reporter: Qwen2.5-VL-3B, frozen.

## Результат

| Система | Correct | Accuracy | Macro-F1 3-class | OK recall | Flashover recall | Broken recall |
|---|---:|---:|---:|---:|---:|---:|
| Qwen Stage 4 baseline | 23/58 | 0.3966 | 0.3749 | 0.3438 | 0.5000 | 0.3333 |
| Hard DINOv2 hybrid | 28/58 | 0.4828 | 0.4671 | 0.1875 | 0.9500 | 0.5000 |
| DINOv2 train-CV second-best fallback | 34/58 | 0.5862 | 0.5922 | 0.4688 | 0.7000 | 0.8333 |

Парное сравнение с Qwen Stage 4 baseline:

| Показатель | Значение |
|---|---:|
| Улучшение correct | +11/58 |
| Helped cases | 21 |
| Hurt cases | 10 |
| Unchanged correct | 13 |
| Unchanged wrong | 14 |
| Exact sign-test p-value | 0.0708 |

## Интерпретация

Гибридная ветка даёт самый сильный практический сигнал на текущей выборке. При этом размер валидации мал, поэтому результат нужно проверять на расширенном наборе. Также важно отслеживать согласованность: если `coarse_class` приходит от DINOv2, а текстовые поля от Qwen, возможны противоречия между классом и описанием.
