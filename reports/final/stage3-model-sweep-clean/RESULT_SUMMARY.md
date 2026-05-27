# Stage 3: результат чистого сравнения Qwen-моделей

Этот прогон проверял frozen-модели семейства Qwen в чистом Stage 3-протоколе без передачи пути к crop в prompt. Датасет, prompt, схема вывода и evaluator были зафиксированы; менялась только модель.

Источник прогона: Kaggle kernel `kostyaryazanov/notebookd64e91cba0`, version 16.

## Настройка

- Датасет: clean test split, historical `val_v2`, 58 GT crop
- Prompt: `qwen_vlm_labels_v1_prompt_v7f_flashover_unclear_to_unknown_nocroppath`
- Max pixels: `401408`
- Формат вывода: `vlm_labels_v1`
- GPU: Kaggle T4

## Результаты

| model | full run | coarse acc | correct | coarse macro-F1 | visibility macro-F1 | tag Jaccard | OK recall | flashover recall | broken recall | verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen2.5-VL-3B-Instruct | yes | 0.4828 | 28/58 | 0.2946 | 0.5218 | 0.1977 | 0.3125 | 0.7500 | 0.5000 | контроль |
| Qwen2.5-VL-7B-Instruct | yes | 0.5000 | 29/58 | 0.1556 | 0.5593 | 0.4066 | 0.8750 | 0.0500 | 0.0000 | слабый сигнал, как есть не подходит |
| Qwen3-VL-4B-Instruct | yes | 0.5345 | 31/58 | 0.2748 | 0.5577 | 0.2876 | 0.8438 | 0.1000 | 0.3333 | слабый сигнал, сильный сдвиг по классам |
| Qwen2.5-VL-7B-Instruct-AWQ | no | - | - | - | - | - | - | - | - | preflight не пройден |
| Qwen3-VL-2B-Instruct | no | - | - | - | - | - | - | - | - | full run упал на проверке схемы |

## Интерпретация

Этот sweep не дал сильной замены базовой Qwen2.5-VL-3B. Qwen3-VL-4B и Qwen2.5-VL-7B немного улучшают обычную accuracy за счёт того, что намного чаще правильно предсказывают нормальные изоляторы. Но обе модели почти теряют класс `defect_flashover`, а это как раз один из главных проблемных случаев проекта.

Особенно показателен результат Qwen2.5-VL-7B. Модель даёт `29/58` правильных ответов, но recall для flashover падает до `1/20`, а recall для broken — до `0/6`. То есть увеличение размера модели не решает задачу дефектов в текущей постановке, а просто меняет профиль ошибок.

Qwen3-VL-4B показывает лучшую обычную accuracy: `31/58`, но её macro-F1 ниже, чем у контрольной 3B-модели, а flashover recall всего `2/20`. Поэтому эту модель нельзя продвигать в Stage 4 как финальную без дополнительного исправления границы между `insulator_ok` и `defect_flashover`.

## Решение

Не заменять чистый Stage 3 baseline на Qwen2.5-VL-7B или Qwen3-VL-4B в текущем виде.