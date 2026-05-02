# Сводка промежуточных результатов

## Краткий вывод

На текущем clean validation slice основной предел качества находится не в детекторе, а в семантическом распознавании состояния crop. Qwen-only Stage 4 даёт `23/58 = 0.3966`, а гибридный вариант DINOv2+Qwen повышает результат до `34/58 = 0.5862`.

## Основные числа

| Блок | Вариант | Результат |
|---|---|---:|
| Detector baseline | Faster R-CNN mAP@[0.50:0.95] | 0.5664 |
| Detector baseline | Faster R-CNN mAP@0.50 | 0.7597 |
| Detector baseline | Faster R-CNN AR@100 | 0.7385 |
| Stage 3 | Qwen2.5-VL-3B, GT crop | 0.4655 coarse acc |
| Stage 4 | Qwen2.5-VL-3B, predicted context crop | 23/58 = 0.3966 |
| Stage 4 hybrid | DINOv2 train-CV policy + Qwen reporter | 34/58 = 0.5862 |

## Сравнение Stage 4

| Система | Correct | Accuracy | Macro-F1 3-class | OK recall | Flashover recall | Broken recall |
|---|---:|---:|---:|---:|---:|---:|
| Qwen Stage 4 baseline | 23/58 | 0.3966 | 0.3749 | 0.3438 | 0.5000 | 0.3333 |
| Hard DINOv2 hybrid | 28/58 | 0.4828 | 0.4671 | 0.1875 | 0.9500 | 0.5000 |
| DINOv2 train-CV second-best fallback | 34/58 | 0.5862 | 0.5922 | 0.4688 | 0.7000 | 0.8333 |

## Что улучшилось

Гибридная система снижает переуверенное распознавание flashover в hard-DINOv2 варианте и одновременно лучше восстанавливает `defect_broken`. Это даёт более сбалансированную confusion matrix по трём основным классам.

## Что остаётся слабым

Главная нерешённая граница — `insulator_ok` против `defect_flashover`. Даже лучший вариант всё ещё относит часть нормальных изоляторов к flashover-like дефектам. Для финального исследования нужна расширенная валидация и отдельная работа с качеством `defect_broken`.
