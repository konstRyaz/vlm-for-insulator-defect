# Исследование дефектов изоляторов ЛЭП с использованием детектора и VLM

Репозиторий содержит исследовательскую версию конвейера анализа изображений изоляторов:

```text
изображение -> детектор объекта -> crop изолятора -> VLM/гибридная модель -> структурированное описание дефекта
```

Цель проекта — оценить, где VLM реально добавляет пользу поверх классического визуального классификатора.

## Текущий статус

Ключевые зафиксированные результаты:

| Этап | Метод | Метрика | Значение |
|---|---|---:|---:|
| Stage 2 | Faster R-CNN, COCO-оценка | mAP@[0.50:0.95] | 0.5664 |
| Stage 2 | Faster R-CNN, COCO-оценка | mAP@0.50 | 0.7597 |
| Stage 3 | Qwen2.5-VL-3B на GT-crop | coarse accuracy | 0.4655 |
| Stage 4 | Qwen2.5-VL-3B на crop детектора (pad=0.30) | pipeline correct | 23/58 = 0.3966 |
| Stage 4 | DINOv2 coarse classifier + Qwen structured reporter | pipeline correct | 34/58 = 0.5862 |

Лучший промежуточный вариант — гибридная Stage 4 схема:

```text
crop детектора -> DINOv2 признаки -> LogisticRegression coarse_class -> Qwen2.5-VL структурирует остальные поля
```

## Основные документы

- `docs/01_problem_statement.md`
- `docs/02_data_and_format.md`
- `docs/03_detector_baseline.md`
- `docs/04_vlm_protocol.md`
- `docs/05_gibrid_dinov2_qwen.md`
- `docs/06_reproducibility.md`
- `docs/07_vlm_topk_reranker_protocol.md`

## Поздние результаты по VLM-benefit

DINOv2 остаётся основным closed-set классификатором. VLM не показала себя как лучшая замена классификатора по raw accuracy, но дала пользу как дополнительный review/safety слой поверх DINOv2.

Основные найденные преимущества VLM:
- выбор рискованных случаев для ручной проверки;
- фильтрация ложных тревог, особенно `insulator_ok -> defect_flashover`;
- low-review flashover overclaim checker;
- bad-crop / open-set safety gate.

Подробные отчёты:
- `reports/final/vlm-benefit-summary.md`
- `reports/final/vlm-benefit-development-value.md`
- `reports/final/vlm-benefit-reference-rationale.md`
- `reports/final/vlm-benefit-limitations.md`
- `reports/final/vlm-benefit-summary-2026-05-16/`
