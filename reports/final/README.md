# Финальные отчёты

Эта папка содержит отчётные материалы, которые можно использовать для презентации проделанной работы.

## Основные файлы

- `results-summary.md` — базовая сводка по ранним этапам и Stage3/4.
- `stage4-final-analysis.md` — итоговый анализ Stage4.
- `vlm-model-comparison.md` — сравнение VLM/backbone-подходов, если присутствует в репозитории.

## VLM как слой проверки, риска и безопасности

- `vlm-benefit-summary.md` — краткий итог найденных преимуществ VLM.
- `vlm-benefit-development-value.md` — практическая польза VLM для development/integration.
- `vlm-benefit-reference-rationale.md` — интуитивное и референсное обоснование.
- `vlm-benefit-limitations.md` — ограничения и неподтверждённые claims.
- `vlm-benefit-artifacts/` — компактные таблицы с claims и ключевыми результатами.
- `vlm-benefit-summary-2026-05-16/` — подробные Stage12–15 отчёты и компактные артефакты.

## Итоговая позиция

DINOv2 остаётся основным классификатором. VLM полезна не как replacement classifier, а как дополнительный review/safety слой:
- выбирает рискованные случаи для ручной проверки;
- помогает ловить ложные тревоги;
- проверяет спорные flashover-предсказания;
- отсекает плохие crop;
- делает pipeline более практичным для human-in-the-loop сценария.

## Дополнительные финальные отчёты
- `experiment-timeline.md` — хронология Stage1–15.
- `structured-output-comparison.md` — сравнение VLM по JSON/schema/visibility/evidence tags.
- `../../REPRODUCIBILITY.md` — воспроизводимость и ограничения clean repo.
- `../../DATA_ACCESS.md` — доступ к данным и исключённые heavy artifacts.
