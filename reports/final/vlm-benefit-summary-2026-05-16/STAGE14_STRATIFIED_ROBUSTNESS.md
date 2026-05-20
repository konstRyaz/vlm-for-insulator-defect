# Stage14: Stratified Robustness

## Зачем нужен Stage14
Historical dev/test имели различия по долям дефектов, что могло искажать выводы по AUPRC и review-value.
Stage14 сделал repeated stratified resplits для проверки устойчивости benefit-а.

## Что делали
- Repeated stratified splits по 240 записям.
- Оценка `dino_only`, `dino_plus_vlm` и baselines по target-ам review/risk.
- Обязательная prevalence-aware отчетность:
  - `positive_prevalence`
  - `AUPRC`
  - `AUPRC - prevalence`

## Ключевые наблюдения
- В resplit test подвыборках размер около **60** (при `test_size=0.25`), доля `defect_vs_ok` в среднем около **0.2125**, что близко к overall.
- Для `general_error` `dino_plus_vlm` в среднем выше `dino_only` по `AUPRC-prevalence`:
  - `dino_only`: mean ~ **0.1649**
  - `dino_plus_vlm`: mean ~ **0.2066**
- Для `false_alarm`:
  - `dino_only`: mean ~ **0.1579**
  - `dino_plus_vlm`: mean ~ **0.3443**
- Для `ok_to_flashover_false_alarm`:
  - `dino_only`: mean ~ **0.2024**
  - `dino_plus_vlm`: mean ~ **0.4463**

## Вывод
Review/risk benefit `dino_plus_vlm` не сводится только к старому historical split и сохраняется на repeated stratified resplits, особенно на false-alarm related targets.
