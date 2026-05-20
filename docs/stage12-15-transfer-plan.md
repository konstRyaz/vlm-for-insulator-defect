# План переноса Stage12–15 в чистый репозиторий

## Зачем переносить

В черновом репозитории накоплены поздние результаты Stage12–15, которые формируют итоговый научный вывод: VLM полезна не как прямой классификатор, а как слой проверки, риска и безопасности вокруг DINOv2. В чистом репозитории уже есть базовая структура, но не хватает финальной истории VLM-benefit, компактных артефактов и воспроизводимых скриптов.

## Что переносить

1. Финальные русскоязычные отчёты в `reports/final`.
2. Компактные CSV с claims и ключевыми метриками.
3. Подробную папку `reports/final/vlm-benefit-summary-2026-05-16/`.
4. Скрипты Stage12–15, которые воспроизводят risk-review, bad-crop safety, stratified robustness и E02 flashover checker.
5. Несколько методологических docs: split terminology, vlm_labels_v1 spec, detector-to-vlm contract.

## Что не переносить

1. Большие outputs.
2. Raw images.
3. Zip-пакеты.
4. Raw VLM outputs JSONL, если они тяжёлые.
5. Старые экспериментальные notebooks, не влияющие на финальную историю.
6. Кеши, временные файлы, IDE-файлы.

## Финальная структура

Рекомендуемая структура в clean repo:

```text
reports/
  final/
    vlm-benefit-summary.md
    vlm-benefit-development-value.md
    vlm-benefit-reference-rationale.md
    vlm-benefit-limitations.md
    vlm-benefit-artifacts/
      claims_table.csv
      main_results_table.csv
      artifact_index.csv
    vlm-benefit-summary-2026-05-16/
      ...
docs/
  stage12-15-transfer-plan.md
scripts/
  stage12_*.py
  stage13_*.py
  stage14_*.py
  stage15_*.py
```

## Главный итоговый тезис

DINOv2 остаётся основным классификатором. VLM даёт пользу как слой review/safety/control: помогает выбирать рискованные случаи, ловить ложные flashover-тревоги, отсеивать плохие crop и строить более практичный human-in-the-loop inspection workflow.
