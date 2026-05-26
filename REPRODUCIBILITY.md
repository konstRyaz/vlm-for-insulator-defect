# Воспроизводимость

## Статус

Этот репозиторий является clean/presentation версией исследования. Он содержит код, конфиги, финальные отчёты, компактные таблицы результатов и скрипты воспроизводимости. Он не является полным тяжёлым архивом всех runtime outputs.

## Что можно воспроизвести из репозитория

- подготовку COCO/toy data;
- базовые detector/crop pipeline проверки;
- Stage3/Stage4 evaluation при наличии данных и outputs;
- Stage10 top-k/oracle/reranker diagnostics при наличии prediction CSV;
- Stage12 risk-review / bad-crop / structured-output evaluations при наличии соответствующих компактных inputs;
- Stage14 stratified robustness при наличии full-dataset/risk tables;
- Stage15 E02 post-eval при наличии full manifest и VLM outputs.

## Что требует внешних данных

- raw IDID images;
- full crop images;
- Kaggle GPU runs для VLM inference;
- raw VLM responses;
- generated outputs, не включённые в clean repo.

## Почему heavy outputs не включены

В Git не включаются:
- `outputs/`;
- `raw_outputs.jsonl`;
- изображения;
- Kaggle runtime cache;
- zip/rar delivery packages.

Их отсутствие не означает, что эксперименты не проводились. Итоговые результаты зафиксированы в компактных CSV/MD в `reports/final/`.

## Минимальные проверки

```powershell
python scripts/smoke_imports.py
Get-ChildItem scripts -Filter *.py | ForEach-Object { python -m py_compile $_.FullName }
```

## Основной итог

Репозиторий воспроизводит методологию и содержит код для повторения ключевых расчётов, но для полного повторения всех чисел нужны внешние данные и Kaggle/VLM inference artifacts.
