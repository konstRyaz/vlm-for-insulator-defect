# Данные и формат

## Формат входных данных

Скрипты рассчитаны на COCO-подобную структуру:

```text
raw_dir/
├── train/
│   ├── images/
│   └── annotations.json
├── val/
│   ├── images/
│   └── annotations.json
└── test/
    ├── images/
    └── annotations.json
```

`test` может отсутствовать на промежуточном этапе.

## Подготовка

Для подготовки данных используется:

```bash
python scripts/prepare_data.py \
  --dataset coco \
  --raw_dir data/raw/<dataset> \
  --out_dir data/processed/<dataset>
```

Для проверки pipeline без реального датасета есть toy COCO:

```bash
python scripts/make_toy_coco.py --out_dir data/raw/toy_coco
```

## Crop-артефакты

Crop для VLM экспортируются отдельными скриптами. Важно, чтобы путь к crop не содержал class-coded подсказок. В отчётных VLM-запусках используется `_nocroppath`-протокол: модель получает изображение и текстовую инструкцию, но не получает путь к файлу как часть prompt.

## Почему данные не лежат в репозитории

Полные изображения, crop-архивы и локальные outputs могут быть тяжёлыми и зависят от окружения. Поэтому в репозитории сохраняются только код, конфигурации, схемы и компактные отчётные таблицы.
