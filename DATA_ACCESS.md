# Данные и внешние артефакты

## Что хранится в clean repo

- код;
- конфиги;
- prompt templates;
- schema;
- финальные отчёты;
- компактные таблицы результатов.

## Что не хранится

- сырые изображения IDID;
- crop images;
- full `outputs/`;
- raw VLM JSONL responses;
- Kaggle runtime directories;
- zip/rar delivery packages.

## Почему

Эти файлы тяжёлые, промежуточные или зависят от runtime-среды. Для GitHub-репозитория они намеренно исключены.

## Как восстановить

1. Получить исходные данные IDID согласно правилам доступа к датасету.
2. Подготовить COCO/crop данные через scripts и docs репозитория.
3. Запустить соответствующие Kaggle notebooks/scripts для VLM inference.
4. Использовать компактные отчётные tables в `reports/final/` как reference для проверки результата.
