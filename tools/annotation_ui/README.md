# Локальный интерфейс проверки VLM-меток

Минимальный браузерный интерфейс для ручной проверки JSONL-записей формата `vlm_labels_v1`.

## Возможности

- просмотр одного crop за раз;
- отображение служебных полей записи;
- редактирование visibility, tags, краткого описания и заметок;
- фильтры, переходы назад/вперёд и горячие клавиши;
- безопасное сохранение в sidecar-файл.

## Установка

```bash
pip install -r tools/annotation_ui/requirements.txt
```

## Запуск

```bash
python tools/annotation_ui/app.py \
  --input outputs/stage3_pilot_mini/val/vlm_labels_v1_pilot.jsonl \
  --host 127.0.0.1 \
  --port 8501
```

Открыть в браузере:

```text
http://127.0.0.1:8501
```

## Горячие клавиши

- `Left` — предыдущая запись;
- `Right` — следующая запись;
- `1` — visibility `clear`;
- `2` — visibility `partial`;
- `3` — visibility `ambiguous`;
- `Ctrl+S` — сохранить.
