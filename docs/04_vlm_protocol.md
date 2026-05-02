# VLM-протокол и оценка Stage 3/Stage 4

## Структурированный выход

VLM должна возвращать JSON по схеме `schemas/vlm_labels_v1.schema.json`. Основные поля:

- `coarse_class`;
- `visual_evidence_tags`;
- `visibility`;
- `short_canonical_description_en`;
- `report_snippet_en`.

Схема нужна, чтобы отделить качество визуального решения от качества свободного текста.

## Stage 3

Stage 3 использует crop по ground-truth bbox. Это верхняя оценка качества VLM-блока при почти идеальной локализации.

Основной Qwen baseline:

- модель: `Qwen/Qwen2.5-VL-3B-Instruct`;
- prompt family: `qwen_vlm_labels_v1_prompt_v7f_flashover_unclear_to_unknown_nocroppath`;
- coarse accuracy: `0.4655`;
- parse/schema success: `1.0000 / 1.0000`.

## Stage 4

Stage 4 использует crop по предсказанию детектора. Это end-to-end сценарий.

Лучший Qwen-only вариант:

- detector predicted crop;
- context padding `0.30`;
- `max_pixels=401408`;
- pipeline correct: `23/58 = 0.3966`.