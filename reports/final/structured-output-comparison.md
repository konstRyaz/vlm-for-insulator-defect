# Сравнение VLM по structured output

Помимо accuracy класса, в проекте отдельно сравнивались VLM как structured reporter. Поля на выходе: `coarse_class`, `visibility`, `visual_evidence_tags`, `needs_review`, короткое описание.

## Что сравнивали

На Stage3 разные VLM сравнивались по единому `vlm_labels_v1` контракту:

- parse success;
- schema validity;
- coarse accuracy / macro-F1;
- visibility accuracy / macro-F1;
- tag mean Jaccard для evidence tags.

## Ключевой вывод

Модель с лучшей raw class accuracy не обязательно является лучшим structured reporter.

Например:
- InternVL3-2B base дал лучшую raw accuracy около `0.5517`, но tag Jaccard был около `0.0330`.
- Qwen2.5-VL-3B дал accuracy около `0.4828`, зато был стабильнее по structured fields: parse/schema `1.0`, visibility macro-F1 около `0.5218`, tag mean Jaccard около `0.1977`.

## Что это значит

Evidence tags модели предсказывают не очень хорошо: даже лучшая модель по этому показателю дала tag mean Jaccard около `0.20`. Поэтому structured JSON -- отдельное направление оценки, но не сильное преимущество.


Основные таблицы:
- `reports/final/tables/stage3_vlm_backbone_comparison.csv`
- `reports/final/vlm-structured-output-comparison/stage3_vlm_backbone_comparison.csv`
- `reports/final/vlm-structured-output-comparison/vlm_backbone_comparison_report.md`
