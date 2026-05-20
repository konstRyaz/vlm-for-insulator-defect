# Итоговый исследовательский срез

Папка фиксирует компактные финальные материалы после очистки протокола от утечки `crop_path` и стабилизации Stage12–15.

Содержимое:

- `results-summary.md` — общий итог исследовательской ветки.
- `vlm-model-comparison.md` — сравнение frozen VLM на clean Stage 3.
- `adaptation-and-domain-audit.md` — LoRA/SFT, TL-CLIP, PowerGPT и Power-LLaVA.
- `stage4-final-analysis.md` — парное сравнение Qwen Stage 4 и DINOv2+Qwen champion.
- `tables/` — компактные CSV/JSON с ключевыми метриками.

Главный вывод: лучший текущий результат даёт гибрид `DINOv2 coarse classifier + Qwen structured reporter`.

## VLM как слой проверки, риска и безопасности

- `vlm-benefit-summary.md` — краткий итог найденных преимуществ VLM.
- `vlm-benefit-development-value.md` — практическая польза VLM для development/integration.
- `vlm-benefit-reference-rationale.md` — интуитивное и референсное обоснование.
- `vlm-benefit-limitations.md` — ограничения и неподтверждённые claims.
- `vlm-benefit-summary-2026-05-16/` — подробные Stage12–15 отчёты и компактные артефакты.
