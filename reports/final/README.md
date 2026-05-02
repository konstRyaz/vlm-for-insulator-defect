# Итоговый исследовательский срез
Папка фиксирует последние компактные результаты после очистки протокола от утечки `crop_path`.

Содержимое:

- `results-summary.md` — общий итог текущей исследовательской ветки.
- `vlm-model-comparison.md` — сравнение frozen VLM на clean Stage 3.
- `adaptation-and-domain-audit.md` — LoRA/SFT, TL-CLIP, PowerGPT и Power-LLaVA.
- `stage4-final-analysis.md` — парное сравнение Qwen Stage 4 и DINOv2+Qwen champion.
- `tables/` — CSV/JSON таблицы для проверки чисел.
- `figures/README.md` — пояснение по локальной сборке визуального обзора helped/hurt случаев.

Главный вывод: лучший текущий результат даёт не замена frozen VLM и не наивный LoRA, а гибрид `DINOv2 coarse classifier + Qwen structured reporter`. Он поднимает Stage 4 с `23/58` до `34/58` корректных объектов на clean validation slice.
