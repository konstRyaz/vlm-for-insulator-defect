# Промежуточные результаты

Содержимое папки:

- `results-summary.md` — общий текстовый отчёт.
- `hybrid-system-analysis.md` — разбор лучшей DINOv2+Qwen ветки.
- `leaderboard.md` — компактная таблица основных запусков.
- `tables/` — CSV-таблицы для проверки метрик.
- `figures/` — графики из финального Stage 4 comparison package.

Главный результат папки: гибридная Stage 4-система `DINOv2 coarse classifier + Qwen structured reporter` улучшает Qwen-only baseline с `23/58` до `34/58` объектов на текущем clean validation slice.
