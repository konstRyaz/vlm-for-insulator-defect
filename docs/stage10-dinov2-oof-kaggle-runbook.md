# Stage10 DINOv2 OOF Kaggle Runbook

Цель: получить leakage-safe OOF top-k предсказания DINOv2 для development split и объединить их с test final predictions.

## Шаги
1. Подготовить входные таблицы `vlm_labels_v1` и `stage10_vlm_eval_reference.csv`.
2. Запустить `scripts/stage10_generate_dinov2_dev_oof.py` на Kaggle (CPU/GPU по доступности).
3. Сформировать unified predictions через `scripts/stage10_prepare_full_oof_plus_test_predictions.py`.
4. Пересобрать full table: `scripts/stage10_build_full_dataset_table.py`.
5. Проверить oracle-анализ: `scripts/stage10_analyze_full_dataset_oracle.py`.

## Критические проверки
- Для development использовать только OOF (не in-sample).
- Для test использовать только final predictions.
- Проверить join coverage и отсутствующие `record_id`.

## Артефакты
- `stage10_dinov2_full_oof_plus_test_predictions.csv`
- `stage10_full_dataset_table.csv`
- `stage10_full_oracle_summary.md`
