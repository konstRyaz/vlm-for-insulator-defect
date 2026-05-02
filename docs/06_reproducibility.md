# Воспроизводимость

## Установка окружения

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Проверка импортов:

```bash
python scripts/smoke_imports.py
```

## Toy-запуск

```bash
python scripts/make_toy_coco.py --out_dir data/raw/toy_coco
python scripts/prepare_data.py --dataset coco --raw_dir data/raw/toy_coco --out_dir data/processed/toy_coco
bash scripts/smoke_run_toy.sh
```

## Основные группы скриптов

| Группа | Скрипты |
|---|---|
| Подготовка данных | `scripts/idid_to_coco.py`, `scripts/prepare_data.py`, `scripts/make_toy_coco.py` |
| Детектор | `src/train.py`, `src/eval.py`, `src/infer.py` |
| Crop/VLM | `scripts/export_vlm_crops.py`, `scripts/run_stage3_vlm_baseline.py`, `scripts/run_stage4_detector_to_vlm.py` |
| Оценка | `scripts/eval_stage3_vlm_baseline.py`, `scripts/eval_stage4_detector_to_vlm.py`, `scripts/bootstrap_eval_ci.py` |
| Визуализация | `scripts/visualize_stage3_eval_results.py`, `scripts/visualize_stage4_eval_results.py` |
| Гибрид | `scripts/hybrid_merge_qwen_reporter.py` |
| Финальный анализ | `scripts/analyze_stage4_paired_cases.py`, `scripts/audit_no_leak_stage3_stage4.py`, `scripts/build_stage4_visual_review.py` |

## Конфигурации

- `src/configs/` — конфигурации обучения/оценки detector baseline.
- `configs/pipeline/` — конфигурации detector-to-VLM и Stage 3/Stage 4.
- `configs/pipeline/prompts/` — prompt-файлы для VLM. Они сохранены в исходном виде, чтобы не ломать воспроизводимость уже проведённых запусков.

## Отчётные артефакты

Компактные сводки находятся в:

```text
reports/intermediate/
reports/final/
```

`final` содержит закрывающий clean-срез: frozen VLM comparison, LoRA/SFT repair, audit доменных моделей, paired Stage 4 champion analysis и no-leak audit.

Полные локальные outputs и датасеты не включаются в Git.
