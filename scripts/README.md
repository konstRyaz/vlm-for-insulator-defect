# Скрипты

Основные entrypoint-группы:

- подготовка данных: `make_toy_coco.py`, `idid_to_coco.py`, `prepare_data.py`;
- детектор: запуск через `src/train.py`, `src/eval.py`, `src/infer.py`;
- VLM/crop: `export_vlm_crops.py`, `run_stage3_vlm_baseline.py`, `run_stage4_detector_to_vlm.py`;
- оценка: `eval_stage3_vlm_baseline.py`, `eval_stage4_detector_to_vlm.py`, `bootstrap_eval_ci.py`;
- визуализация: `visualize_stage3_eval_results.py`, `visualize_stage4_eval_results.py`;
- гибридная ветка: `hybrid_merge_qwen_reporter.py`;
- финальный анализ: `analyze_stage4_paired_cases.py`, `audit_no_leak_stage3_stage4.py`, `build_stage4_visual_review.py`.
