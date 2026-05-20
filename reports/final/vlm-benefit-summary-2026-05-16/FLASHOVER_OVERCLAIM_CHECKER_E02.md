# FLASHOVER_OVERCLAIM_CHECKER_E02

## Зачем нужен E02
E02 — это checker утверждения "на изображении действительно есть flashover", когда DINOv2 выдала `top1=defect_flashover`.
Цель: улучшить triage ложных `insulator_ok -> defect_flashover` срабатываний при малом бюджете review.

## Валидность full run
- Kernel: `stage15-e02-full-t4-v1`
- GPU: `Tesla T4`
- Model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Full manifest: `n=212`
- E02_CORE (`dino_top1 == defect_flashover`): `n=36`
- `image_load_success_rate=1.0`
- `parse_ok_rate=1.0`
- `schema_ok_rate=1.0`
- `runtime_error_rate=0.0`

## Budget-matched сравнение (review_count=4 из 36, 11.11%)
- `vlm_binary_rank`:
  - false_alarm_capture = `0.2308`
  - true_flashover_retention = `0.9524`
  - true_flashover_review_rate = `0.0476`
  - net_gain = `+2`
- `margin_rank`:
  - false_alarm_capture = `0.0769`
  - true_flashover_retention = `0.8571`
  - true_flashover_review_rate = `0.1429`
  - net_gain = `-2`

## Вывод
**SUPPORTED (low-review режим):** VLM-checker лучше margin-only отбирает ложные flashover-срабатывания при одинаковом малом review-бюджете.

Ограничение: это не claim о доминировании VLM на всех review-бюджетах и не claim об улучшении raw accuracy.
