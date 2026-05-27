# Stage15 E02: Flashover overclaim checker

## Смысл эксперимента

E02 проверяет узкую и практическую гипотезу: если DINOv2 предсказал `defect_flashover`, может ли VLM помочь понять, подтверждается ли это визуально.

Задача VLM здесь не "выбрать класс", а проверить конкретное утверждение:

```text
видны ли реальные признаки flashover на поверхности изолятора?
```

## Валидность прогона

Полный VLM inference был выполнен на `Qwen/Qwen2.5-VL-3B-Instruct` с нормальным JSON parsing:

```text
image_load_success_rate = 1.0
parse_ok_rate = 1.0
schema_ok_rate = 1.0
runtime_error_rate = 0.0
```

## Ключевой результат

На наборе `E02_CORE`, где `dino_top1 == defect_flashover`, при одинаковом малом бюджете ручной проверки `4/36`:

```text
VLM:
false_alarm_capture = 0.2308
true_flashover_retention = 0.9524
true_flashover_review_rate = 0.0476
net_gain = +2

margin-only:
false_alarm_capture = 0.0769
true_flashover_retention = 0.8571
true_flashover_review_rate = 0.1429
net_gain = -2
```