# Сводка VLM-benefit (Supervisor-ready)

## Итоговая позиция
DINOv2 остаётся основным closed-set классификатором.
VLM не заменяет его по raw accuracy, но добавляет пользу как review/safety слой.

## SUPPORTED
1. Улучшение risk/review routing.
2. Улучшение false-alarm triage.
3. Low-review flashover overclaim checker (E02).
4. Bad-crop/open-set safety gate.

Ключевой E02 результат (subset `dino_top1 == defect_flashover`, бюджет review `4/36`):
- VLM false_alarm_capture = `0.2308` vs margin = `0.0769`
- VLM true_flashover_review_rate = `0.0476` vs margin = `0.1429`
- net_gain `+2` vs `-2`

## NOT SUPPORTED
1. Улучшение raw closed-set accuracy.
2. Универсальное превосходство в top-k reranking.
3. Стабильно надёжные structured evidence tags.
4. Универсальное доминирование над margin-only на всех review-бюджетах.

## Финальная формулировка
DINOv2 остаётся основным классификатором. VLM полезна как слой проверки и безопасности: лучше маршрутизирует случаи в review, помогает ловить ложные тревоги, даёт low-review flashover checker и улучшает bad-crop safety.
