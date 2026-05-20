# Короткая записка для руководителя

## Что проверяли
Мы проверяли, может ли VLM дать практическую пользу в задаче диагностики дефектов изоляторов, когда DINOv2 уже используется как основной классификатор.

## Что не подтвердилось
- VLM не стала лучшей заменой DINOv2 по raw accuracy.
- Top-k reranker не показал устойчивого универсального выигрыша.
- Structured evidence tags пока недостаточно стабильны для сильного claim.

## Что подтвердилось
1. VLM улучшает risk/review routing.
2. VLM помогает triage ложных тревог.
3. E02 flashover overclaim checker работает в low-review режиме: при одинаковом малом бюджете review лучше margin-only ловит `insulator_ok -> defect_flashover`.
4. VLM полезна для bad-crop/open-set safety gate.

## Почему это научно корректно
Мы не утверждаем, что VLM заменяет классификатор. Корректный вывод: DINOv2 — основной closed-set слой, VLM — дополнительный слой проверки, маршрутизации рисков и безопасности.

## Что делать дальше
- Провести field/shadow pilot.
- Калибровать safety tradeoff (capture vs review-load).
- Добавить human validation объяснений VLM.
- Перейти к asset-level сценариям.
