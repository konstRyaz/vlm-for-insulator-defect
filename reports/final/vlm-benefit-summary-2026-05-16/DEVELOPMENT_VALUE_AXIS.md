# Development Value Axis

## Позиционирование
Это не схема "ИИ вместо инспектора".
Это схема "ИИ как слой вокруг сильного DINOv2-классификатора, который экономит внимание эксперта и снижает стоимость ошибок".

## Основные development/product use-cases
1. Shadow-mode review queue:
   - лучшее ранжирование кейсов для ручной проверки.
2. False-alarm triage:
   - снижение ложных тревог, в том числе `OK -> flashover`.
3. Bad-crop safety:
   - отдельные operating modes (strict / balanced / lenient) под разные risk-профили.
4. Reviewer packets / report drafts:
   - ускорение ручного ревью и унификация карточек.
5. Cost-sensitive utility:
   - выбор policy под бизнес-стоимость review и ошибок.
6. Field/shadow pilot:
   - переход от офлайн-метрик к operational validation.

## Практический смысл
VLM-слой улучшает качество процесса инспекции даже там, где raw accuracy классификации не растет.
