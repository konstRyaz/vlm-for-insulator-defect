# Ограничения текущих результатов

1. **VLM не улучшила прямую raw accuracy классификации дефектов.** Основной классификатор в финальной архитектуре — DINOv2.

2. **Structured evidence tags пока ненадёжны.** Evidence tags и binary checklist давали слабые метрики и склонность к лишним defect-evidence на нормальных изоляторах.

3. **Bad-crop safety имеет trade-off.** Strict VLM safety хорошо отсекает плохие crop, но может отправлять слишком много clean crop на review.