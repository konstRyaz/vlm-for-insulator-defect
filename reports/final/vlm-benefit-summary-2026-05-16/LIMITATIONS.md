# Ограничения

1. Небольшой объём данных и чувствительность метрик к составу среза.
2. VLM не улучшила raw closed-set accuracy относительно DINOv2.
3. Top-k reranking не подтверждён как устойчивый источник прироста.
4. Structured evidence tags/checklist пока не дают надёжного standalone-claim.
5. E02 claim бюджет-специфичен: подтверждён в low-review режиме, но не как универсальное доминирование над margin-only.
6. E02 снижает автоматические ошибки через review-routing, но не выполняет автоисправление класса.
7. Нужна human/field validation причин и объяснений VLM.
8. Строгие safety-режимы могут давать высокий review-load на clean случаях.
