# Stage12: risk/review и bad-crop safety

Stage12 показал, что VLM полезнее рассматривать не как прямой классификатор, а как дополнительный сигнал риска и безопасности.

## Risk/review

На test DINO+VLM улучшил ранжирование ошибок относительно DINO-only:

```text
DINO-only AUPRC ≈ 0.4147
DINO+VLM AUPRC ≈ 0.5453
```

Accepted accuracy при `10% review` выросла примерно:

```text
0.6731 -> 0.7115
```

## Bad-crop safety

Обычный closed-set classifier вынужден выбрать класс даже на плохом crop. В bad-crop stress тесте closed-set false accept был около `1.0`, а strict VLM safety снижал false accept примерно до `0.0133`.

## Вывод

VLM полезна как safety/review layer: она помогает не принимать автоматические решения на сомнительных или плохих входах.
