# Stage13: trade-off и расширение benefit-ов

Stage13 исследовал, как сохранить найденные преимущества VLM и уменьшить побочные эффекты.

## Главные выводы

1. Risk/review benefit подтвердился как наиболее полезное направление.
2. Bad-crop safety имеет trade-off: строгий режим безопаснее, но чаще отправляет clean crop на review.
3. False-alarm triage стал перспективным направлением, особенно для `insulator_ok -> defect_flashover`.
4. Некоторые направления остались proxy-only или diagnostic-only и не должны подаваться как доказанные claims.

## Ограничения

Flashover overclaim checker стал полноценным claim-supporting экспериментом только позже, после исправления safe VLM runner и полного E02 post-eval.
