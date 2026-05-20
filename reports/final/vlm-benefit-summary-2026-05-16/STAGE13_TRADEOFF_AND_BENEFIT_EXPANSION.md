# Stage13: Tradeoff and Benefit Expansion

## Что исследовали
- safety tradeoff: strict vs balanced vs lenient;
- false-alarm triage;
- cost-sensitive utility;
- detector/VLM integration hypotheses;
- claim verification / overclaim checker / multiview branch.

## Supported
- `risk/review` branch подтверждена (продолжает Stage12).
- `bad-crop safety` tradeoff подтвержден (strict снижает bad false accept, balanced снижает review-cost).
- `cost-sensitive utility` как operational framing поддержан (иллюстративный economic layer).

## Diagnostic / Proxy / Caveat
- Flashover overclaim checker: **DIAGNOSTIC_ONLY** для части ранних прогонов; использовать только реальные run artifacts.
- Claim verification: **DIAGNOSTIC_ONLY/PROXY_ONLY** в частях, где был mapping/remap вместо отдельного независимого inference цикла.
- Detector+VLM crop guard: **DIAGNOSTIC_ONLY**, если в конкретном варианте не были полноценно задействованы независимые detector-geometry признаки.

## Вывод
Stage13 усилил идею operational value, но часть архитектурных веток следует трактовать как диагностические, а не как финальные доказательства.
