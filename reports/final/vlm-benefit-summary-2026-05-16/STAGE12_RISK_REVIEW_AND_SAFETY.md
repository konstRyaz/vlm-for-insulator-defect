# Stage12: Risk/Review and Safety

Stage12 фокусировался на benefit за пределами raw accuracy: review routing, false-alarm triage, safety gate.

## Risk/Review (historical test)
Источник: `outputs/stage12/risk_review_final_test_v2/*`

- `R0_dino_only` AUPRC (`general_error`): **0.414659**
- `R2_dino_plus_vlm` AUPRC (`general_error`): **0.545291**

Accepted accuracy:
- @10% review:
  - `R0_dino_only`: **0.673077**
  - `R2_dino_plus_vlm`: **0.711538**
- @20% review:
  - `R0_dino_only`: **0.673913**
  - `R2_dino_plus_vlm`: **0.739130**

## Bad-crop / Open-set safety
Источник: `outputs/stage13_tradeoff_benefit_expansion/E01_safety_pareto/policy_sweep_bad_vs_clean.csv`

- Closed-set baseline (`accept all`) bad false accept: **1.000000**
- Strict safety bad false accept: **0.013333**
- Current/Balanced policy bad false accept: **0.356667**

Вывод: safety-контур действительно снижает bad false accept, но strict режим имеет высокую цену по clean review.
