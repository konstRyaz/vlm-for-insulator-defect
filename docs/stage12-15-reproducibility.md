# Stage12–15 reproducibility notes

## Stage12

Основные направления:
- risk/review routing;
- bad-crop safety;
- structured-output audit.

Ключевые скрипты:
- `stage12_train_dev_risk_models.py`
- `stage12_eval_risk_review_final_test.py`
- `stage12_eval_bad_crop_safety.py`
- `stage12_eval_bad_crop_safety_v2.py`
- `stage12_eval_structured_output_v2_pilot.py`
- `stage12_eval_structured_baselines.py`

## Stage13

Основные направления:
- trade-off/safety pareto;
- false-alarm triage;
- cost-sensitive utility;
- claim/multiview diagnostics.

Ключевые скрипты:
- `stage13_risk_review_budget_sweep.py`
- `stage13_safety_pareto_sweep.py`
- `stage13_cost_sensitive_utility.py`
- `stage13_eval_flashover_overclaim.py`

## Stage14

Основные направления:
- repeated stratified split robustness.

Ключевые скрипты:
- `stage14_build_resplit_dataset.py`
- `stage14_make_stratified_splits.py`
- `stage14_run_resplit_risk_review.py`
- `stage14_aggregate_robustness.py`
- `stage14_build_report.py`

## Stage15

Основные направления:
- safe VLM runner;
- E02 flashover overclaim checker;
- development-value experiments.

Ключевые скрипты:
- `stage15_safe_vlm_runner.py`
- `stage15_safe_path_resolver.py`
- `stage15_build_e02_full_manifest.py`
- `stage15_eval_e02_flashover_full.py`
- `stage15_eval_e02_flashover_posteval_v2.py`
