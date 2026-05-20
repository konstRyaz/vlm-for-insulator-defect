# VLM Benefit Summary (2026-05-16)

Это актуальная сводка результатов Stage 12-15 по линии VLM-benefit.

Ключевая позиция:
- `DINOv2` остается основным closed-set классификатором.
- `VLM` используется как слой `risk/review/safety/development`, а не как замена классификатора.

Подтвержденные benefit-направления:
- quality review queue / risk routing;
- false-alarm triage (особенно `insulator_ok -> defect_flashover`);
- bad-crop/open-set safety gate.

Неподтвержденные направления:
- улучшение direct raw accuracy;
- стабильный top-k reranking benefit;
- надежные structured evidence tags/binary checklist.

Основные файлы:
- `VLM_BENEFIT_SUMMARY.md`
- `STAGE12_RISK_REVIEW_AND_SAFETY.md`
- `STAGE13_TRADEOFF_AND_BENEFIT_EXPANSION.md`
- `STAGE14_STRATIFIED_ROBUSTNESS.md`
- `DEVELOPMENT_VALUE_AXIS.md`
- `LIMITATIONS.md`
- `claims_table.csv`
- `main_results_table.csv`
- `artifact_index.csv`
- `STAGE15_DEVELOPMENT_PLAN.md`



## Stage15 E02
- See `FLASHOVER_OVERCLAIM_CHECKER_E02.md` and `e02_flashover_overclaim_checker/` for budget-matched evidence.
