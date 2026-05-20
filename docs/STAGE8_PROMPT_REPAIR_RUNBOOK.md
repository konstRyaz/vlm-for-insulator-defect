# Stage 8 prompt repair runbook

## 1. Применить patch

Распаковать архив в корень репозитория:

```text
vlm-for-insulator-defect-detection/
  prompts/
  scripts/
  notebooks/
  docs/
  README_STAGE8_PROMPT_REPAIR.md
```

Проверить, что `prompts` — папка:

```powershell
Get-Item prompts | Format-List FullName,PSIsContainer,Length
Get-ChildItem prompts
```

Ожидается:

```text
PSIsContainer : True
stage8_bad_crop_checker_prompt.txt
stage8_multi_crop_verifier_prompt.txt
stage8_retrieval_fewshot_prompt.txt
stage8_unified_verifier_prompt.txt
```

## 2. Закоммитить

```powershell
git add prompts scripts/validate_stage8_prompts_static.py scripts/validate_stage8_vlm_outputs_schema.py notebooks/stage8_prompt_repair_smoke_kaggle.ipynb docs/STAGE8_PROMPT_REPAIR_RUNBOOK.md README_STAGE8_PROMPT_REPAIR.md
git status
git commit -m "Repair Stage 8 VLM prompts and add validators"
git push
```

## 3. Запустить static audit локально или на Kaggle

```bash
python scripts/validate_stage8_prompts_static.py --repo . --out-dir stage8_prompt_static_audit
```

Ожидаемый результат:

```text
OK: no obvious static prompt problems found.
```

## 4. Запустить smoke notebook на Kaggle

Открыть:

```text
notebooks/stage8_prompt_repair_smoke_kaggle.ipynb
```

Он делает:

1. clone ветки;
2. static prompt audit;
3. маленький Stage 8 VLM-прогон;
4. schema validation raw outputs;
5. упаковку smoke-аудита.

Ожидаемый итог:

```text
OK: raw VLM outputs pass schema checks.
```

Скачать:

```text
/kaggle/working/stage8_prompt_repair_smoke_audit.tar.gz
```

## 5. Только после smoke test запускать полный Stage 8

Если smoke test чистый, можно запускать полный Stage 8 repair-run:

```bash
python repo/scripts/run_stage8_vlm_assisted_architectures.py   --config repo/configs/stage8_vlm_assisted_architectures.yaml   --output-root /kaggle/working/stage8_vlm_assisted_repair_results   --archive-path /kaggle/working/stage8_vlm_assisted_repair_results.tar.gz
```

## 6. Как интерпретировать

Если после исправленных промптов и чистого schema validation VLM всё ещё проигрывает no-VLM DINOv2, тогда отрицательный результат уже гораздо честнее: причина не в enum/prompts, а в самой архитектуре/модели/датасете.
