# VLM for Insulator Defect Detection

Итоговый чистый репозиторий исследования по применению vision-language models для анализа дефектов изоляторов.

## Краткий итог

В проекте проверялась идея использования VLM в pipeline инспекции изоляторов:

```text
изображение
→ detector / crop
→ DINOv2 classifier
→ VLM review/safety layer
→ автоматическое решение или ручная проверка
```

Главный вывод: **VLM не стала лучшей заменой DINOv2-классификатора по raw accuracy**, но дала пользу в другой роли — как дополнительный слой проверки, риска и безопасности поверх DINOv2.

DINOv2 отвечает на вопрос:

```text
какой класс?
```

VLM помогает ответить на вопрос:

```text
можно ли доверять этому автоматическому решению без человека?
```

## Основные найденные преимущества VLM

1. **Проверка рискованных случаев**
   VLM помогает выбирать, какие случаи отправить на ручную проверку. В экспериментах DINO+VLM улучшил AUPRC для поиска ошибок примерно с `0.41` до `0.54`, а accepted accuracy при `10% review` выросла примерно с `0.6731` до `0.7115`.

2. **Фильтрация ложных тревог**
   VLM помогает находить случаи, где DINOv2 слишком резко объявляет дефект на нормальном изоляторе, особенно ошибки вида `insulator_ok -> defect_flashover`.

3. **Low-review flashover overclaim checker**
   На subset, где DINOv2 предсказал `defect_flashover`, при одинаковом малом бюджете review `4/36` VLM поймала больше ложных тревог, чем margin-only: `0.2308` против `0.0769`, и лучше сохранила настоящие flashover: `0.9524` против `0.8571`.

4. **Bad-crop / open-set safety gate**
   Обычный closed-set classifier обязан выбрать класс даже на плохом crop. VLM может сказать, что crop непригоден для автоматической классификации. В экспериментах closed-set baseline имел bad-crop false accept около `1.0`, а строгий VLM safety-режим снижал его примерно до `0.0133`.

5. **Development / review layer**
   VLM полезна для построения более практичного human-in-the-loop pipeline: review flags, reason codes, bad-crop flags, false-alarm checks и будущие карточки инспектора.

## Главные отчёты

- `reports/final/vlm-benefit-summary.md` — краткий итог найденных преимуществ.
- `reports/final/vlm-benefit-development-value.md` — практическая польза VLM для development/integration.
- `reports/final/vlm-benefit-reference-rationale.md` — интуитивное и референсное обоснование.
- `reports/final/vlm-benefit-limitations.md` — ограничения и неподтверждённые claims.
- `reports/final/vlm-benefit-summary-2026-05-16/` — подробные Stage12–15 отчёты и компактные артефакты.

## Что не является главным claim

- VLM не улучшила прямую closed-set classification accuracy.
- VLM не стала надёжным top-k reranker.
- Structured evidence tags пока недостаточно надёжны.
- E02 flashover checker поддержан именно как low-review/high-precision triage, а не как универсальное доминирование над margin-only на всех review budgets.

## Структура

```text
docs/                         методологические документы и runbook-и
reports/final/                финальные русскоязычные отчёты
reports/final/tables/         компактные таблицы старых Stage3/4 результатов
reports/final/vlm-benefit-*   финальная история Stage12–15 по VLM-benefit
scripts/                      воспроизводимые скрипты экспериментов
src/                          основной код проекта
configs/                      конфиги pipeline/model runs
notebooks/                    только ключевые notebooks/runbook scripts
```

## Не включается в чистый репозиторий

В чистый репозиторий не должны попадать тяжёлые runtime-артефакты:

```text
outputs/
raw_outputs.jsonl
data/raw/**/images
data/processed/**/images
*.zip
*.rar
__pycache__/
*.pyc
kaggle_upload/
kaggle_runs/
```

## Финальные материалы по истории экспериментов
- `reports/final/experiment-timeline.md` — хронология Stage1–15.
- `reports/final/structured-output-comparison.md` — сравнение VLM по JSON/schema/visibility/evidence tags.
- `REPRODUCIBILITY.md` — что воспроизводится из clean repo и какие внешние данные нужны.
- `DATA_ACCESS.md` — данные и исключённые heavy artifacts.
