# Stage 10: исследование VLM как проверяющего модуля поверх DINOv2

Stage 10 переводит фокус с открытой VLM-классификации на более ограниченную задачу: может ли VLM выбирать между несколькими кандидатами, которые уже предложил DINOv2. Идея в том, что VLM не должна заново генерировать класс с нуля, а должна визуально проверить top-1/top-2/top-3 варианты DINOv2 и помочь решить, какой из них выглядит более обоснованным.

## Что было собрано

Была собрана единая crop-level таблица на `240` записей:

- `182` development crop с честными out-of-fold top-k предсказаниями DINOv2;
- `58` test crop с финальными DINOv2-предсказаниями;
- labels и predictions состыкованы для всех `240/240` записей;
- missing labels: `0`;
- missing predictions: `0`.

Development-предсказания являются out-of-fold: каждый crop предсказан fold-моделью, которая не обучалась на этом crop. Это позволяет подбирать будущую VLM-policy на development split без утечки правильных ответов.

| Срез | n | top-1 accuracy | top-2 oracle | top-3 oracle | top-1 wrong | recoverable top-2 | recoverable top-3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| development | 182 | 0.9176 | 0.9945 | 1.0000 | 15 | 14 | 15 |
| test | 58 | 0.6552 | 0.9828 | 1.0000 | 20 | 19 | 20 |
| all | 240 | 0.8542 | 0.9917 | 1.0000 | 35 | 33 | 35 |

Главный вывод: почти все ошибки top-1 уже можно было бы исправить, если бы система умела правильно выбирать между top-2/top-3 кандидатами. Поэтому наиболее перспективная роль VLM здесь — не самостоятельная классификация, а визуальная проверка ограниченного набора вариантов.

## Типы исправимых ошибок

Основные recoverable пары:

| Правильный класс | DINOv2 top-1 | DINOv2 top-2 | n |
|---|---|---|---:|
| insulator_ok | defect_flashover | insulator_ok | 13 |
| defect_flashover | insulator_ok | defect_flashover | 9 |
| defect_broken | insulator_ok | defect_broken | 6 |
| insulator_ok | defect_broken | insulator_ok | 4 |

Это показывает, что главная сложность остаётся на границе `insulator_ok` vs `defect_flashover`. Также есть отдельный риск пропуска `defect_broken`.

## Простые baseline без VLM

Перед запуском VLM были проверены простые правила без VLM:

- review по margin;
- switch-to-top2 по margin;
- class-wise margin rules;
- random baselines.

Главный результат: простые правила на основе margin почти не извлекают oracle-потенциал. Лучшая policy, выбранная на development, дала только `+1` на development и `0` gain на test. Это означает, что одного DINOv2 score/margin недостаточно, чтобы уверенно выбирать между top-1/top-2/top-3.
