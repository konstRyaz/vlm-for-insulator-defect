# Stage 10: VLM как top-k reranker/checker

Этот документ фиксирует следующий исследовательский шаг после гибридной Stage 4-системы. Вместо open-ended классификации crop-изображения VLM рассматривается как ограниченный visual checker/reranker поверх кандидатов DINOv2.

## Мотивация

Полный crop-level join был собран для `240` объектов: `182` development-записи и `58` test-записей. Для development split используются out-of-fold DINOv2-предсказания, чтобы policy не подбиралась на in-sample ответах. Для test split используются final DINOv2-предсказания.

Главный сигнал:

| Срез | n | DINOv2 top-1 | top-2 oracle | top-3 oracle |
|---|---:|---:|---:|---:|
| development | 182 | 0.9176 | 0.9945 | 1.0000 |
| test | 58 | 0.6552 | 0.9828 | 1.0000 |
| all | 240 | 0.8542 | 0.9917 | 1.0000 |

Это означает, что ошибки DINOv2 чаще являются ошибками выбора top-1, а не ошибками генерации кандидатов: правильный класс почти всегда уже лежит в top-2/top-3.

## Методология split-ов

В VLM-экспериментах веса VLM не обновляются. Поэтому historical `train` используется как development split для подбора prompt/policy, а historical `val` используется как test split для финальной оценки.

Для supervised no-VLM classifier поверх frozen DINOv2 features первая часть данных является training/development split. Чтобы использовать её для выбора VLM-policy, нужны out-of-fold predictions: каждый crop должен быть предсказан моделью, которая этот crop не видела.

## Роль VLM

Проверяются три роли VLM:

- top-k reranker: выбрать лучший класс среди `dino_top1`, `dino_top2`, `dino_top3`;
- checker/verifier: подтвердить или опровергнуть визуальные evidence для top-1;
- review/safety layer: оставить top-1, переключить на другого кандидата или отправить случай на ручной review.


## Текущие non-VLM baselines

Простые margin-only политики на development дают очень слабый выигрыш: лучший dev-selected вариант улучшает development на один объект и не даёт gain на test. Это усиливает ценность следующей проверки VLM: если VLM сможет выбирать switch/review лучше margin-only правил, это будет отдельный полезный сигнал.

## Воспроизведение

Основные entrypoints:

```bash
python scripts/stage10_generate_dinov2_dev_oof.py ...
python scripts/stage10_build_full_dataset_table.py ...
python scripts/stage10_analyze_full_dataset_oracle.py ...
python scripts/stage10_eval_nonvlm_policy_baselines.py ...
python scripts/stage10_build_vlm_inference_manifest.py ...
```

Компактные таблицы находятся в `reports/intermediate/tables/` с префиксом `stage10_`.
