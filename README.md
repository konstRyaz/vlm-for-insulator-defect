# Исследование дефектов изоляторов ЛЭП с использованием детектора и VLM

Репозиторий содержит промежуточную исследовательскую версию конвейера для анализа изображений изоляторов:

```text
изображение -> детектор объекта -> crop изолятора -> VLM/гибридная модель -> структурированное описание дефекта
```

Цель проекта — оценить, насколько связка классического детектора и визуально-языковой модели подходит для диагностики дефектов изоляторов. В текущей версии основной акцент сделан не на максимальном “production”-качестве, а на разложении ошибки по этапам и проверке нескольких исследовательских гипотез.

## Текущий статус

На текущем срезе зафиксированы четыре ключевых результата.

| Этап | Метод | Метрика | Значение |
|---|---|---:|---:|
| Stage 2 | Faster R-CNN, COCO-оценка | mAP@[0.50:0.95] | 0.5664 |
| Stage 2 | Faster R-CNN, COCO-оценка | mAP@0.50 | 0.7597 |
| Stage 3 | Qwen2.5-VL-3B на GT-crop | coarse accuracy | 0.4655 |
| Stage 4 | Qwen2.5-VL-3B на crop детектора с context pad 0.30 | pipeline correct | 23/58 = 0.3966 |
| Stage 4 | DINOv2 coarse classifier + Qwen structured reporter | pipeline correct | 34/58 = 0.5862 |

Лучший промежуточный вариант — гибридная Stage 4-система:

```text
crop детектора -> DINOv2-признаки -> LogisticRegression coarse_class -> Qwen2.5-VL структурирует остальные поля
```

Вариант `stage4_dinov2_packfix_secondbest035` улучшает результат с `23/58` до `34/58` объектов на той же валидационной выборке. При этом выборка мала, поэтому результат следует считать сильным промежуточным сигналом, а не финальным статистическим доказательством.

Дополнительные проверки frozen VLM, LoRA/SFT и доменных моделей сведены в `reports/final/`. Этот срез фиксирует, что текущий прирост даёт именно гибридная ветка, а не простая замена VLM или наивная LoRA-адаптация.

## Структура репозитория

```text
.
├── configs/        # конфигурации конвейера и промпты VLM
├── docs/           # методическая документация
├── reports/        # компактные промежуточные результаты, таблицы и графики
├── schemas/        # JSON Schema для структурированных VLM-ответов
├── scripts/        # скрипты подготовки данных, запуска и оценки
├── src/            # основной Python-код детектора и вспомогательных модулей
├── tools/          # локальный инструмент разметки/проверки VLM-меток
├── notebooks/      # очищенные экспериментальные notebook-запуски
├── data/           # локальные данные, не входят в Git
└── outputs/        # локальные результаты запусков, не входят в Git
```

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Для проверки импортов:

```bash
python scripts/smoke_imports.py
```

## Минимальная проверка на toy COCO

```bash
python scripts/make_toy_coco.py --out_dir data/raw/toy_coco

python scripts/prepare_data.py \
  --dataset coco \
  --raw_dir data/raw/toy_coco \
  --out_dir data/processed/toy_coco

bash scripts/smoke_run_toy.sh
```

## Основные документы

- `docs/01_problem_statement.md` — постановка задачи и классы.
- `docs/02_data_and_format.md` — формат данных и подготовка.
- `docs/03_detector_baseline.md` — базовый детектор.
- `docs/04_vlm_protocol.md` — VLM-протокол, Stage 3/Stage 4 и защита от утечек.
- `docs/05_gibrid_dinov2_qwen.md` — текущий лучший гибридный результат.
- `docs/06_reproducibility.md` — порядок воспроизведения.
- `docs/07_vlm_topk_reranker_protocol.md` — Stage 10: VLM как constrained top-k reranker/checker.

## ???????? ??????? ?? VLM-benefit

DINOv2 ???????? ???????? closed-set ???????????????.
VLM ???? ?????????? ?????? ??? review/safety ????: triage ???????? ???????, ????? ?????? ?????? ? bad-crop safety.
