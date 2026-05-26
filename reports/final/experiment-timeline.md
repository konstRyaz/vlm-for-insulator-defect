# Хронология экспериментов

Этапы ниже отражают развитие исследования от базового detector/crop pipeline до итоговой роли VLM как review/safety layer. Не все этапы являются самостоятельными финальными claims: часть была диагностической, часть — технической, часть — отрицательной. Финальные поддержанные выводы сформированы в Stage12–15.

| Stage | Смысл этапа | Что делали | Главный вывод |
|---|---|---|---|
| Stage 1 | Постановка задачи и данные | Разобрали IDID, структуру изображений, bbox-разметку, классы дефектов. | Нужно строить pipeline `image → detector → crop → classifier/VLM → structured output`. |
| Stage 2 | Конвертация данных и detector baseline | COCO-style conversion, detector-разметка, train/val/test, detector/crop pipeline. | Основа проекта — detector/crop-представление. |
| Stage 3 | Первые VLM и crop-level эксперименты | VLM на GT-crop, prompt sweep, `vlm_labels_v1`, structured JSON. | VLM может давать structured output, но как прямой classifier нестабильна. |
| Stage 4 | Detector-to-VLM и DINOv2 hybrid | Predicted detector crop, DINOv2 features + LogisticRegression/policy, Qwen structured reporter. | DINOv2-based hybrid сильнее прямой VLM классификации; старый champion около `34/58 = 0.5862`. |
| Stage 5 | Расширение no-VLM baseline и backbone sweep | Разные visual features/backbone, HF/timm, flashover binary, SVM/review варианты. | Frozen visual features + лёгкий classifier дают сильный baseline. |
| Stage 6 | Full-train / stress / rescue | Дожимали no-VLM baseline, стресс-тестировали VLM, искали rescue-benefit. | VLM плохо подавать как replacement classifier; лучше искать safety/review/explainability. |
| Stage 7 | VLM-assisted architectures | VLM verifier, DINO+Qwen verifier, review gate, contradiction checker, multi-crop, retrieval few-shot. | VLM override часто портит class accuracy, но reviewer/checker роль остаётся перспективной. |
| Stage 8 | Prompt repair и JSON validity | Исправляли prompt-ы, enum-форматы, schema validation, raw JSON. | Без строгого output schema нельзя честно оценивать VLM. |
| Stage 9 | Первые safety/review benefits | Policy sweep, bad-crop/open-set checker, испорченные crop. | Нашёлся сильный safety benefit: VLM может отказаться от плохого crop. |
| Stage 10 | Top-k oracle и reranker potential | Full-dataset table, DINOv2 OOF top-k predictions, top-2/top-3 oracle. | Oracle высок, но это потенциал; VLM reranking сам по себе не стал сильным claim. |
| Stage 11 | LoRA / fine-tuning feasibility | Проверяли возможность VLM LoRA/SFT как supervised направления. | Не стало главным результатом текущей работы. |
| Stage 12 | Risk-review, bad-crop safety, structured-output audit | Review routing, accepted accuracy, bad-crop safety, evidence tags/checklist. | Подтвердились review/risk и bad-crop benefits; structured tags слабые. |
| Stage 13 | Trade-off и расширение benefit-ов | Strict vs balanced safety, false-alarm triage, cost utility, claim/multiview diagnostics. | Укрепился нарратив VLM как risk/safety layer; часть направлений осталась diagnostic-only. |
| Stage 14 | Stratified split robustness | Repeated stratified splits для проверки устойчивости. | Review/risk benefit не развалился на более нормальных split-ах. |
| Stage 15 | Development-value и E02 flashover checker | Safe VLM runner, shadow review queue, E02 full inference/post-eval. | Поддержан low-review flashover overclaim checker. |

## Главная траектория

Исследование началось с попытки использовать VLM как прямой классификатор дефектов, но эксперименты показали, что DINOv2 baseline сильнее по raw accuracy. После этого фокус был перенесён на другую роль VLM: слой проверки, риска и безопасности. В этой роли VLM дала подтверждённые преимущества: выбор случаев для ручной проверки, фильтрация ложных тревог, отсеивание плохих crop и low-review проверка спорных flashover-предсказаний.
