# Stage 9 interpretation guide

## Split terminology

Stage 9 reuses completed Stage 8 outputs and does not retrain or rerun Qwen.
For VLM-assisted policies, prompt and policy choices should be described as
chosen on the development/validation split; the reported Stage 9 policy and
bad-crop metrics are computed on the test split. The VLM was not trained on the
historical `train` split.

## Main question

Ищем не только `overall accuracy`, а именно VLM-benefit.

## Где VLM может честно выиграть

### 1. Bad-crop / open-set safety

Closed-set no-VLM classifier почти всегда вынужден выдать один из известных классов.

VLM checker может сказать:

```json
{
  "is_usable_crop": false,
  "needs_review": true,
  "safe_action": "send_to_review"
}
```

Это является benefit, если:

```text
vlm_safe_behavior_rate высокий
vlm_false_accept_rate низкий
no_vlm_false_confident_known_class_rate высокий
```

### 2. Selective classification

VLM может не улучшить accuracy на всех примерах, но улучшить reliability:

```text
coverage < 1
accepted_accuracy > baseline_accuracy
review_rate разумный
dangerous_miss_rate не растёт
```

Это можно описывать как:

> VLM gate повышает точность автоматически принятых решений ценой отправки части примеров на ручную проверку.

### 3. Explanation / structured output

Даже если class accuracy не выше, VLM выдаёт:

- `evidence_tags`;
- `visibility`;
- `short_description`;
- `needs_review`.

No-VLM classifier сам по себе этого не даёт.

## Как не интерпретировать

Не писать:

> VLM лучше DINOv2 по классификации.

если raw metrics этого не подтверждают.

Корректнее:

> По raw class accuracy обученный DINOv2 baseline остаётся сильнее. VLM полезна как safety/review/explanation module.

## Какие таблицы нужны в отчёт

1. Overall class metrics:
   - A0 DINOv2;
   - A1/A2/A3 VLM-assisted;
   - A5/A6.

2. Selective/review metrics:
   - policy;
   - threshold;
   - coverage;
   - review_rate;
   - accepted_accuracy;
   - accepted_macro_f1;
   - dangerous_miss_rate.

3. Bad-crop:
   - n;
   - vlm_safe_behavior_rate;
   - vlm_false_accept_rate;
   - vlm_review_rate;
   - no_vlm_closed_set_false_confident_known_class_rate.

## Возможный итоговый вывод

> Stage 9 показал, что текущая VLM-вставка не улучшает прямую классификацию дефекта относительно обученного DINOv2 baseline. Однако VLM даёт отдельный benefit в safety/review сценарии: она способна фильтровать плохие crop и переводить сомнительные случаи в review, что недоступно closed-set classifier без отдельного open-set модуля.
