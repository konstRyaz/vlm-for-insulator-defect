# Сравнение frozen VLM

Сравнение frozen VLM закрывает вопрос: можно ли просто заменить Qwen2.5-VL-3B на другую открытую VLM и получить лучший структурированный Stage 3 reporter без дообучения.

Протокол был фиксирован: clean `val_v2` GT crops, тот же `vlm_labels_v1` contract, тот же evaluator, без `crop_path` и class-like filename hints в prompt.

| model | parse | schema | acc | macro-F1 | visibility macro-F1 | tag Jaccard | решение |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-VL-3B control | 1.0000 | 1.0000 | 0.4828 | 0.2946 | 0.5218 | 0.1977 | baseline anchor |
| InternVL3-2B base | 1.0000 | 1.0000 | 0.5517 | 0.2853 | 0.2949 | 0.0330 | не продвигать |
| InternVL3-2B defect-recall | 1.0000 | 1.0000 | 0.3966 | 0.2255 | 0.2949 | 0.0517 | overcall defects |
| InternVL3-2B balanced | 1.0000 | 1.0000 | 0.5000 | 0.2316 | 0.2949 | 0.1580 | low defect recall |
| LLaVA-OneVision 0.5B | 0.7931 | 0.2414 | 0.1207 | 0.0609 | 0.1021 | 0.0000 | schema/semantics fail |
| SmolVLM2 2.2B | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0896 | 0.0000 | parse fail |
| SmolVLM2 500M | 0.6034 | 0.6034 | 0.3276 | 0.1134 | 0.0526 | 0.0000 | class collapse |
| Phi-3.5-Vision | — | — | — | — | — | — | generic pipeline incompatible |

InternVL3-2B дал лучший raw accuracy, но не улучшил macro-F1, visibility и evidence tags. Поэтому он не стал новым Stage 4 reporter. LLaVA-OneVision и SmolVLM2 не удержали структурированный контракт достаточно надёжно.

Вывод: broad frozen VLM swap на этом этапе не дал лучшего reporter. Следующий полезный путь — либо гибридный coarse classifier, либо более аккуратная domain adaptation, но не дальнейший широкий перебор frozen VLM.

Подробные таблицы: `tables/stage3_vlm_backbone_comparison.csv` и `tables/vlm_backbone_paired_summary.csv`.
