# Финальный Stage 4 анализ

Финальное парное сравнение смотрит на один и тот же набор из 58 GT objects. Baseline — Qwen Stage 4 с context pad 0.30. Candidate — DINOv2+Qwen champion `stage4_dinov2_packfix_secondbest035`.

| metric | value |
|---|---:|
| total objects | 58 |
| Qwen baseline correct | 23 |
| DINOv2+Qwen correct | 34 |
| delta correct | +11 |
| Qwen baseline rate | 0.3966 |
| DINOv2+Qwen rate | 0.5862 |
| helped | 21 |
| hurt | 10 |
| both correct | 13 |
| both wrong | 14 |
| sign-test p | 0.0708 |
| bootstrap delta 95% CI | [0.0000, 0.3793] |

Парный анализ показывает не только рост accuracy, но и характер trade-off. Гибрид заметно помогает на дефектных классах, особенно там, где Qwen-only reporter колебался между normal и flashover/broken. Цена — часть normal crops начинает уходить в defect classes.

No-leak audit по текущим champion-артефактам прошёл: `30` файлов просканировано, `0` prompt-visible hits. Это не означает, что в артефактах вообще нет class/path строк: такие строки допустимы в manifests, predictions и case tables. Критично именно то, что они не попадают в prompt-visible input.

Для ручного просмотра helped/hurt случаев сохранён HTML:

```text
reports/final/figures/README.md
```

Решение: зафиксировать DINOv2+Qwen как текущий основной исследовательский результат. Дальше улучшения стоит искать не в broad prompt tuning, а в confidence/review policy, расширении validation/test slice или более целевой class-balanced adaptation.
