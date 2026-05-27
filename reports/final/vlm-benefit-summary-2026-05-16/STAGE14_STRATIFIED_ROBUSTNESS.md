# Stage14: stratified robustness check

## Зачем нужен Stage14

Historical split был несбалансированным: доля дефектов в development и test различалась. Поэтому была проверка, не является ли review/risk benefit артефактом старого split-а.

## Что сделали

Были построены repeated stratified splits с более нормальной долей классов. На этих split-ах заново оценивались risk/review модели и сравнивались DINO-only, VLM-only, DINO+VLM и простые baselines.

## Вывод

Основной benefit не развалился: DINO+VLM оставался лучше DINO-only для ранжирования общих ошибок и false alarms. Это усиливает вывод, что VLM даёт устойчивый review/risk signal, а не просто выигрывает из-за случайного состава test split.
