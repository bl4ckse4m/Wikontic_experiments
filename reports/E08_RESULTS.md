# E08 — сравнение готовых пайплайнов

Статус: **incomplete; no winner**. Run: `e08_main_judging_20261101T202207_663551Z`.
UTC сборки отчёта: 2026-11-01T21:38:32.042132+00:00.

## Условия

50 фиксированных tuning-абзацев MuSiQue-Ans v1.0 train. Четыре single-pass метода и четыре варианта кросс-промпт ансамбля; существующие predictions не перегенерировались.
Stateless OpenRouter API, `anthropic/claude-sonnet-5.5`, reasoning=low, только Anthropic без fallback; два сообщения system/user без tools и истории. 886 уникальных фактов и 200 пар, один проход в фиксированном порядке.
Source-first gold и человеческой проверки нет. Это **judged precision**, не независимая точность и не recall. Calibration в агентном интерфейсе не валидирует API-harness.

## Общая таблица

Интервалы — исходные approximate strata-Wilson/Bonferroni envelopes, не строгие тесты превосходства. В availability остаются все 50 входов; ошибки/пустые выдачи не исключаются.

| Метод | ok/failed/pending | Empty ok | Inputs с facts / 50 | Occurrences N | Размечено / sampled | Full P [interval] | Core P [interval] | N / input | Оценка supported occurrences / input | Chars / input |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| baseline_fixed | 50/0/0 | 0 | 50 | 543 | 101/150 | withheld | withheld | 10.86 | withheld | 2048 |
| coverage | 50/0/0 | 0 | 50 | 580 | 94/150 | withheld | withheld | 11.60 | withheld | 2165 |
| fewshot | 50/0/0 | 0 | 50 | 553 | 93/150 | withheld | withheld | 11.06 | withheld | 2124 |
| autoontik | 50/0/0 | 0 | 50 | 568 | 100/150 | withheld | withheld | 11.36 | withheld | 2104 |
| heuristic_union | 50/0/0 | 0 | 50 | 1227 | 102/150 | withheld | withheld | 24.54 | withheld | 4751 |
| heuristic_vote2 | 50/0/0 | 1 | 49 | 326 | 104/150 | withheld | withheld | 6.52 | withheld | 1291 |
| llm_union | 47/3/0 | 0 | 47 | 815 | 116/150 | withheld | withheld | 16.30 | withheld | 3114 |
| llm_vote2 | 47/3/0 | 0 | 47 | 363 | 108/150 | withheld | withheld | 7.26 | withheld | 1358 |

Supported occurrences — оценка числа правильных **выходных вхождений**, включая смысловые дубли; не уникальные факты и не покрытие. Chars — одна и та же JSON-сериализация исходных fact fields, суммарно по всем входам.

## Отдельный аудит LLM-merger v2

- proposed_merge: sampled=100, missing=25; оценка withheld.
- proposed_separate: sampled=100, missing=41; оценка withheld.

Пары отбирались pair-uniform внутри proposed-merge/separate на успешных merger partitions. Это условные ошибки эквивалентности, не extraction recall. Эвристический partition отдельно по этим парам не оценивался, поэтому аудит сам по себе не доказывает превосходства v2.

## Исполнение и ресурсы

- Состояния API batches: `{"ok": 69, "failed": 1, "pending": 30}`; labels: `{"binary": 579, "merger_audit": 134}`.
- Reported judge cost: $1.233184; jobs с неизвестной стоимостью: 1.
- Reported judge usage: `{"prompt_tokens": 217677, "completion_tokens": 79783, "total_tokens": 297460}`; provider metadata: `{"Anthropic": 69, "unreported": 1}`.
- Execution ledger: 70 attempts и 70 key checks; frozen key guard margin=$0.50. Лимит ключа не повышался; стоимость не критерий выбора.
- Single methods — 1 extraction/вход; heuristic ensemble — 3; LLM ensemble — 3 extraction + 1 merger/вход. Source usage по выбранным входам — в metrics.json; старые replacements/retries не полностью реконструированы, отсутствующая цена не равна нулю.
- В metrics.json отдельно сохранены доступные attempts и latency по retained records; сумма latency не равна wall-clock при исторической параллельной обработке. Union/vote2 повторно используют одни extraction/merger ресурсы: строки методов нельзя суммировать как независимые расходы.

## Изменения исполнения до labels и ограничения

API delivery заменяет файловые агентные сессии. Научные денежные caps и provider price filter удалены; введена отдельная техническая защита остатка ключа. Содержание rubric, samples, batches/order, модель и output limits сохранены; hashes и проверка эквивалентности запросов находятся в run artifacts.
Первый batch основной очереди использован как технический smoke, затем продолжены только неотправленные batches. Дополнительных quality-conditioned retries, нового extractor/merger или ручных repairs labels нет.
Результаты ограничены tuning subset, historical route/retry differences (включая AutoOntik), непроверенным silver judge, общими labels и зависимостью фактов внутри абзаца. Target precision≥0.95 не сертифицирована даже при высокой точечной оценке. Старые matcher-зависимые P/R/F1 не используются как доказательство улучшения.

## Артефакты

- `runs/e08_main_judging_20261101T202207_663551Z/metrics.json`, `binary_precision.json`, `merger_audit.json`, `api_final_status.json`.
- `data/experiments/e08/api_judging/`: фиксированный plan/requests, append-only attempts/responses и sanitized key checks.
- `data/judging/e08/{binary,merger_audit}/judgments.jsonl`: сырые содержательные labels из сохранённых API-ответов.
- Исходные selection/rubric/prediction locks сохранены. Публичный model catalog, forecast, версии, команды, hashes исходного/адаптированного кода и stdout/stderr — в run directory.

## Граница решения

При неполных labels E08 не завершён, победитель не выбирается. При полных labels основные contrasts — AutoOntik vs baseline_fixed и llm_union vs heuristic_union; близкие точечные оценки не образуют надёжный рейтинг. Автоматического запуска collection-1000 или расширения эксперимента нет.

## Фактическая остановка и решение

**Наблюдение.** 70-й POST вернул HTTP **429**, `limit_source=openrouter_admission_control`:
«OpenRouter could not verify available credits for this request in time. Retry shortly.»
Provider для этой ошибки не указан. Это не ответ судьи; batch из 32 binary facts
сохранён как failed, без labels. По замороженному правилу HTTP/transport failures
остановлены также следующие 30 batches. Ошибочный POST не повторялся.
Диагностика: `runs/e08_main_judging_20261101T202207_663551Z/stop_diagnostic.json`.

После остановки read-only `/key` показал daily usage **$1.233184**, remaining
**$3.766816**. Это совпало с суммой reported costs 69 успешных ответов на момент
snapshot; окончательную стоимость failed POST и задержки billing это не доказывает.
Неизвестная стоимость сохранена явно; conservative accounting с reservation
failed batch = **$1.377076**, это не счёт. Источник:
`runs/e08_main_judging_20261101T202207_663551Z/key_status_after_stop.json`.

**Интерпретация.** Получена ошибка сервиса проверки кредитов, а не подтверждение
достижения лимита $5. Данных недостаточно для причинного диагноза admission control.
Все восемь precision estimates и обе pair-error estimates withheld по исходным
правилам missingness; не вычислялись показатели только на удобном завершённом префиксе.

**Решение.** E08 не завершён; ни один метод не выбран и не отвергнут по качеству.
Существующие labels и locks сохранены, collection-1000 не запускается.
Следующий шаг — устранить/прояснить admission-control сбой и отдельно зафиксировать
допустимый порядок продолжения. Текущий terminal halt не снимается автоматически;
повтор failed batch или смена судьи не входят в выполненный протокол. Продолжение
только 30 неотправленных batches само по себе не восполнит labels failed batch.
