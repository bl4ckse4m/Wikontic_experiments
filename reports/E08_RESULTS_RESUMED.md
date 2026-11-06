# E08 — результаты после разрешённого продолжения

Статус: **complete silver evaluation; no superiority inferred**. Run: `e08_resume_20261101T214802_996074Z`.
UTC сборки отчёта: 2026-11-01T22:06:42.066626+00:00.

## Условия

50 фиксированных tuning-абзацев MuSiQue-Ans v1.0 train. Четыре single-pass метода и четыре варианта кросс-промпт ансамбля; существующие predictions не перегенерировались.
Stateless OpenRouter API, `anthropic/claude-sonnet-5.5`, reasoning=low, только Anthropic без fallback; system/user без tools и истории. Те же 886 уникальных фактов / 200 пар, rubric, batches и порядок.
Для каждого ID сохраняется одна содержательная оценка. Разрешён один повтор исходного HTTP429 без labels и 30 ранее неотправленных jobs: максимум 101 HTTP-попытка на 100 научных batches, не повторная оценка успешных ответов.
Source-first gold и человеческой проверки нет. Это **judged precision**, не независимая точность и не recall. Calibration в агентном интерфейсе не валидирует API-harness.

## Общая таблица

Интервалы — исходные approximate strata-Wilson/Bonferroni envelopes, не строгие тесты превосходства. В availability остаются все 50 входов; ошибки/пустые выдачи не исключаются.

| Метод | ok/failed/pending | Empty ok | Inputs с facts / 50 | Occurrences N | Размечено / sampled | Full P [interval] | Core P [interval] | N / input | Оценка supported occurrences / input | Chars / input |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| baseline_fixed | 50/0/0 | 0 | 50 | 543 | 150/150 | 0.940 [0.848; 0.977] | 0.947 [0.857; 0.980] | 10.86 | 10.21 | 2048 |
| coverage | 50/0/0 | 0 | 50 | 580 | 150/150 | 0.947 [0.857; 0.981] | 0.960 [0.875; 0.988] | 11.60 | 10.98 | 2165 |
| fewshot | 50/0/0 | 0 | 50 | 553 | 150/150 | 0.933 [0.840; 0.973] | 0.953 [0.865; 0.985] | 11.06 | 10.32 | 2124 |
| autoontik | 50/0/0 | 0 | 50 | 568 | 150/150 | 0.933 [0.839; 0.974] | 0.973 [0.896; 0.991] | 11.36 | 10.60 | 2104 |
| heuristic_union | 50/0/0 | 0 | 50 | 1227 | 150/150 | 0.934 [0.717; 0.977] | 0.954 [0.746; 0.982] | 24.54 | 22.92 | 4751 |
| heuristic_vote2 | 50/0/0 | 1 | 49 | 326 | 150/150 | 0.967 [0.811; 0.995] | 0.980 [0.828; 0.998] | 6.52 | 6.30 | 1291 |
| llm_union | 47/3/0 | 0 | 47 | 815 | 150/150 | 0.927 [0.698; 0.976] | 0.961 [0.742; 0.986] | 16.30 | 15.11 | 3114 |
| llm_vote2 | 47/3/0 | 0 | 47 | 363 | 150/150 | 0.961 [0.807; 0.989] | 0.993 [0.851; 0.999] | 7.26 | 6.97 | 1358 |

Supported occurrences — оценка числа правильных **выходных вхождений**, включая смысловые дубли; не уникальные факты и не покрытие. Chars — одинаковая JSON-сериализация fact fields по всем входам.

## Основные contrasts

Разность — первый метод минус второй; только описательные точечные оценки, без вывода о превосходстве.
- autoontik vs baseline_fixed: full_label: -0.0067; core_label: +0.0268.
- llm_union vs heuristic_union: full_label: -0.0067; core_label: +0.0067.

## Отдельный аудит LLM-merger v2

- proposed_merge: ошибок 7/100, rate=0.070, Wilson=[0.034; 0.137], population pairs=803.
- proposed_separate: ошибок 0/100, rate=0.000, Wilson=[0.000; 0.037], population pairs=22465.

Пары отбирались pair-uniform внутри proposed-merge/separate на успешных merger partitions. Это условные ошибки эквивалентности, не extraction recall. Эвристический partition отдельно по этим парам не оценивался; аудит не доказывает превосходства v2.

## Исполнение и ресурсы

- Последние состояния 100 API jobs: `{"ok": 100}`; labels: `{"binary": 886, "merger_audit": 200}`. Это не счётчик исторических HTTP-попыток.
- История HTTP: 70 исходных + 31 новых = 101 (максимум 101). Исходный failed HTTP429 сохранён даже при latest ok после разрешённого повтора.
- Reported judge cost: $1.855470; неизвестная стоимость: 1 записей. Conservative accounting с резервами: $1.999362 — не фактический счёт.
- Исходный admission-control HTTP429 не дал labels; provider null/unreported и ошибка не доказывают отсутствие billing. Его неизвестная стоимость остаётся в `missing_cost_jobs` с префиксом `prior_attempt/` и в accounting, отдельно от результата повтора.
- Reported judge usage: `{"prompt_tokens": 324520, "completion_tokens": 120643, "total_tokens": 445163}`; provider metadata: `{"Anthropic": 100, "unreported_historical_admission_error": 1}`.
- Key checks: 70 исходных / 31 новых; исходный guard margin=$0.50. Стоимость — учёт, не критерий выбора.
- После завершения read-only `/key`: remaining=$3.14453, daily usage=$1.85547; это snapshot учёта, не доказательство окончательного billing failed попытки.
- Сумма latency всех 101 HTTP receipts: 893.24 s, включая HTTP429; пауза между сегментами в эту сумму не входит.
- Smoke evidence скопировано из исходного run: это initial transport/schema/identity smoke, не новый вызов и не проверка качества.
- Отдельный continuation smoke — первый разрешённый retry batch, все 32 labels приняты после проверки HTTP/schema/IDs/model/provider; это не дополнительная калибровка и не аудит качества.
- Single methods — 1 extraction/вход; heuristic ensemble — 3; LLM ensemble — 3 extraction + 1 merger/вход. Source usage, attempts и latency retained records — в metrics.json. Старые retries не полностью реконструированы; отсутствующая цена не равна нулю.
- Сумма latency не равна wall-clock при параллельной обработке. Union/vote2 используют общие ресурсы: строки методов нельзя суммировать как независимые расходы.

## История исполнения и ограничения

Исходные delivery plan, requests, halt и receipts не изменялись. Новый continuation plan и ledger фиксируют разрешённое пользователем продолжение после HTTP429: только один точный повтор label-less failed job и 30 never-attempted jobs, без quality-conditioned retries, relabeling и научных изменений.
API delivery заменил файловые агентные сессии до main labels; научные денежные caps и provider price filter были удалены, техническая защита ключа сохранена. Содержание rubric, samples, модель и output limits не менялись.
Результаты ограничены tuning subset, historical route/retry differences (включая AutoOntik), непроверенным silver judge, общими labels и зависимостью фактов внутри абзаца. Target precision≥0.95 не сертифицирована даже при высокой точечной оценке. Старые matcher-зависимые P/R/F1 не используются как доказательство улучшения.

## Артефакты

- `runs/e08_resume_20261101T214802_996074Z/metrics.json`, `binary_precision.json`, `merger_audit.json`, `api_final_status.json`, `score_provenance.json`.
- `runs/e08_resume_20261101T214802_996074Z/api_judging/`: continuation plan, история исходного состояния и новый append-only ledger; исходный `data/experiments/e08/api_judging/` сохранён.
- `data/judging/e08/{binary,merger_audit}/judgments.jsonl`: объединённые labels, исходные строки сохранены. Selection/rubric/prediction locks и оба scorer не менялись.
- `reports/E08_RESULTS.md` — неизменённый исторический неполный отчёт; данный отчёт имеет отдельный путь. Provenance связывает continuation plan/history, новый ledger, исходные receipts, labels и scores по SHA-256.
- `runs/e08_resume_20261101T214802_996074Z/original_label_snapshots/`: исходные 579/134 labels восстановлены только из прежних receipts; их hashes совпали с provenance частичного scoring. Main judgments после continuation — накопительный полный файл, не снимок старой остановки.

## Граница решения

При неполных labels E08 не завершён, победитель не выбирается. Полнота silver labels сама по себе не доказывает превосходства, истинный recall или пригодность корпуса. Близкие точечные оценки не образуют надёжный рейтинг. Ограниченное проектное решение оформляется отдельно после чтения результатов; автоматического запуска collection-1000 нет.

## Решение по завершённому E08

**Наблюдения.** Все 886 binary и 200 pair labels получены, пропусков нет.
Primary full-label contrasts AutoOntik−baseline_fixed и llm_union−heuristic_union
равны примерно −0.67 процентного пункта каждый. На этой silver-оценке ни один
из двух переходов не показал положительного точечного прироста основного показателя.
Это не доказательство равенства методов или статистически установленного ухудшения.

У heuristic_vote2 наибольшая точечная full judged precision (0.967), но это всего
326 выходных вхождений / 6.52 на вход против 1227 / 24.54 у heuristic_union.
Это разные наборы выдачи, а не измерение потери или прироста recall. Широкие
приблизительные интервалы и непроверенный judge не позволяют считать 0.967
сертификацией target precision≥0.95 или надёжным рейтингом.

У LLM-merger обнаружены 7/100 ошибочных proposed merges по silver judge; 0/100
ошибок proposed separate означает только отсутствие обнаруженных ошибок в этой
подвыборке (Wilson upper≈0.037), не отсутствие missed duplicates в population.
Разница размеров union 1227→815 не выделяет чистый эффект дедупликации: кроме
непроверенной эквивалентности, LLM-merger имеет 3 failed inputs против 0 у heuristic.

**Интерпретация.** E08 даёт полностью наблюдаемое, воспроизводимое описательное
сравнение judged precision, выдачи и доступности, но не определяет надёжного
победителя по уникальным подтверждённым фактам в компактной форме. Shared labels,
paragraph dependence, tuning-only sample, исторические route/retry differences и
возможные систематические ошибки judge остаются альтернативными объяснениями
малых различий. Пересечение приведённых intervals само по себе не является тестом
равенства; формальный тест превосходства здесь не выполнен.

**Рабочее решение:** оставить `baseline_fixed` временным single-pass контролем,
а не объявлять его лучшим или готовым источником student-корпуса. Это простой
вариант с 50/50 доступностью; E08 не обосновал замену его на AutoOntik или
принятие LLM-merger как улучшенного этапа. Цена не участвовала в выборе.
Все остальные варианты сохранены; они не отвергаются по непроверенному recall.

**Следующий шаг, не автоматический запуск:** определить отдельную source-first
silver-процедуру оценки уникального подтверждённого покрытия и смысловых потерь
дедупликации при сохранении запрета человеческой разметки. Именно этих величин
не хватает для конечного выбора компактного extractor+merger; увеличение числа
выходных триплетов или повторное ранжирование по точечной precision пробел не закроет.
E08 закрыт как exploratory silver evaluation; collection-1000 и обучение не запущены.
