# wikontic-exp — поиск метода извлечения гиперреляционных фактов для Wikontic

Самостоятельный исследовательский проект: подбор и обоснование метода промптинга и
объединения независимых ответов моделей для извлечения `(subject, relation, object,
qualifiers)` из абзацев MuSiQue. Корпус «текст → структурированные факты» готовится
для последующей дистилляции (student не обучается в этой версии).

Контракт фактов: `schemas/facts_v1.json` (`wikontic_facts_v1`), формат Wikontic
`{"triplets": [...]}` с `qualifiers`, `subject_type`/`object_type` сохранены для
сопоставимости. Поле `claim_ids` и список `claims` — служебные (claimwise-вариант),
в student-target не попадают.

## Статус

Реальные API-прогоны **не выполнялись**: конфигурация доступа отсутствует (см.
`reports/REPORT.md`, раздел «Недостающие настройки»). Всё ниже проверено на
детерминированном mock/dry-run; mock-результаты не являются экспериментальными.

## Установка

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
# зависимости: httpx, tenacity, PyYAML (pytest — для тестов)
```

## Данные

Источники (URL, commit, sha256) — `external/source_manifest.json`. Официальный
релиз MuSiQue-Ans v1.0 скачан в `musique_raw/` (zip 272MB, в git не включается).

```bash
# выборка tuning-100 / collection-1000 (seed=42), манифесты, дедуп, gold-срез
.venv/bin/python -m wikontic_exp.cli prepare-data --data-dir data \
  --split-file musique_raw/data/musique_ans_v1.0_train.jsonl
```

- Единица извлечения: `title + paragraph_text` (все абзацы, включая distractors).
- tuning-100 QA → 1863 уникальных input; collection-1000 QA → 14281 уникальный input.
- QA ID и абзацы между tuning и collection не пересекаются; внутренние повторы
  абзацев коллекции дедуплицированы, все ссылки `(sample_id, paragraph_idx)` сохранены.
- Question/answer/decomposition в extraction-запросы не передаются (лежат отдельно в
  `qa_sidecar.jsonl`).

## Запуск

```bash
# проверка доступа и возможностей endpoint (usage, формат ответа)
.venv/bin/python -m wikontic_exp.cli probe

# извлечение одним методом (пилот на 12 входах)
.venv/bin/python -m wikontic_exp.cli run \
  --method configs/methods/coverage_single.json \
  --inputs data/tuning/pilot12_inputs.jsonl \
  --run-dir runs/gptoss_pilot_coverage

# ансамбль: 3 независимые генерации (НЕ single-call система) + объединение
.venv/bin/python -m wikontic_exp.cli run --method configs/methods/coverage_ensemble3.json \
  --inputs data/tuning/inputs.jsonl --run-dir runs/gptoss_ens3
.venv/bin/python -m wikontic_exp.cli aggregate --run-dirs runs/gptoss_ens3 \
  --algorithm union_dedup --out runs/gptoss_ens3/merged_union.json
.venv/bin/python -m wikontic_exp.cli aggregate --run-dirs runs/gptoss_ens3 \
  --algorithm vote --vote-threshold 2 --out runs/gptoss_ens3/merged_vote2.json

# оценка (proxy + gold, если есть разметка)
.venv/bin/python -m wikontic_exp.cli evaluate --run-dirs runs/gptoss_ens3 \
  --inputs data/tuning/inputs.jsonl --gold data/gold/<input_id>.gold.json \
  --out runs/gptoss_ens3/eval.json

# абляция qualifiers: Q0/Q1/Q2 из одного замороженного набора фактов
.venv/bin/python -m wikontic_exp.cli qual-ablation \
  --frozen runs/gptoss_ens3/merged_union.json --out-dir runs/gptoss_ens3/qual_ablation

# корпус для дистилляции
.venv/bin/python -m wikontic_exp.cli export-corpus --inputs data/tuning/inputs.jsonl \
  --merged runs/gptoss_ens3/merged_union.json --teacher-meta runs/gptoss_ens3/teacher_meta.json \
  --out runs/gptoss_ens3/corpus.jsonl

# mock/dry-run всего пайплайна без доступа
.venv/bin/python -m wikontic_exp.cli run --method configs/methods/coverage_single.json \
  --inputs data/tuning/pilot12_inputs.jsonl --run-dir runs/mock_pilot_coverage --mock

# тесты обязательных проверок
.venv/bin/python -m pytest tests/ -q
```

## Доступ к API

Настройки — только через переменные окружения или `configs/access.local.yaml`
(в git не попадает, см. `configs/access.example.yaml`):

| Переменная | Назначение |
|---|---|
| `WIKONTIC_API_BASE` | base URL OpenAI-совместимого endpoint |
| `WIKONTIC_API_KEY` | ключ |
| `WIKONTIC_MODEL` | точный model id (из конфигурации доступа, не по названию) |
| `WIKONTIC_ANSWERER_MODEL` | (опционально) answerer для QA-абляции |
| `WIKONTIC_MAX_CONCURRENCY`, `WIKONTIC_TIMEOUT_S`, `WIKONTIC_MAX_RETRIES` | лимиты |

Сначала выполняется `probe`: поддержка `/models`, наличие `usage`, формат ответа —
ничего не предполагается заранее.

## Промпты-кандидаты

| Конфиг | Промпт | Что проверяется |
|---|---|---|
| `baseline_single` | `prompts/baseline/propmt_1_types_qualifiers.txt` | Wikontic baseline verbatim (с его JSON-дефектами) |
| `baseline_fixed_single` | `prompts/baseline_fixed/...` | только форматные правки (diff сохранён) |
| `coverage_single` | `prompts/candidates/coverage_v1.txt` | явные правила покрытия конструкций и семантики |
| `fewshot_single` | `prompts/candidates/fewshot_v1.txt` | coverage + разнообразные демонстрации |
| `claimwise_single` | `prompts/candidates/claimwise_v1.txt` | организация ответа по атомарным claims (1 вызов) |
| `coverage_ensemble3` | `coverage_v1` ×3 gen | ансамбль + union_dedup / vote |

Все extractor'ы — один вызов на вход (single-pass); ансамбль явно помечен как
несколько вызовов. Обоснование — `reports/literature_review.md`.

## Надёжность

- Retries (tenacity): сетевые/timeout/429/5xx, exponential backoff + jitter,
  `Retry-After`; постоянные ошибки (400/401/403) без повторов; бюджет повторов один.
- Raw response сохраняется до парсинга; `invalid_json` / `schema_error` / `truncated` /
  `api_error` — отдельные статусы, не «пустые успешные» извлечения.
- Кэш-ключ: `hash(input, model, prompt, schema, gen_params, gen_index)`; `gen_index`
  различает намеренные независимые генерации ансамбля; retry не создаёт новую генерацию.
- Ключ merger: хэши всех входных ответов + версия алгоритма. Каждый факт в merged
  имеет происхождение (model/prompt/gen_index/cache_key) и число поддержавших генераций.
- Запись ответов атомарная (tmp+rename), resume по наличию записи.

## Структура

```
configs/methods/     конфиги методов (промпт + gen_params + n_generations)
prompts/             baseline (неизменённый), baseline_fixed (+diff), candidates
schemas/facts_v1.json  контракт фактов
src/wikontic_exp/    data_prep, api_client, runner, aggregation, evaluation,
                     qualifiers (Q0/Q1/Q2), distill_export, cli
data/tuning|collection|gold/  inputs, qa_sidecar, манифесты, gold-протокол
runs/                raw responses, parsed, usage, merged, eval, corpus
reports/             literature_review.md, REPORT.md
tests/               обязательные проверки (15 тестов)
external/            скачанные публичные файлы + source_manifest.json
```
