# REPORT — статус на 2026-10-08 (вечер, MSK)

> **Обновление 25.10.2026:** gpt-oss доступен до 27 октября; креды вынесены во
> внешний файл `~/.config/wikontic/creds` (`llm_url`, `llm_key`). Актуальные команды,
> ограничения и статус новых проверок: [OCTOBER_FOLLOWUP.md](OCTOBER_FOLLOWUP.md).
> Код подготовлен; новые тесты и прогоны пока не выполнены. Разделы ниже — журнал
> предыдущего этапа, а не актуальное описание окна доступа.

## 0. Окно доступа

В постановке сказано: вечером 8 октября первый день двухдневного окна gpt-oss
заканчивается; Qwen3-235B-A22B-Instruct-2507 ожидается примерно через две недели.
Сейчас **23:5x MSK 2026-10-08** — по календарю остаётся ~1 день окна, но реальный
остаток зависит от инфраструктуры; его нужно проверить при появлении доступа
(`wikontic-exp probe`). Совместного доступа к обеим моделям не предполагается.

## 1. Недостающие настройки (блокер реальных прогонов)

Конфигурация доступа в проекте отсутствует. Нужно положить в
`configs/access.local.yaml` (не в git) или в переменные окружения:

| Что | Где взять | Переменная |
|---|---|---|
| base URL OpenAI-совместимого API университетской инфраструктуры | у выдавшего доступ | `WIKONTIC_API_BASE` |
| API-ключ | там же | `WIKONTIC_API_KEY` |
| точный model id варианта gpt-oss | из конфигурации доступа (не угадывать по названию) | `WIKONTIC_MODEL` |
| (позже) model id Qwen3-235B-A22B-Instruct-2507 | при открытии второго окна | тот же механизм |
| (опционально) model id answerer для QA-абляции | там же | `WIKONTIC_ANSWERER_MODEL` |

После заполнения — первая команда: `.venv/bin/python -m wikontic_exp.cli probe`
(проверка `/models`, наличия `usage`, формата ответа; ничего не предполагается).

**Ни одного запроса к API не отправлялось. Никакие «экспериментальные» числа
имитацией не подменялись.**

## 2. Что реализовано и проверено (mock/dry-run + тесты)

| Компонент | Статус | Проверка |
|---|---|---|
| Скелет проекта, CLI, зависимости | готово | `pip install -e ".[dev]"`, все команды CLI |
| Скачивание источников (Wikontic-промпты, MuSiQue v1.0) | готово | `external/source_manifest.json`: URL, commit, git blob sha, sha256 |
| Baseline verbatim + `baseline_fixed` (только формат) + diff | готово | `prompts/baseline_fixed/baseline_to_fixed.diff` |
| Схема `wikontic_facts_v1` + JSON Schema | готово | `schemas/facts_v1.json`, валидация при парсинге |
| Промпты-кандидаты (coverage / fewshot / claimwise / ensemble3) | готово | `prompts/candidates/`, конфиги в `configs/methods/` |
| Выборка tuning-100 / collection-1000 (seed 42) | готово | `data/*/​*_manifest.json`; QA-id и абзацы между выборками не пересекаются |
| Дедупликация абзацев + refs | готово | tuning: 2000 слотов → 1863 input; collection: 19993 слотов → 14281 input |
| Anti-leak (без question/answer в extraction-входах) | готово | тест `test_extraction_inputs_have_no_question_or_answer` |
| Gold-срез (30 входов, стратифицирован) + протокол разметки | готово | `data/gold/gold_selection.json`, `data/gold/gold_template.json` |
| Async API-клиент (httpx + tenacity) | готово | тест retry transient / fail-fast permanent |
| Runner: кэш, resume, raw-before-parse, статусы ошибок | готово | mock-прогон + тесты resume/invalid_json/usage-отсутствия |
| Различение независимых генераций (gen_index в ключе) | готово | `test_independent_generations_have_distinct_cache_keys` |
| Aggregation: union_dedup, vote, provenance, конфликты | готово | `test_merge_paraphrases_but_keep_distinct_episodes`, `test_vote_threshold` |
| Оценка: proxy + gold P/R (one-to-one matching), компактность, стоимость | готово | `test_one_to_one_matching_prevents_recall_inflation` |
| Qualifiers Q0/Q1/Q2 + аудит сохранности условий | готово | `wikontic-exp qual-ablation` |
| Экспорт корпуса дистилляции (quality_status=unreviewed) | готово | `wikontic-exp export-corpus` |
| Обзор исследований | готово | `reports/literature_review.md` (15 работ, верификация по аннотациям) |
| Тесты обязательных проверок (15) | проходят | `pytest tests/ -q` → 15 passed |

Сквозной dry-run на пилоте из 12 входов (mock, не эксперимент):
`runs/mock_pilot_coverage/`, `runs/mock_pilot_ens3/` (3 генерации → union/vote →
eval → qual-ablation → corpus_mock.jsonl).

## 3. Что НЕ сделано (честно) и почему

1. **Реальные прогоны gpt-oss по tuning-100** — блокер: нет credentials/URL/model id.
2. **Выбор лучшего метода** — не выбирался: нет реальных ответов. Сравнение
   baseline / baseline_fixed / coverage / fewshot / claimwise + ансамблей
   запускается по одной команде на метод (см. §4).
3. **Сбор collection-1000 корпуса** — не начат; конфигурация не заморожена, т.к.
   метод не выбран. Манифест collection уже построен (1000 QA id, input-файлы).
4. **Независимая gold-разметка** — протокол и срез готовы, разметка не выполнена;
   до неё все метрики — proxy, выбор метода — предварительный.
5. **QA-абляция qualifiers (ответы на вопросы MuSiQue из Q0/Q1/Q2)** —
   инфраструктура готова, нужны доступные answerer и замороженный набор фактов.
6. **Student-дистилляция** — вне текущей постановки (требуется только корпус).
7. **Межмодельное объединение gpt-oss + Qwen** — до появления Qwen не проводится;
   сохранённые ответы gpt-oss переиспользуются без повторного доступа.

## 4. Команды продолжения (при появлении доступа)

```bash
# 1) заполнить configs/access.local.yaml (см. §1), затем:
.venv/bin/python -m wikontic_exp.cli probe

# 2) короткий пилот 12 входов на всех кандидатах
for m in baseline_single baseline_fixed_single coverage_single fewshot_single claimwise_single; do
  .venv/bin/python -m wikontic_exp.cli run --method configs/methods/$m.json \
    --inputs data/tuning/pilot12_inputs.jsonl --run-dir runs/gptoss_pilot_$m
done

# 3) сравнение proxy-метрик по пилоту
for m in baseline_single baseline_fixed_single coverage_single fewshot_single claimwise_single; do
  .venv/bin/python -m wikontic_exp.cli evaluate --run-dirs runs/gptoss_pilot_$m \
    --inputs data/tuning/pilot12_inputs.jsonl --out runs/gptoss_pilot_$m/eval.json
done

# 4) расширение перспективных вариантов на tuning-100 (1863 input)
for m in coverage_single fewshot_single coverage_ensemble3; do
  .venv/bin/python -m wikontic_exp.cli run --method configs/methods/$m.json \
    --inputs data/tuning/inputs.jsonl --run-dir runs/gptoss_tuning_$m
done
.venv/bin/python -m wikontic_exp.cli aggregate --run-dirs runs/gptoss_tuning_coverage_ensemble3 \
  --algorithm union_dedup --out runs/gptoss_tuning_coverage_ensemble3/merged_union.json
.venv/bin/python -m wikontic_exp.cli aggregate --run-dirs runs/gptoss_tuning_coverage_ensemble3 \
  --algorithm vote --vote-threshold 2 --out runs/gptoss_tuning_coverage_ensemble3/merged_vote2.json

# 5) gold-разметка 30 входов (data/gold/gold_template.json) → data/gold/<id>.gold.json
#    затем evaluate --gold ...; фиксация precision-порога ДО финального сравнения

# 6) заморозка конфигурации и сбор 1000 (до отключения модели)
.venv/bin/python -m wikontic_exp.cli run --method configs/methods/<WINNER>.json \
  --inputs data/collection/inputs.jsonl --run-dir runs/gptoss_collection_<WINNER>
```

## 5. Зафиксированные решения (осознанные, с оговорками)

- Вариант датасета: **MuSiQue-Ans v1.0, split train** (19938 записей; test без
  публичных ответов не используется). Выборка детерминирована seed=42.
- Пересечения tuning/collection по QA id и абзацам — **нулевые**; внутренние
  повторы абзацев внутри collection учтены дедупликацией с полным списком refs
  (19993 слотов → 14281 уникальных input).
- Форматные правки baseline отделены от смысловых (см. diff); улучшение покрытия
  живёт только в candidates.
- Majority vote может терять редкие верные факты, union — накапливать ошибки:
  оба правила сравниваются на tuning, решение по данным, не по вкусу.
- Precision-порог 0.95 зафиксирован как *целевая настройка* до финального
  сравнения, не как обещанный результат.

## 6. Ограничения текущего состояния

- Метрики в mock-прогонах не имеют отношения к качеству методов (фикстуры).
- Обзор литературы: числовые результаты многих статей на лендингах не выставлены —
  помечены «not verified»; прямых публикаций про union-слияние выходов LLM-IE не
  найдено (зафиксировано как пробел, наши правила объединения — собственные гипотезы).
- Студенческий target не очищен ручной проверкой (quality_status=unreviewed).

## 7. Результаты пилота (pilot-12, gpt-oss-120b, 2026-10-09)

| метод | ok | truncated | фактов | факт/абзац | lex. support | compl. токены | время |
|---|---|---|---|---|---|---|---|
| baseline | 11/12 | 1 | 110 | 10.0 | 0.936 | 19.0k | 204s |
| baseline_fixed | 12/12 | 0 | 122 | 10.2 | 0.918 | 18.0k | 196s |
| coverage | 11/12 | 1 | 113 | 10.3 | 0.894 | 23.5k | 258s |
| fewshot | 11/12 | 1 | 113 | 10.3 | 0.885 | 26.7k | 290s |
| claimwise | 9/12 | 3 | 77 | 8.6 | 0.909 | 31.7k | 343s |

Выводы пилота:
- Truncation = `finish_reason: length` (упёрлись в max_tokens). `max_tokens` поднят до 6000
  во всех активных конфигах (gen_params входят в ключ кэша — изменение до tuning-прогонов).
- Квалификатор `polarity` (отрицание) встречается ТОЛЬКО у coverage/fewshot/claimwise
  (9 вхождений), у baseline и baseline_fixed — 0: смысловые правила работают, эффект
  нужно подтверждать gold-разметкой.
- Число фактов методы не различает (~10/абзац); различия ожидаемы в покрытии/точности.
- claimwise: 25% truncated, стоимость/факт ~2× baseline → отброшен по критерию остановки
  (проверен на пилоте, в tuning не идёт).
- Бюджет: ~17s/вызов при concurrency 8 (НЕ повышаем) → tuning-1863 ≈ 1.3ч/метод,
  collection-14281 ≈ 10ч/метод.

## 8. Runbook: этапы A–C (запускает человек, shell у агента недоступен)

```bash
cd /home/alexey/PycharmProjects/Wikontic_experiments

# --- ЭТАП A: перегон pilot-12 с max_tokens=6000 (цель: ноль truncated) ---
for m in baseline_single baseline_fixed_single coverage_single fewshot_single; do
  .venv/bin/python -m wikontic_exp.cli run --method configs/methods/$m.json \
    --inputs data/tuning/pilot12_inputs.jsonl --run-dir runs/gptoss_pilot2_$m
  .venv/bin/python -m wikontic_exp.cli evaluate --run-dirs runs/gptoss_pilot2_$m \
    --inputs data/tuning/pilot12_inputs.jsonl --out runs/gptoss_pilot2_$m/eval.json
done

# --- ЭТАП B: gold-файлы (per-input для ручной разметки; pilot8 уже готов) ---
.venv/bin/python scripts/make_gold_files.py --inputs data/tuning/inputs.jsonl \
  --selection data/gold/gold_selection.json --out-dir data/gold/pending \
  --pilot-inputs data/tuning/pilot12_inputs.jsonl --pilot-n 8
# ручная разметка: data/gold/pending/pilot8_annotation.json (8 входов, ранний precision)
# и data/gold/pending/*.gold.json (30 входов из gold_selection)

# --- ЭТАП C: tuning-100 (1863 input) на 3 методах ---
for m in coverage_single baseline_fixed_single fewshot_single; do
  .venv/bin/python -m wikontic_exp.cli run --method configs/methods/$m.json \
    --inputs data/tuning/inputs.jsonl --run-dir runs/gptoss_tuning_$m
done
# оценка каждого:
for m in coverage_single baseline_fixed_single fewshot_single; do
  .venv/bin/python -m wikontic_exp.cli evaluate --run-dirs runs/gptoss_tuning_$m \
    --inputs data/tuning/inputs.jsonl --out runs/gptoss_tuning_$m/eval.json
done
# пересечения фактов между методами:
.venv/bin/python scripts/analyze_overlap.py \
  --run-dirs runs/gptoss_tuning_coverage_single runs/gptoss_tuning_baseline_fixed_single \
            runs/gptoss_tuning_fewshot_single \
  --out reports/overlap_tuning.json

# --- ЭТАП C2: ансамбль 3 генерации на подмножестве 300 входов ---
head -300 data/tuning/inputs.jsonl > data/tuning/subset300_inputs.jsonl
.venv/bin/python -m wikontic_exp.cli run --method configs/methods/coverage_ensemble3.json \
  --inputs data/tuning/subset300_inputs.jsonl --run-dir runs/gptoss_ens3_300
.venv/bin/python scripts/analyze_overlap.py --run-dirs runs/gptoss_ens3_300 --ensemble \
  --out reports/ensemble_gain_300.json
# при значимом приросте от 2-й/3-й генерации — расширить до полного tuning:
#   --inputs data/tuning/inputs.jsonl --run-dir runs/gptoss_ens3_full

# --- Затем (по данным + gold): выбор метода, заморозка; сбор collection НЕ запускать ---
```

Ожидаемое время: этап A ~20 мин; этап C ~4ч (3 метода × 1.3ч); этап C2 ~2ч (900 вызовов).
Resume: повторный запуск той же команды пропускает готовые ответы (ключ кэша включает
gen_params — после правки max_tokens пилот пересчитается, это ожидаемо).

## 9. Статус после A+B+C (2026-10-09)

- Этап A: pilot-12 при max_tokens=6000 — **12/12 ok у всех 4 методов**, truncated нет.
- Этап B: gold-файлы сгенерированы (`data/gold/pending/`: 30 per-input + `pilot8_annotation.json`).
  Разметка не выполнена — все числа остаются proxy.
- Этап C (tuning-1863): coverage 1859 ok / 4 truncated; baseline_fixed 1861 / 2;
  fewshot 1857 / 6. Потери 0.2% — НЕ перезапускаем; 12 входов с truncated просто
  отсутствуют в выдаче соответствующего метода (учтено в сравнении: per-input matching).
- Качественный сигнал на полном tuning: квалификатор `polarity` — у coverage и fewshot
  (≥100 вхождений каждый), у baseline_fixed — **0** (правила сохранения отрицания работают
  только в candidates).

## 10. Следующий этап (запускает человек)

```bash
cd /home/alexey/PycharmProjects/Wikontic_experiments

# 1) Сравнение методов (D-7, proxy + gold по мере разметки) — один отчёт
.venv/bin/python scripts/compare_methods.py \
  --methods coverage=runs/gptoss_tuning_coverage_single \
            baseline_fixed=runs/gptoss_tuning_baseline_fixed_single \
            fewshot=runs/gptoss_tuning_fewshot_single \
  --inputs data/tuning/inputs.jsonl --gold-dir data/gold/pending \
  --out reports/tuning_comparison.json

# 2) Этап C2: ансамбль 3 генерации на 300 входах (~2ч)
head -300 data/tuning/inputs.jsonl > data/tuning/subset300_inputs.jsonl
.venv/bin/python -m wikontic_exp.cli run --method configs/methods/coverage_ensemble3.json \
  --inputs data/tuning/subset300_inputs.jsonl --run-dir runs/gptoss_ens3_300
.venv/bin/python scripts/analyze_overlap.py --run-dirs runs/gptoss_ens3_300 --ensemble \
  --out reports/ensemble_gain_300.json

# 3) Ручная gold-разметка (можно параллельно с 1-2):
#    сначала data/gold/pending/pilot8_annotation.json (8 входов → ранний precision),
#    затем data/gold/pending/*.gold.json — заполнить "triplets", поставить status="annotated".
#    После разметки перезапустить команду (1) — gold-метрики подтянутся автоматически.
```

После (1)+(2)+разметки — выбор метода и заморозка конфигурации (D-8, сбор collection
НЕ запускать без явного указания).

## 11. Кросс-промпт ансамбль из готовых ответов (после сравнения single-методов)

Сравнение (2026-10-09): coverage 19384 / baseline_fixed 18882 / fewshot 18966 сем. фактов;
union кластеров трёх методов 39995, «только у одного метода» ~9000–9500 у каждого,
совпадение всех трёх — 4683 (11.7%). Методы слабо согласны; верность уникальных
кластеров требует gold.

```bash
# ансамбль 3 промптов (3 вызова на вход, НЕ single-call): union и vote>=2
.venv/bin/python -m wikontic_exp.cli aggregate \
  --run-dirs runs/gptoss_tuning_coverage_single runs/gptoss_tuning_baseline_fixed_single runs/gptoss_tuning_fewshot_single \
  --algorithm union_dedup --out runs/ensemble_xprompt/merged_union.json
.venv/bin/python -m wikontic_exp.cli aggregate \
  --run-dirs runs/gptoss_tuning_coverage_single runs/gptoss_tuning_baseline_fixed_single runs/gptoss_tuning_fewshot_single \
  --algorithm vote --vote-threshold 2 --out runs/ensemble_xprompt/merged_vote2.json

# сравнение 5 вариантов (3 single + 2 ансамбля)
.venv/bin/python scripts/compare_methods.py \
  --methods coverage=runs/gptoss_tuning_coverage_single \
            baseline_fixed=runs/gptoss_tuning_baseline_fixed_single \
            fewshot=runs/gptoss_tuning_fewshot_single \
            xprompt_union=runs/ensemble_xprompt/merged_union.json \
            xprompt_vote2=runs/ensemble_xprompt/merged_vote2.json \
  --inputs data/tuning/inputs.jsonl --gold-dir data/gold/pending \
  --out reports/tuning_comparison.json
```

Разметка: `reports/ANNOTATION_GUIDE.md` (порядок, правила, пример-образец).
