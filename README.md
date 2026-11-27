# Wikontic

Эксперименты по извлечению триплетов из текста: сравниваем промпты внутри каждой модели, затем сравниваем одиночные и объединённые выдачи. Для чтения сохранённых данных, предпросмотра запусков и подсчёта метрик сеть и API-ключи не нужны.

## Текущее сравнение

1. Сравниваем четыре промпта — [`baseline_fixed`](prompts/baseline_fixed/propmt_1_types_qualifiers_fixed.txt), [`coverage`](prompts/candidates/coverage_v1.txt), [`fewshot`](prompts/candidates/fewshot_v1.txt) и [`autoontik`](prompts/autoontik/prompt_1_with_types_and_qualifiers.txt) — на [одних и тех же 50 абзацах](runs/prompt_comparison/sample50.json). Промпт выбираем **отдельно для gpt-oss и второй модели института (модель B)**. Для gpt-oss готовы [все 200 списков](runs/prompt_comparison/answers_complete.jsonl) на этой выборке (2 266 триплетов). Модель B пока не выбрана: прежний Qwen3-235B вернул HTTP 404, а его сохранённый пилот не пересекается с выборкой. Разметка текущей выборки ещё не завершена.
2. Claude получает заголовок, абзац и **один полный список** за запрос. Названия промпта и модели извлечения не передаются. Сравниваем размер выдачи, число подтверждённых триплетов и **Precision** — долю подтверждённых — на одних и тех же абзацах. Для исторического `autoontik` использован другой маршрут API (OpenRouter); различия с другими историческими выдачами нельзя приписать только промпту.
3. После выбора промптов складываем одиночные списки обеих моделей. Каждая модель института по отдельности группирует **один и тот же пул в том же порядке**, а итоговый список состоит из исходных представителей групп. Claude проверяет только подтверждённость, **не дубли**. Смысловую уникальность и полноту не измеряем.

[Схема эксперимента](research/EXPERIMENT_FLOW.md) · [Настройки и команды по этапам](research/PROMPTS_TESTING.md)

## Быстрый старт

Нужен Python 3.11+. Из корня репозитория установите пакет:

```bash
python -m pip install -e .
```

Проверить план оценки gpt-oss **без вызовов API**:

```bash
python scripts/run_judging.py \
  --sample runs/prompt_comparison/sample50.json \
  --answers runs/prompt_comparison/answers_complete.jsonl \
  --out runs/prompt_comparison/gptoss_sample50_grounding.jsonl
```

Ожидаемый вывод: `50 абзацев, 200 выдач; 200 запросов до учёта сохранённых результатов`. Только `--execute` отправляет запросы; `--limit 4` ограничивает первый запуск четырьмя заданиями. Для реального запуска укажите в `~/.config/wikontic/creds` `LLM_URL` и `LLM_KEY` для института, `openrouter_key` — для Claude. Можно передать другой путь через `--creds-file`. Ключи не добавляйте в Git.

| Скрипт | Назначение |
|---|---|
| [`run_extraction.py`](scripts/run_extraction.py) | Извлекает триплеты на сервере института. Принимает точный `--model-id`; при обрезании ответа допускает один повтор с повышенным лимитом. |
| [`run_judging.py`](scripts/run_judging.py) | Записывает оценки Claude `0/1`, сырые ответы и данные об использовании API, если они доступны. |
| [`score_judgments.py`](scripts/score_judgments.py) | Без API считает Precision и размер выдачи на общих оценённых абзацах. Помечает неполные сравнения. |
| [`run_dedup.py`](scripts/run_dedup.py) | Передаёт объединённый пул одной из двух моделей института. Проверяет, что каждый исходный ID вошёл ровно в одну группу. |

Тесты без сети: `python -m pip install -e '.[dev]'`, затем `python -m pytest -q tests`.

## Где что лежит

- [`data/tuning/`](data/tuning/) — 100 вопросов MuSiQue-Ans v1.0 и 1 863 уникальных абзаца. [`data/collection/`](data/collection/) — отдельная выборка из 1 000 вопросов, не используется для настройки. Модели получают только `title` и `paragraph_text` из `inputs.jsonl`; вопросы и ответы хранятся отдельно в `qa_sidecar.jsonl`.
- [`data/answers/`](data/answers/) — сохранённые ответы моделей. `input_id` связывает их с абзацем, `raw_text` хранит текст ответа, `status` отмечает в том числе ошибки. `source_file`, если есть, указывает на оригинал в архиве до очистки. Для офлайн-примера с двумя моделями найдите `1521738704ca6c09` в `data/tuning/inputs.jsonl`, `data/answers/gptoss_paragraph.jsonl` и `data/answers/qwen_paragraph.jsonl`. Это исторический пилот, не часть текущих 50 абзацев.
- [`runs/prompt_comparison/`](runs/prompt_comparison/) — `sample50.json` с ID выборки и `answers_complete.jsonl` с 7 452 подготовленными выдачами gpt-oss на всём `tuning`. `judgments_full.jsonl` — прежняя частичная разметка, **не результат текущего сравнения**. История дозапусков обрезанных ответов — в [`runs/prompt_recovery/`](runs/prompt_recovery/).

## Исторические результаты

- **Сравнение со списками Claude** (`93141b2`): [отчёт](reports/tuning_comparison.md), [числа и ID](reports/tuning_comparison.json), [разметка 37 абзацев](data/gold/pending/). P/R/F1 в отчёте отражают совпадение с этими списками, а не прямую проверку выданного триплета по тексту.
- **Прямая проверка по тексту, E08** (`0974f43`): [итоговый отчёт](reports/E08_RESULTS_RESUMED.md), [выборка](data/experiments/e08/selection.json), [кандидаты](data/judging/e08/binary/candidates.jsonl) и [886 оценок](data/judging/e08/binary/judgments.jsonl), связанных по `annotation_id`. [200 оценок пар](data/judging/e08/merger_audit/) относятся к историческому аудиту объединения и не входят в знаменатель Precision.
