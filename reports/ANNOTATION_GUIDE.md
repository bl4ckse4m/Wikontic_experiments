# Как делать gold-разметку фактов

Цель: независимый эталон **всех явно сообщённых фактов** абзаца — для precision/recall
методов извлечения. Размечаем не «что извлекли системы», а «что на самом деле сказано
в тексте» (пропущенные системами факты — это recall).

## 1. Что заполнять

Порядок: сначала ранний набор, потом основной.

1. **`data/gold/pending/pilot8_annotation.json`** — 8 абзацев в одном файле (≈1–2 часа).
   У каждого элемента заполни поле `"triplets"`; в конце файла у `"status"` поставь `"annotated"`.
2. **`data/gold/pending/<input_id>.gold.json`** — 30 файлов по одному входу.
   Заполни `"triplets"`, у `"status"` внутри файла поставь `"annotated"`.

Формат факта (схема `wikontic_facts_v1`, `schemas/facts_v1.json`):

```json
{
  "subject": "Ada Lovelace",
  "relation": "creator of work",
  "object": "first published algorithm for the Analytical Engine",
  "qualifiers": [
    {"relation": "point in time", "object": "1843"},
    {"relation": "attribution", "object": "text"}
  ],
  "subject_type": "human",
  "object_type": "work"
}
```

`subject_type`/`object_type` — желательны, но допустимы без них. Никаких других полей
не добавляй (служебные `claim_ids` и `claims` не нужны).

## 2. Правила (краткая версия протокола)

- **Только явно сказанное.** Без знаний из головы, без транзитивных следствий
  («X родился в Y» не даёт «Y — город X»), без искусственных инверсий.
- **Координацию разбивай:** «физик и химик» → два факта `occupation`.
- **Конкретные отношения:** `date of birth`, `occupation`, `award received`, `located in`…
  Не `related_to`. Не кусок предложения в `object` — короткое однозначное имя.
- **Местоимения** разрешай по контексту, если антецедент однозначен; если нет — не угадывай.
- **Qualifiers — только привязанные к данному факту уточнения** (время, место, роль,
  количество, единица, приблизительность, отрицание, модальность, атрибуция, условие).
  Не приписывай всем фактам все даты абзаца.
- **Дата как объект vs как qualifier:**
  - `(Marie Curie, date of birth, 7 November 1867)` — самостоятельное отношение;
  - `(Marie Curie, award received, Nobel Prize in Physics)` + `{"relation": "point in time", "object": "1903"}` — уточнение события.
- **Отрицание / модальность / атрибуция — обязательны к сохранению:**
  - отрицание: `{"relation": "polarity", "object": "negative"}`;
  - возможное/планируемое: `{"relation": "modality", "object": "possible"}` (и т.п.);
  - чужое утверждение: `{"relation": "attribution", "object": "<кто>"}`
  Никогда не превращай возможное или чужое в безусловный факт.
- **«falsely claimed» (ложно утверждал):** сам акт утверждения — факт положительный,
  а содержание НЕ является фактом: либо не выноси содержание, либо вынеси с
  `polarity: negative` + `attribution`. См. пример ниже.
- **Разные эпизоды не сливать** только из-за совпадения s/r/o: две премии в разные годы —
  два факта, у каждого свой `point in time`.
- **Отсутствующие уточнения не выдумывать.** Если чего-то в тексте нет — не ставь.

## 3. Пример-образец

Текст: «Charles-Albert Demoustier (13 March 1760 – 2 March 1801) was a French writer.
He falsely claimed to be a descendant of La Fontaine by his mother and Racine by his father.»

Эталон:

```json
[
  {"subject": "Charles-Albert Demoustier", "relation": "date of birth", "object": "13 March 1760",
   "qualifiers": [], "subject_type": "human", "object_type": "date"},
  {"subject": "Charles-Albert Demoustier", "relation": "date of death", "object": "2 March 1801",
   "qualifiers": [], "subject_type": "human", "object_type": "date"},
  {"subject": "Charles-Albert Demoustier", "relation": "occupation", "object": "writer",
   "qualifiers": [{"relation": "country of citizenship", "object": "France"}],
   "subject_type": "human", "object_type": "profession"},
  {"subject": "Charles-Albert Demoustier", "relation": "claimed to be descendant of", "object": "La Fontaine",
   "qualifiers": [{"relation": "via", "object": "his mother"}, {"relation": "veracity", "object": "false"}],
   "subject_type": "human", "object_type": "human"},
  {"subject": "Charles-Albert Demoustier", "relation": "claimed to be descendant of", "object": "Racine",
   "qualifiers": [{"relation": "via", "object": "his father"}, {"relation": "veracity", "object": "false"}],
   "subject_type": "human", "object_type": "human"},
  {"subject": "Charles-Albert Demoustier", "relation": "is descendant of", "object": "La Fontaine",
   "qualifiers": [{"relation": "polarity", "object": "negative"}, {"relation": "attribution", "object": "Charles-Albert Demoustier"}],
   "subject_type": "human", "object_type": "human"},
  {"subject": "Charles-Albert Demoustier", "relation": "is descendant of", "object": "Racine",
   "qualifiers": [{"relation": "polarity", "object": "negative"}, {"relation": "attribution", "object": "Charles-Albert Demoustier"}],
   "subject_type": "human", "object_type": "human"}
]
```

Разбор:
- даты — объекты самостоятельных отношений (не qualifiers);
- «French writer» → `occupation` + гражданство в qualifiers (не два разных факта);
- «falsely claimed»: акт утверждения зафиксирован, а «is descendant of» помечено
  `polarity: negative` и `attribution` — содержание ложного заявления не стало фактом.

## 4. Типичные ловушки

| Ситуация | Как правильно |
|---|---|
| «about 5.2%» | объект `5.2%` + `{"relation": "approximation", "object": "about"}` |
| «X, the former CEO of Y, …» | `(X, occupation, CEO)` + `{"relation": "end time", "object": "former"}` или отдельный факт роли |
| «A and B signed the treaty» | два факта (A signed…, B signed…) |
| «the ministry said inflation fell» | `(ministry, stated, inflation fell)` + `attribution`; НЕ `(inflation, fell, …)` как безусловный факт |
| «not related to X» | `(S, related to, X)` + `polarity: negative` |
| «serves as … since 2015» | роль-факт + `{"relation": "start time", "object": "2015"}` |
| перечисление в конце абзаца (features like:) | выноси только то, что перечислено, без домысливания |

## 5. Как это используется

- Сопоставление с предсказаниями — по смыслу, один-к-одному: один эталонный факт не
  может засчитать несколько предсказаний; пропущенный эталонный факт — это потеря recall.
- Метрики: precision/recall/F1 базовых фактов (s/r/o) и полных утверждений (с
  qualifiers), отдельно — precision/recall самих qualifiers.
- Порог precision для выбора метода — 0.95 (зафиксирован заранее).
- Файлы со `"status": "unannotated"` в оценке не участвуют; результаты без разметки —
  proxy и помечаются как предварительные.

Вопросы по неоднозначным случаям: фиксируй своё решение комментарием в конце файла
(поле `"annotator_notes"`), спорные случаи можно пометить `"uncertain": true` у факта —
это допустимо, главное — не терять scope молча.

## 6. Если размечает LLM, а не эксперт (silver-статус)

Разметка LLM допустима как **silver**, но никогда не как gold. Правила:

1. LLM (или агент) готовит черновик фактов по протоколу выше.
2. **Ты проверяешь** каждый факт по тексту: сказано ли это действительно? нет ли вымысла?
   не потерян ли отрицание/атрибуция? достаточно ли полный список (пропущенное = recall).
3. Ставишь `"status": "silver"` (не `"annotated"`!) и `"annotator_notes": "LLM-drafted,
   human-verified"` (+ какая модель и дата).
4. В отчётах silver-числа всегда подписываются как LLM-assisted; выбор метода остаётся
   **предварительным** до human-gold. Метрики по silver и по gold не смешиваются.

Почему нельзя просто поверить LLM-аннотатору: (а) его ошибки коррелируют с ошибками
экстракторов (часто та же модель/семейство) — precision будет завышен; (б) если LLM
«не замечает» класс фактов, recall всех методов окажется искусственно высоким;
(в) постановка прямо требует не выдавать автоматическую оценку за эталон.

Практичный компромисс при ограниченном бюджете: silver-разметка на pilot8 (8 входов) +
ручной **precision-аудит** случайной выборки ~50–100 предсказанных фактов (вопрос «сказано
ли это в тексте?» — намного проще, чем составлять полный список). Это даёт оценку
precision пригодную для сравнения методов, а recall — только как silver/proxy.
