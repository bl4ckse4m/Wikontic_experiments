# Annotation task for the annotator agent (Claude in opencode, separate session)

Copy everything below the line into a new Claude session in this same opencode
project. The agent has access to the project files — inputs do not need to be pasted.

---

TASK:

You are an annotator of hyper-relational facts for the Wikontic research project
(extraction of `(subject, relation, object, qualifiers)` from MuSiQue paragraphs).
Work only inside the project directory `/home/alexey/PycharmProjects/Wikontic_experiments`.

## Independence constraints (strict)

- Do NOT open or read anything under `runs/` — that directory contains predictions of
  the systems being evaluated. Annotation must stay independent; peeking at system
  outputs would corrupt the evaluation.
- Work ONLY from the paragraph texts in the input files. No world knowledge.

## What to do

1. Read the annotation protocol: `reports/ANNOTATION_GUIDE.md` (rules + worked example).
2. Annotate the inputs in this order:
   - **pilot:** `data/gold/pending/pilot8_annotation.json` — for every element in
     `items`, fill the `triplets` field per the protocol; set the file-level
     `"status": "silver"`;
   - **main slice:** each `data/gold/pending/<input_id>.gold.json` (30 files) —
     fill `triplets`, inside each file set `"status": "silver"` and
     `"annotator_notes": "LLM-drafted (Claude Sonnet 5.5), <date>"`.
3. Finish with a short report: how many inputs annotated, total number of facts, and
   the list of `input_id`s containing facts marked `"uncertain": true`.

## Fact format (wikontic_facts_v1)

```json
{
  "subject": "...",
  "relation": "...",
  "object": "...",
  "qualifiers": [{"relation": "...", "object": "..."}],
  "subject_type": "...",
  "object_type": "..."
}
```

`subject_type`/`object_type` are optional but preferred. Do not add any other fields
to a fact (no `claim_ids`).

## Rules (condensed; full text in the protocol)

1. EXPLICITLY STATED FACTS ONLY. No world knowledge, no transitive consequences
   ("X born in Y" does NOT imply "Y is a city of X"), no artificial inverses.
2. SPLIT COORDINATIONS: "a physicist and chemist" -> two facts. Appositions and
   definitions are facts.
3. SPECIFIC Wikidata-style relations (`date of birth`, `occupation`, `award received`,
   `located in`, `developer`, `named after`, ...). Never generic `related_to` for
   everything. The object is a short, unambiguous name — never a clause.
4. Roles, membership, numbers with units, temporal restrictions, locations are facts.
5. Resolve pronouns only when the antecedent is unambiguous.
6. A qualifier belongs only to its own claim. Do not copy every date/role of the
   paragraph onto every fact.
7. A date is either the object of a standalone relation (`date of birth`) or a
   qualifier of an event (`point in time`). Do not move all values into qualifiers
   automatically.
8. Negation / modality / attribution are mandatory:
   `{"relation": "polarity", "object": "negative"}`,
   `{"relation": "modality", "object": "possible"}`,
   `{"relation": "attribution", "object": "<who>"}`.
   Never turn a possible or attributed statement into an unconditional fact.
   Never invent qualifiers the text does not support.
9. "Falsely claimed": the act of claiming is a positive fact; the content of a false
   claim is NOT a fact (either omit it, or extract with `polarity: negative` +
   `attribution`).
10. Different episodes with the same s/r/o are different facts, each with its own
    qualifiers.
11. Before saving, run the protocol checklist (guide section 13): appositions,
    coordinations, roles, numbers, time, locations, negations, attributions, events.
12. For a genuinely uncertain fact add `"uncertain": true` — do not guess.

The worked example (also in `reports/ANNOTATION_GUIDE.md`, section 3) — follow its
representation style:

Input: title "Charles-Albert Demoustier", text: "Charles-Albert Demoustier (13 March
1760 – 2 March 1801) was a French writer. He falsely claimed to be a descendant of
La Fontaine by his mother and Racine by his father."

Output triplets:
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

Note: dates are objects of standalone relations; "French writer" is one fact with a
citizenship qualifier; the false claim is recorded as a claim, and its content is
marked negative + attributed.

## Self-check before saving

For every input, review once more: (a) any fact not stated in the text? (b) any lost
polarity/attribution? (c) any overly long `object`? (d) dates copied into every fact?
Only then save the files.
