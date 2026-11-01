# E08 rubric calibration — not the main evaluation

Use a fresh annotation session. These are development examples. Do not inspect any private manifest, report, source code, or previous judgments.

# Blind equivalence audit (SILVER, separate from fact correctness)

Read only this task and `data/judging/e08/calibration/merger_audit/pairs.jsonl` within
`/home/alexey/PycharmProjects/Wikontic_experiments`.
Do not read `runs/`, private manifests, method prompts or other annotations.
Write `data/judging/e08/calibration/merger_audit/judgments.jsonl`, preserving existing work.

For each pair, use only the titled paragraph. Decide whether A and B express
the SAME COMPLETE assertion: synonymous entities/predicates and grammatical
inverses are allowed; every meaningful qualifier must be equivalent. Different
dates/episodes, scope, polarity, modality, attribution, missing meaningful
conditions, broader/narrower claims, and one-way entailment are NOT equivalence.
If unsure, do not assert equivalence and explain why. Assess equivalence, not
truth: two equally unsupported assertions may still be duplicates.

Return integer `equivalent` 0 or 1, short rationale and annotator model name.
Set status `silver` (not gold). No new facts, no method guesses.
{"audit_id":"pair-0000","equivalent":1,"rationale":"...","annotator":"...","status":"silver"}

Analysis must distinguish error rates among proposed merges and proposed
non-merges; the two strata have different sampling fractions. Do not compute
unweighted global accuracy from the balanced sample. This audit is preliminary
and does not establish fact precision or recall.
