# Follow-up runbook — 25.10.2026

## OpenRouter: explicit opt-in (28.10.2026)

Add separate variables to your external credentials file (do not share values):

```dotenv
openrouter_key=YOUR_KEY
# Optional; these are the defaults:
openrouter_url=https://openrouter.ai/api/v1
openrouter_model=openai/gpt-oss-120b
```

`--openrouter` is supported by extraction `run`, `probe`, and the LLM dedup module.
Without it, behavior remains primary API only. There is **no automatic fallback**
or cross-provider retry. In OpenRouter mode generic `llm_*`, `OPENAI_*`,
`WIKONTIC_MODEL` and method-config model IDs do not select the route/model:
only OpenRouter-specific settings (or explicit `run --model` / dedup `--model`)
apply. `OPENROUTER_API_KEY` and `OPENROUTER_KEY` environment variables are accepted
and override the file. Endpoint/model overrides use `OPENROUTER_URL` and
`OPENROUTER_MODEL`. Requests consume your OpenRouter balance.

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# Small connectivity test (makes one model request):
.venv/bin/python -m wikontic_exp.cli probe --openrouter \
  --creds-file ~/.config/wikontic/creds

# AutoOntik pilot using ONLY OpenRouter:
.venv/bin/python -m wikontic_exp.cli run --openrouter \
  --creds-file ~/.config/wikontic/creds \
  --method configs/methods/autoontik_single.json \
  --inputs data/tuning/inputs.jsonl --limit 12 \
  --run-dir runs/openrouter_pilot_autoontik_single

# Merger pilot using saved university extraction outputs:
.venv/bin/python -m wikontic_exp.llm_dedup --openrouter \
  --creds-file ~/.config/wikontic/creds \
  --run-dirs runs/gptoss_tuning_coverage_single \
             runs/gptoss_tuning_baseline_fixed_single \
             runs/gptoss_tuning_fewshot_single \
  --selection-run-dir runs/gptoss_ens3_300 \
  --out-dir runs/openrouter_llm_dedup_xprompt_300
```

Use **separate run directories** so usage summaries and comparisons do not pool
university and OpenRouter responses. Route/provider and an endpoint hash (never
the key) are recorded in metadata. OpenRouter route identity is part of both
extraction and merger cache keys, even with an identical model string; historical
primary cache keys are preserved. Changing the key alone does not invalidate
successful cache entries. Use `--retry-failures` to retry failures in the same
OpenRouter run directory. OpenRouter may choose different underlying serving
providers; raw responses are retained, but exact equivalence to university
inference is not assumed.

Offline regression command: `.venv/bin/python -m pytest -q`. Tests use only
synthetic credentials. Implementation has not been executed in this environment;
no paid requests or real credential reads were performed during the change.

## Status and scope

Implemented: runtime external credentials loader, AutoOntik method configuration
and pinned snapshot downloader, conservative LLM dedup runner, blind stratified
binary judgment sampler/scorer, pair-audit export, annotation tasks and offline
regression tests. **Not yet executed or validated in this environment**: snapshot
download, pytest, AutoOntik pilot/full tuning, LLM dedup pilot/full tuning, Claude
judging. No new experimental metrics are available. Collection is not started.

API window: gpt-oss is available through 27 October (user-provided information).
The external credentials file is read by the program at runtime, never sourced
as shell code; its contents must not be printed or committed. Neither the agent
nor the offline snapshot/judging commands need to inspect it.

Run commands from the project root. Set the exact model ID explicitly because
the new credential file is only required to contain `llm_url` and `llm_key`:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export LLM_CREDS_FILE="$HOME/.config/wikontic/creds"
export WIKONTIC_MODEL='Openai/Gpt-oss-120b'
```

Confirm that the model ID above is still the endpoint's exact model ID; do not
substitute a different model silently. `llm_url` must be the OpenAI-compatible
API base URL (e.g. ending in `/v1`), not the full `/chat/completions` URL.
Explicit environment values override external credentials; external `llm_*`
values override legacy `.env`/YAML settings. `--creds-file` overrides
`LLM_CREDS_FILE`; default is `~/.config/wikontic/creds`. The minimal parser accepts
KEY=VALUE / KEY: VALUE and optional `export`; it does not execute substitutions.

## 1. Offline setup and regression checks

```bash
.venv/bin/python scripts/snapshot_autoontik.py
.venv/bin/python -m pytest -q
```

The snapshot script downloads five verbatim prompts at pinned commit
`9368d6149050f3085a9450e03e4d4c1052722525`, checks Git blob IDs, saves SHA-256 hashes
in `external/autoontik_source_manifest.json`, and generates
`reports/autoontik_vs_baseline.diff`. Existing different files are not overwritten.
It reads no credentials and contacts only GitHub. The new credential tests use
synthetic temporary files, not the real credential path.

Do not continue to long API runs if these checks fail.

## 2. AutoOntik: pilot, inspect, then full tuning

```bash
.venv/bin/python -m wikontic_exp.cli run \
  --method configs/methods/autoontik_single.json \
  --inputs data/tuning/inputs.jsonl --limit 12 \
  --run-dir runs/gptoss_pilot_autoontik_single
```

Inspect `status_summary.json` and response status/error fields. Upstream has a
list-vs-object ambiguity; our parser accepts a top-level object only. Preserve
verbatim prompt failures as results; create a separately named fixed adaptation
if needed. Pilot and other methods must use the same input IDs when comparing
pilot quality. This command uses the first 12 tuning inputs, not an assumed
historical pilot selection. Do not treat pilot numbers as full tuning results.

After pilot validation:

```bash
.venv/bin/python -m wikontic_exp.cli run \
  --method configs/methods/autoontik_single.json \
  --inputs data/tuning/inputs.jsonl \
  --run-dir runs/gptoss_tuning_autoontik_single
```

## 3. LLM dedup pilot: same input IDs as C2, original three prompt outputs

```bash
.venv/bin/python -m wikontic_exp.llm_dedup \
  --run-dirs runs/gptoss_tuning_coverage_single \
             runs/gptoss_tuning_baseline_fixed_single \
             runs/gptoss_tuning_fewshot_single \
  --selection-run-dir runs/gptoss_ens3_300 \
  --out-dir runs/llm_dedup_xprompt_300
```

This deduplicates the cross-prompt ensemble on C2's 300 input IDs; it does NOT
silently add the three C2 generations or AutoOntik to that ensemble. Run C2
itself as a distinct experiment if desired:

```bash
.venv/bin/python -m wikontic_exp.llm_dedup \
  --run-dirs runs/gptoss_ens3_300 \
  --out-dir runs/llm_dedup_coverage3_300
```

The runner passes every original candidate in an input, without fuzzy blocking
or lossy normalized pre-merges. One extra merger call per paragraph, separate
from extraction. It returns a strict partition of integer candidate IDs and an
existing representative; no synthesized facts or qualifier unions. All originals
and provenance are retained. `n_support` means complete-assertion support, not
the historical merger's possibly broader core-plus-compatible-qualifier support.
This change in semantics must accompany tables comparing support histograms.

Caches include version, model, generation parameters, prompt, paragraph,
candidates and source metadata. Raw responses are saved before parsing. Missing,
duplicate, extra IDs and truncations are failures, never heuristic fallback.
Permanent API errors halt subsequent batches after saving the current batch;
any unprocessed inputs are listed explicitly.
Use the same command to resume; `--retry-failures` explicitly retries failures.
New model/prompt/parameters produce new cache keys. Outputs:
`merged_union.json`, `merged_vote2.json`, `selection.json`, and `responses/`.
Pairwise heuristic agreement in diagnostics is NOT quality validation.

### Audit merger decisions before full tuning

```bash
.venv/bin/python scripts/prepare_merger_audit.py \
  --dedup-dir runs/llm_dedup_xprompt_300 \
  --out-dir data/judging/merger_audit --n-per-stratum 30
```

Give the annotator `data/judging/MERGER_AUDIT_TASK.md`, not `runs/` or the private
manifest. Audit 30 proposed merged and 30 proposed non-merged pairs (or all if a
stratum is smaller); inspect wrong merges especially across polarity, episode
and qualifier scope. The balanced strata are not a representative global pair
distribution. This requires separate judgments from the 0/1 fact-correctness task.
An LLM audit is still silver; human checks would provide stronger evidence.

```bash
.venv/bin/python scripts/score_merger_audit.py \
  --manifest data/judging/merger_audit/audit_private.json \
  --judgments data/judging/merger_audit/judgments.jsonl \
  --out reports/merger_audit_round1.json
```

Report wrong-merge rate among proposed merges and missed-duplicate rate among
proposed non-merges separately, with their sampling intervals. Do not identify
the judge's decision with human ground truth.

After pilot/audit acceptance, full tuning (not collection):

```bash
.venv/bin/python -m wikontic_exp.llm_dedup \
  --run-dirs runs/gptoss_tuning_coverage_single \
             runs/gptoss_tuning_baseline_fixed_single \
             runs/gptoss_tuning_fewshot_single \
  --out-dir runs/llm_dedup_xprompt_300
```

Reusing the directory reuses cache for pilot inputs and computes the rest;
`selection.json` then describes the full selection. Preserve the pilot selection
separately before extending if an annotation round depends on it, or use a new
directory for full tuning. Do not extend while Claude is judging a pilot round.

## 4. Blind binary precision evaluation on the SAME 300 inputs

Run after AutoOntik extraction and LLM pilot; retain pilot selection unchanged:

```bash
.venv/bin/python -m wikontic_exp.binary_judging prepare \
  --selection runs/llm_dedup_xprompt_300/selection.json \
  --methods coverage=runs/gptoss_tuning_coverage_single \
    baseline_fixed=runs/gptoss_tuning_baseline_fixed_single \
    fewshot=runs/gptoss_tuning_fewshot_single \
    autoontik=runs/gptoss_tuning_autoontik_single \
    heuristic_union=runs/ensemble_xprompt/merged_union.json \
    heuristic_vote2=runs/ensemble_xprompt/merged_vote2.json \
    llm_union=runs/llm_dedup_xprompt_300/merged_union.json \
    llm_vote2=runs/llm_dedup_xprompt_300/merged_vote2.json \
  --n 150 --seed 42 --out-dir data/judging/round1
```

Population = actual prediction occurrences for each method on selected inputs;
no prior fuzzy dedup. Strata = presence of qualifiers × n_support, approximately
proportional allocation with at least one example in each nonempty stratum.
Exactly identical facts from the same input can share a blind annotation, but
their sampled occurrences and population weights remain distinct. The private
manifest stores source mappings, frame hashes and sample sizes; the annotator
receives only title + paragraph + fact + opaque ID in `candidates.jsonl`.
No QA fields, method names, support counts or old silver annotations are shown.

Give Claude the path `data/judging/ANNOTATION_TASK.md`. The task is English and
references only files in this project. Use a separate session. Labels remain
`silver`, never `gold`. No extraction API is used by prepare/score.

```bash
.venv/bin/python -m wikontic_exp.binary_judging score \
  --manifest data/judging/round1/sampling_private.json \
  --judgments data/judging/round1/judgments.jsonl \
  --out reports/binary_precision_round1.json
```

Main metric: population-weighted `full_label` judged precision; diagnostic:
`core_label` precision and stratum precision. Missing labels withhold the method's
estimate, rather than count as zeros. Report coverage (`n_inputs_with_predictions`
vs `n_evaluation_inputs`) alongside precision: failed/empty extraction is not
measured by prediction precision. Compare output counts on the same inputs.

The scorer uses Wilson intervals within strata and a conservative weighted
Bonferroni/Wilson envelope for overall precision, not an incorrect pooled Wilson
interval for unequal sampling fractions. These are approximate sampling intervals,
not a measure of Claude's systematic error; repeated facts/paragraphs introduce
dependence. A 150-fact sample is exploratory and may be insufficient to decide
a precision threshold near 0.95. No recall or F1 is inferred from binary labels.
Do not tune prompts on these labels and call a rescore an independent test.

## Reporting corrections

- SequenceMatcher + hand-set threshold is a project heuristic control, not a
  published or validated semantic matcher.
- Shared references do not guarantee stable method ranking; previous claims
  that P/R are guaranteed lower bounds and ranking is reliable were too strong.
  Reference omissions, reference errors, and accidental string matches can
  distort both absolute metrics and ranking.
- Old silver P/R/F1 remain historical, matcher-dependent measurements. The new
  main comparison uses direct binary judgments of predictions, without matching
  strings to an independently enumerated reference set.
- The provisional union decision is reopened pending these checks. AutoOntik is
  a separate single-pass method first; adding it to an ensemble changes the
  number of extractor calls and requires a separately labeled comparison.
- Collection-1000 still requires explicit authorization; no command here starts it.
