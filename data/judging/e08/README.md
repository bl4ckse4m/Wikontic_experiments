# E08 annotation staging (organizer only)

Current stage: offline preparation, then rubric calibration. Main evaluation is
not prepared yet. Do not send this README, reports, source code, runs, locks,
private manifests, or old annotations to the annotator.

From the project root, with `src` on `PYTHONPATH`:

```bash
.venv/bin/python -m pytest -q && \
  .venv/bin/python -m wikontic_exp.e08_preparation prepare && \
  .venv/bin/python -m wikontic_exp.e08_preparation verify
```

These commands make no model requests and load no credentials. The generator
has been implemented but not executed by the assistant; generated files must
not be assumed to exist until the command succeeds.

After successful preparation, give two fresh Claude sessions these tasks,
one task per session:

1. `data/judging/e08/calibration/binary/TASK.md` — 24 original extracted facts.
2. `data/judging/e08/calibration/merger_audit/TASK.md` — 24 original candidate pairs.

Each task names its allowed local data file and output judgments file. No
expected labels, method identities, support counts, or merger decisions appear
in the public candidate/pair rows. Require the actual annotator model name and
`status: silver`, never gold. Keep the sessions blind to each other's judgments.

The private calibration manifest maps items to preserved E07 source candidates;
it is not a reference answer key. The selected cases cover supported/unsupported
claims, scope/negation, role restrictions, temporal qualifiers, approximation,
inverse relations, and redundant units. They are curated development examples,
not a random sample for reporting method quality. Review all 48 judgments with
a human; do not aggregate them into E08's main precision or merger error rates.

After review, record the exact Claude model, annotation settings/mode, any rubric
clarification, and final task hashes in a separate rubric lock before starting
the main experiment. Preserve the initial tasks and selection lock. No command
here authorizes collection-1000 or automatically launches merger requests.
