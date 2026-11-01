# LLM merger prompt versions

As of 2026-10-30, v2 is frozen as a candidate for E08; do not create a v3 or
extend the development pilots automatically. See `reports/EXPERIMENTS.md` and
`reports/E08_PROTOCOL.md`. The same 20-input v2 pilot has now run: 19 valid
partitions and one truncation; semantic superiority has not been established.

- `llm_dedup_v1.txt`: original pilot prompt; preserved unchanged.
- `llm_dedup_v2.txt`: candidate revision after inspection of the 18 structurally
  successful outputs from the initial 20-input pilot. No quality improvement has
  yet been measured.

V2 distinguishes redundant representation from substantive qualifiers, stresses
bidirectional equivalence and scope/polarity preservation, and asks for a final
cross-group duplicate and ID-coverage review. The runner still validates the
partition mechanically; prompt instructions cannot guarantee model compliance.
Ambiguous negative encodings are not repaired. The representative must be an
original equivalent member, not a richer assertion used to justify a loose merge.

Select v2 explicitly with `--prompt prompts/merger/llm_dedup_v2.txt`.
The runner's default remains v1 to avoid silently changing existing experiments.
Use a separate output directory for v2. Prompt contents participate in cache keys,
but separate directories also keep exported aggregates and usage records distinct.

Compare versions on the original 20 inputs for diagnosis, then on fresh inputs.
The original 20 influenced this revision and are not a held-out test. Use the
same equivalence rubric and include both proposed merges and proposed non-merges
in the audit. Assess extraction correctness separately. A successful JSON/partition
is not a correct semantic clustering.
