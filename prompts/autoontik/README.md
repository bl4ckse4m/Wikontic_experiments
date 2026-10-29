# Pinned AutoOntik prompts

Run `python scripts/snapshot_autoontik.py` from the project root to materialize
five byte-for-byte upstream prompts (commit
`9368d6149050f3085a9450e03e4d4c1052722525`). The script checks Git blob hashes,
records SHA-256 hashes in `external/autoontik_source_manifest.json`, and writes
`reports/autoontik_vs_baseline.diff`. It never loads credentials.

`autoontik_single.json` uses the extraction prompt verbatim, including its
list-vs-object ambiguity. Our existing parser requires a top-level object; a
top-level list is recorded as a schema failure, not silently repaired. If changes prove
necessary, create a separately named adaptation and diff; do not edit the snapshot.

Merger and judging protocols are project adaptations, not verbatim upstream
methods. Upstream `triple_match` ignores qualifiers and chooses one match; it is
not itself a complete-assertion deduplicator. Upstream groundedness uses isolated
sentences without titles; our judge uses the entire titled paragraph and checks
qualifiers. Do not claim those are equivalent experiments or published validation
of the SequenceMatcher heuristic. Similarities to our Wikontic baseline do not
establish historical code lineage.
