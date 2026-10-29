"""Cross-method fact overlap and ensemble marginal-gain analysis.

Usage:
  python scripts/analyze_overlap.py --run-dirs runs/A runs/B runs/C --out reports/overlap.json
  python scripts/analyze_overlap.py --run-dirs runs/ens3 --out reports/ensemble_gain.json --ensemble

--ensemble mode reports the marginal gain of independent generations (gen_index 0,1,2):
cumulative unique facts after adding each generation, plus vote-support histogram.
All matching is semantic (normalized core + qualifier compatibility), same rules as
wikontic_exp.aggregation. Output is analysis only, no method selection is declared.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wikontic_exp.aggregation import cluster_same_claim, load_candidates, load_run_records  # noqa: E402


def group_by_input(cands):
    by = defaultdict(list)
    for c in cands:
        by[c.source["input_id"]].append(c)
    return by


def dedup_facts(cands):
    return cluster_same_claim(cands)


def method_overlap(cands, run_dirs):
    # methods identified by prompt_hash prefix (stable across runs of the same prompt)
    labels = sorted({c.source["prompt_hash"][:8] for c in cands})
    by = group_by_input(cands)
    per_method_unique = defaultdict(int)
    shared_all = 0
    pair_overlap = defaultdict(int)
    total_clusters = 0
    for iid, cs in by.items():
        clusters = dedup_facts(cs)
        total_clusters += len(clusters)
        for cl in clusters:
            methods = {c.source["prompt_hash"][:8] for c in cl}
            if len(methods) == 1:
                per_method_unique[next(iter(methods))] += 1
            if len(methods) == len(labels):
                shared_all += 1
            for m in methods:
                for m2 in methods:
                    if m < m2:
                        pair_overlap[f"{m}|{m2}"] += 1
    return {
        "n_inputs": len(by),
        "n_semantic_fact_clusters": total_clusters,
        "prompt_hash_prefixes": labels,
        "unique_to_one_method": dict(per_method_unique),
        "present_in_all_methods": shared_all,
        "pairwise_shared_clusters": dict(pair_overlap),
        "n_run_dirs": len(run_dirs),
    }


def ensemble_gain(cands):
    by = group_by_input(cands)
    order = [0, 1, 2, 3]
    cum_unique = {g: 0 for g in order}
    union_total = 0
    support_hist = defaultdict(int)
    per_input_union = []
    for iid, cs in by.items():
        clusters = dedup_facts(cs)
        union_total += len(clusters)
        per_input_union.append(len(clusters))
        for cl in clusters:
            gens = {c.source["gen_index"] for c in cl}
            support_hist[len(gens)] += 1
        # cumulative unique facts when adding generations in order
        for g in order:
            sub = [c for c in cs if c.source["gen_index"] <= g]
            if not sub:
                continue
            cum_unique[g] += len(dedup_facts(sub))
    gen_counts = defaultdict(int)
    for c in cands:
        gen_counts[c.source["gen_index"]] += 1
    return {
        "n_inputs": len(by),
        "n_generations_seen": sorted(gen_counts),
        "candidates_per_gen": dict(gen_counts),
        "union_total_clusters": union_total,
        "cumulative_unique_facts_by_max_gen_index": cum_unique,
        "support_histogram": {str(k): v for k, v in sorted(support_hist.items())},
        "note": "cumulative_* : unique semantic facts after including generations 0..g; "
                "support_histogram: how many clusters are backed by exactly k independent generations",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ensemble", action="store_true")
    args = ap.parse_args()
    records = []
    for rd in args.run_dirs:
        records.extend(load_run_records(Path(rd)))
    cands = load_candidates(records)
    out = {"n_records": len(records), "n_candidates": len(cands)}
    if args.ensemble:
        out["ensemble_gain"] = ensemble_gain(cands)
    else:
        out["method_overlap"] = method_overlap(cands, args.run_dirs)
        out["ensemble_gain_if_same_prompt"] = ensemble_gain(cands) if any(
            c.source["gen_index"] > 0 for c in cands) else None
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
