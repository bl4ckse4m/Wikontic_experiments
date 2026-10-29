"""One-shot comparison of extraction methods (stage D-7).

Computes per-method proxy metrics, cost, cross-method semantic overlap, optional
gold metrics (from annotated files), and writes a markdown + JSON report.

Usage:
  python scripts/compare_methods.py \
    --methods coverage=runs/gptoss_tuning_coverage_single \
              baseline_fixed=runs/gptoss_tuning_baseline_fixed_single \
              fewshot=runs/gptoss_tuning_fewshot_single \
    --inputs data/tuning/inputs.jsonl \
    --gold-dir data/gold/pending \
    --out reports/tuning_comparison.json

Gold files: any *.gold.json with "status": "annotated" and non-empty "triplets"
is used; unannotated files are ignored (report stays PROXY-only).
Method choice is NOT declared here: the report is input for the decision, which
must state the precision threshold and be marked preliminary until gold exists.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wikontic_exp.aggregation import Candidate, cluster_same_claim, load_candidates, load_run_records  # noqa: E402
from wikontic_exp.evaluation import compactness, gold_eval, proxy_report, text_support_proxy  # noqa: E402
from wikontic_exp.hashing import sha256_hex  # noqa: E402
from wikontic_exp.runner import aggregate_usage, load_inputs  # noqa: E402


def load_method(path: Path, label: str):
    """Load a method from a run directory (responses/) or a merged aggregate .json."""
    if path.is_dir():
        records = load_run_records(path)
        return load_candidates(records), records, "run"
    data = json.loads(path.read_text(encoding="utf-8"))
    ph = sha256_hex(label)[:8]
    cands = []
    for iid, groups in data.get("results", {}).items():
        for g in groups:
            cands.append(Candidate(fact=g["fact"], source={
                "input_id": iid, "model": "ensemble", "prompt_hash": ph,
                "gen_index": 0, "cache_key": (g.get("sources") or [{}])[0].get("cache_key", ""),
            }))
    # pseudo-records so proxy_report works on merged facts too
    by = defaultdict(list)
    for c in cands:
        by[c.source["input_id"]].append(c.fact)
    records = [{"status": "ok", "task": {"input_id": iid}, "parsed": {"triplets": fs}}
               for iid, fs in by.items()]
    return cands, records, "merged"


def semantic_fact_count(cands) -> int:
    """Per-input semantic dedup (never across inputs)."""
    by = defaultdict(list)
    for c in cands:
        by[c.source["input_id"]].append(c)
    return sum(len(cluster_same_claim(g)) for g in by.values())


def load_annotated_gold(gold_dir: Path):
    """Load verified annotation files.

    status "annotated" = human-authored gold;
    status "silver"    = LLM-assisted annotation, human-verified at least in part.
    Both are usable for metrics but reported separately; silver is NOT gold."""
    gold, kinds = {}, {}
    if not gold_dir or not gold_dir.exists():
        return gold, kinds
    for p in sorted(gold_dir.glob("*.gold.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        st = data.get("status")
        if st in ("annotated", "silver") and data.get("triplets"):
            gold[data["input_id"]] = data["triplets"]
            kinds[data["input_id"]] = st
    # combined sheets (e.g. pilot8_annotation.json with "items")
    for p in sorted(gold_dir.glob("*.json")):
        if p.name.endswith(".gold.json"):
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        st = data.get("status")
        if st not in ("annotated", "silver"):
            continue
        for item in data.get("items", []):
            if item.get("triplets"):
                gold[item["input_id"]] = item["triplets"]
                kinds[item["input_id"]] = st
    return gold, kinds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True,
                    help="label=run_dir pairs")
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--gold-dir", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    methods = []
    for spec in args.methods:
        label, _, rd = spec.partition("=")
        methods.append((label, Path(rd)))

    texts = {r["input_id"]: r for r in load_inputs(args.inputs)}
    gold, gold_kinds = load_annotated_gold(Path(args.gold_dir)) if args.gold_dir else ({}, {})
    n_human = sum(1 for k in gold_kinds.values() if k == "annotated")
    n_silver = sum(1 for k in gold_kinds.values() if k == "silver")

    report = {
        "note": (("GOLD/SILVER + proxy comparison: "
                  f"{n_human} human-annotated, {n_silver} LLM-assisted silver "
                  "(silver is not gold; method choice stays preliminary)")
                 if gold else "PROXY comparison only (no annotated gold)"),
        "gold_kinds": gold_kinds,
        "methods": {},
        "cross_method_overlap": {},
    }

    per_method_cands = {}
    for label, rd in methods:
        cands, records, kind = load_method(rd, label)
        per_method_cands[label] = cands
        preds = [c.fact for c in cands]
        usage = aggregate_usage(rd) if rd.is_dir() else {"note": "merged aggregate: usage of constituent runs"}
        entry = {
            "kind": kind,
            "proxy": proxy_report(records, texts),
            "compactness": compactness(preds),
            "usage": usage,
            "n_semantic_facts": semantic_fact_count(cands),
        }
        if gold:
            by_input_preds = defaultdict(list)
            for c in cands:
                by_input_preds[c.source["input_id"]].append(c.fact)
            # gold eval on the intersection of annotated inputs present in this run
            all_preds, all_golds = [], []
            for iid, g in gold.items():
                all_preds.extend(by_input_preds.get(iid, []))
                all_golds.extend(g)
            entry["gold"] = gold_eval(all_preds, all_golds) if all_golds else None
            entry["gold_n_inputs"] = len(gold)
        report["methods"][label] = entry

    # cross-method semantic overlap on shared inputs
    labels = [l for l, _ in methods]
    by_input_per_method = {}
    for label, _ in methods:
        g = defaultdict(list)
        for c in per_method_cands[label]:
            g[c.source["input_id"]].append(c)
        by_input_per_method[label] = g
    shared_inputs = None
    for label, _ in methods:
        ids = set(by_input_per_method[label])
        shared_inputs = ids if shared_inputs is None else (shared_inputs & ids)
    unique_to = defaultdict(int)
    in_all = 0
    total_clusters = 0
    pair_shared = defaultdict(int)
    for n_done, iid in enumerate(shared_inputs or [], 1):
        cands = []
        for label, _ in methods:
            cands.extend(by_input_per_method[label].get(iid, []))
        for cl in cluster_same_claim(cands):
            total_clusters += 1
            present = set()
            for lbl in labels:
                ph = _prompt_hash(per_method_cands[lbl])
                if any(c.source["prompt_hash"] == ph for c in cl):
                    present.add(lbl)
            if len(present) == 1:
                unique_to[next(iter(present))] += 1
            if len(present) == len(labels):
                in_all += 1
            for a in sorted(present):
                for b in sorted(present):
                    if a < b:
                        pair_shared[f"{a}|{b}"] += 1
        if n_done % 200 == 0:
            print(f"overlap: {n_done}/{len(shared_inputs or [])} inputs...", flush=True)
    report["cross_method_overlap"] = {
        "n_shared_inputs": len(shared_inputs or []),
        "n_semantic_fact_clusters": total_clusters,
        "unique_to_one_method": dict(unique_to),
        "present_in_all_methods": in_all,
        "pairwise_shared_clusters": dict(pair_shared),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # markdown summary
    title = ("# Comparison (proxy + gold/silver)" if gold else "# Comparison (proxy only)")
    if gold:
        title += f"\n\nannotation: {n_human} human gold, {n_silver} LLM-assisted silver (silver != gold)"
    lines = [title, ""]
    lines.append("| method | ok_rate | n_facts | sem.facts | lex.support | compl.tok | facts/1kchars |" +
                 (" gold P/R/F1 core | gold P/R/F1 full |" if gold else ""))
    lines.append("|---|---|---|---|---|---|---|" + ("---|---|" if gold else ""))
    for label in labels:
        e = report["methods"][label]
        p = e["proxy"]
        u = e["usage"]
        tok = str(u["completion_tokens"] + u.get("completion_tokens_est", 0)) if "completion_tokens" in u else "—"
        row = (f"| {label} | {p['ok_rate']:.3f} | {p['n_facts']} | {e['n_semantic_facts']} | "
               f"{p['lexical_support_rate']:.3f} | {tok} | "
               f"{e['compactness']['facts_per_1k_chars']:.2f} |")
        if gold and e.get("gold"):
            g = e["gold"]["core"]
            f = e["gold"]["full_statements"]
            row += f" {g['precision']:.3f}/{g['recall']:.3f}/{g['f1']:.3f} | {f['precision']:.3f}/{f['recall']:.3f}/{f['f1']:.3f} |"
        lines.append(row)
    lines.append("")
    lines.append("Overlap: " + json.dumps(report["cross_method_overlap"], ensure_ascii=False))
    md = out.with_suffix(".md")
    md.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {out} and {md}")


def _prompt_hash(cands):
    return cands[0].source["prompt_hash"] if cands else ""


if __name__ == "__main__":
    main()
