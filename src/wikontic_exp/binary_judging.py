"""Blind sampling of actual predictions and stratified binary silver precision.

python -m wikontic_exp.binary_judging prepare --help
python -m wikontic_exp.binary_judging score --help
No model calls or credential access. Annotators see only candidates.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

from .aggregation import load_run_records
from .hashing import hash_obj
from .runner import load_inputs
from .llm_dedup import write_json


def prediction_rows(path: Path) -> list[dict]:
    """Preserve every prediction occurrence; don't pre-dedup with fuzzy matching."""
    rows = []
    if path.is_dir():
        for r in load_run_records(path):
            if r.get("mock"):
                raise ValueError("Mock predictions cannot be evaluated as experiments")
            if r.get("status") == "ok" and r.get("parsed"):
                for i, fact in enumerate(r["parsed"]["triplets"]):
                    rows.append({"input_id": r["task"]["input_id"], "fact": fact,
                                 "source": {"cache_key": r["cache_key"], "fact_index": i},
                                 "n_support": 1})
    else:
        data = json.loads(path.read_text())
        if "results" not in data:
            raise ValueError("Expected merged results JSON or extraction run directory")
        for iid, facts in sorted(data["results"].items()):
            for i, merged in enumerate(facts):
                rows.append({"input_id": iid, "fact": merged["fact"],
                             "source": {"fact_index": i, "sources": merged.get("sources", [])},
                             "n_support": merged.get("n_support", 1)})
    for row in rows:
        row["stratum"] = f"support={row['n_support']};qualifiers={int(bool(row['fact'].get('qualifiers')))}"
    return sorted(rows, key=lambda r: (r["input_id"], hash_obj(r)))


def allocation(sizes: dict[str, int], n: int) -> dict[str, int]:
    """At least one per nonempty stratum, then approximately proportional."""
    n = min(n, sum(sizes.values()))
    if n < len(sizes):
        raise ValueError("Sample size must cover every nonempty stratum")
    allocated = {s: 1 for s in sizes}
    while sum(allocated.values()) < n:
        s = max((s for s in sizes if allocated[s] < sizes[s]),
                key=lambda s: (sizes[s] / (allocated[s] + 1), s))
        allocated[s] += 1
    return allocated


def prepare(args: argparse.Namespace) -> dict:
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty annotation directory; existing work is not overwritten")
    inputs = {r["input_id"]: r for r in load_inputs(args.inputs)}
    selection = getattr(args, "selection", None)
    selected_ids = None
    if selection:
        selected_ids = set(json.loads(Path(selection).read_text())["input_ids"])
        if not selected_ids or selected_ids - set(inputs):
            raise ValueError("Invalid selected input IDs")
    pool = {}
    methods = {}
    for spec in args.methods:
        name, raw_path = spec.split("=", 1)
        if name in methods:
            raise ValueError("Duplicate method name")
        rows = prediction_rows(Path(raw_path))
        if selected_ids is not None:
            rows = [r for r in rows if r["input_id"] in selected_ids]
        if not rows:
            raise ValueError(f"No predictions for {name}")
        strata = defaultdict(list)
        for row in rows:
            if row["input_id"] not in inputs:
                raise ValueError("Prediction has no source paragraph")
            strata[row["stratum"]].append(row)
        sizes = {s: len(rs) for s, rs in sorted(strata.items())}
        counts = allocation(sizes, args.n)
        samples = []
        for s in sizes:
            rng = random.Random(hash_obj({"seed": args.seed, "method": name, "stratum": s}))
            for row in rng.sample(strata[s], counts[s]):
                iid = row["input_id"]
                annotation_id = hash_obj({"input_id": iid, "fact": row["fact"]})
                pool[annotation_id] = {"annotation_id": annotation_id, "input_id": iid,
                                       "title": inputs[iid]["title"],
                                       "paragraph_text": inputs[iid]["paragraph_text"],
                                       "fact": row["fact"]}
                samples.append(dict(row, annotation_id=annotation_id))
        methods[name] = {"path": raw_path, "population": len(rows), "sizes": sizes,
                         "sample_counts": counts, "samples": samples,
                         "population_hash": hash_obj(rows),
                         "n_evaluation_inputs": len(selected_ids) if selected_ids is not None else len(inputs),
                         "n_inputs_with_predictions": len({r["input_id"] for r in rows})}
    out.mkdir(parents=True, exist_ok=True)
    # Shuffle blind IDs independently of method; same exact fact in same input is judged once.
    candidates = [pool[k] for k in sorted(pool)]
    random.Random(args.seed).shuffle(candidates)
    (out / "candidates.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in candidates))
    write_json(out / "sampling_private.json", {"version": "binary-silver-1.0", "seed": args.seed,
               "n_requested_per_method": args.n, "methods": methods,
               "note": "Annotator must not open this file; strata weighted by population, not equally"})
    return {"methods": len(methods), "unique_facts_to_judge": len(pool), "out_dir": str(out)}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if not n:
        raise ValueError("Cannot score empty sample")
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    delta = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, center - delta), min(1.0, center + delta)]


def score(args: argparse.Namespace) -> dict:
    manifest = json.loads(Path(args.manifest).read_text())
    expected = {r["annotation_id"] for m in manifest["methods"].values() for r in m["samples"]}
    judgments = {}
    for row in load_inputs(args.judgments):
        aid = row.get("annotation_id")
        if aid not in expected or aid in judgments:
            raise ValueError("Unknown or duplicate annotation_id")
        if row.get("status") != "silver":
            raise ValueError("LLM judgments must have status=silver")
        if (not row.get("annotator") or not row.get("rationale")
                or not isinstance(row.get("evidence"), str)):
            raise ValueError("Annotator, rationale and evidence string are required")
        for label in ("core_label", "full_label"):
            if type(row.get(label)) is not int or row[label] not in (0, 1):
                raise ValueError("Labels must be integers 0 or 1")
        if row["full_label"] > row["core_label"]:
            raise ValueError("Full assertion cannot be correct when core is incorrect")
        judgments[aid] = row
    report = {"quality_status": "LLM-assisted silver; preliminary", "recall": "not measured",
              "interval_note": "Population-weighted stratified estimate. Per-stratum Wilson intervals; overall envelope uses Bonferroni-adjusted Wilson bounds. Approximate, assumes observations independent within strata; repeated facts/paragraphs and LLM systematic errors are not captured.",
              "methods": {}}
    for name, method in manifest["methods"].items():
        samples = method["samples"]
        missing = sum(r["annotation_id"] not in judgments for r in samples)
        result = {"population": method["population"], "sampled": len(samples), "missing": missing,
                  "n_evaluation_inputs": method.get("n_evaluation_inputs"),
                  "n_inputs_with_predictions": method["n_inputs_with_predictions"]}
        if missing:
            result["status"] = "incomplete; precision withheld (missing labels are not negatives)"
            report["methods"][name] = result
            continue
        z = NormalDist().inv_cdf(1 - 0.05 / (2 * len(method["sizes"])))
        result["strata"] = {}
        for label in ("core_label", "full_label"):
            estimate = lo = hi = 0.0
            for s, population in method["sizes"].items():
                selected = [r for r in samples if r["stratum"] == s]
                n = len(selected)
                k = sum(judgments[r["annotation_id"]][label] for r in selected)
                weight = population / method["population"]
                bounds = wilson(k, n, z) if n < population else [k / n, k / n]
                estimate += weight * k / n
                lo += weight * bounds[0]
                hi += weight * bounds[1]
                result["strata"].setdefault(s, {})[label] = {
                    "population": population, "n": n, "positive": k,
                    "precision": k / n, "wilson95": wilson(k, n) if n < population else bounds}
            result[label] = {"precision": estimate, "approx_interval95": [lo, hi]}
        report["methods"][name] = result
    write_json(Path(args.out), report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sp = sub.add_parser("prepare")
    sp.add_argument("--methods", nargs="+", required=True, help="NAME=run_directory_or_merged_json")
    sp.add_argument("--inputs", default="data/tuning/inputs.jsonl")
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--n", type=int, default=150, help="prediction occurrences per method")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--selection", default=None, help="selection.json from LLM dedup; compare same inputs")
    sp.set_defaults(func=prepare)
    sp = sub.add_parser("score")
    sp.add_argument("--manifest", required=True)
    sp.add_argument("--judgments", required=True)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=score)
    args = p.parse_args()
    print(json.dumps(args.func(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
