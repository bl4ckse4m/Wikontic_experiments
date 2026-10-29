"""Prepare a blind random pair audit, stratified by LLM merged/not-merged.

Judge actual equivalence independently; heuristic disagreement is NOT correctness.
No API calls. Run after the LLM dedup pilot. Does not overwrite annotation work.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dedup-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-per-stratum", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.n_per_stratum < 1:
        p.error("n-per-stratum must be positive")
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty audit directory")
    strata = defaultdict(list)
    root = Path(args.dedup_dir)
    merged = json.loads((root / "merged_union.json").read_text())
    for path in sorted((root / "responses").glob("*.json")):
        rec = json.loads(path.read_text())
        if rec.get("status") != "ok" or rec["input_id"] not in merged["results"]:
            continue
        # Ignore old cache records from other prompt/model configurations.
        current = merged["results"][rec["input_id"]]
        current_key = current[0].get("dedup_cache_key") if current else None
        if current_key and current_key != rec["cache_key"]:
            continue
        labels = {i: g for g, group in enumerate(rec["groups"]) for i in group["members"]}
        payload = rec["payload"]
        cands = payload["candidates"]
        for i in range(len(cands)):
            for j in range(i):
                same = labels[i] == labels[j]
                strata[same].append({"input_id": rec["input_id"], "title": payload["title"],
                                     "paragraph_text": payload["paragraph_text"],
                                     "a": cands[j]["fact"], "b": cands[i]["fact"],
                                     "source_key": rec["cache_key"], "pair": [j, i]})
    rng = random.Random(args.seed)
    public, private = [], []
    for same in (True, False):
        pool = strata[same]
        for row in rng.sample(pool, min(args.n_per_stratum, len(pool))):
            audit_id = f"pair-{len(private):04d}"
            public.append({"audit_id": audit_id, **{k: row[k] for k in ("title", "paragraph_text", "a", "b")}})
            private.append({"audit_id": audit_id, "llm_same": same, "input_id": row["input_id"],
                            "source_key": row["source_key"], "pair": row["pair"]})
    if not public:
        raise ValueError("No completed dedup pairs available")
    rng.shuffle(public)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pairs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in public))
    (out / "audit_private.json").write_text(json.dumps({
        "seed": args.seed, "population_pairs": {str(k): len(v) for k, v in strata.items()},
        "pairs": private}, ensure_ascii=False, indent=2))
    print(f"Prepared {len(public)} blind pair judgments")


if __name__ == "__main__":
    main()
