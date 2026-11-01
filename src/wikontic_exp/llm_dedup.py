"""Conservative LLM equivalence clustering, independent of extraction and judging.

Run with python -m wikontic_exp.llm_dedup. No API call occurs on import.
Failures are explicit, never silently replaced by heuristic clusters.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from .aggregation import load_candidates, load_run_records, same_claim
from .api_client import AccessConfig, OpenAICompatClient
from .hashing import hash_obj
from .runner import load_inputs

VERSION = "llm-equivalence-1.0"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def validate_partition(answer: dict, n: int) -> list[dict]:
    if not isinstance(answer, dict) or not isinstance(answer.get("groups"), list):
        raise ValueError("Expected groups array")
    seen = []
    for group in answer["groups"]:
        if not isinstance(group, dict):
            raise ValueError("Group must be an object")
        members = group.get("members")
        rep = group.get("representative")
        if (not isinstance(members, list) or not members
                or any(type(i) is not int for i in members)
                or type(rep) is not int or rep not in members):
            raise ValueError("Invalid members or representative")
        seen.extend(members)
    if sorted(seen) != list(range(n)):
        raise ValueError("Each candidate must occur exactly once; no missing/extra/duplicate IDs")
    return answer["groups"]


def materialize(groups: list[dict], candidates: list) -> list[dict]:
    """Keep the original representative verbatim; never union qualifiers."""
    merged = []
    for group in groups:
        members = [candidates[i] for i in group["members"]]
        support = sorted({(c.source["model"], c.source["prompt_hash"], c.source["gen_index"])
                          for c in members})
        merged.append({
            "fact": candidates[group["representative"]].fact,
            "n_support": len(support),
            "support": [{"model": m, "prompt_hash": p, "gen_index": g} for m, p, g in support],
            "sources": [dict(c.source, candidate_id=i) for i, c in zip(group["members"], members)],
            "candidate_ids": group["members"],
            "representative_id": group["representative"],
        })
    return merged


def partition_disagreements(groups: list[dict], candidates: list) -> dict:
    """Pair-level diagnostic, NOT accuracy: heuristic is not a reference."""
    labels = {i: g for g, group in enumerate(groups) for i in group["members"]}
    counts = defaultdict(int)
    examples = []
    for i in range(len(candidates)):
        for j in range(i):
            llm_same = labels[i] == labels[j]
            heuristic_same = same_claim(candidates[i].fact, candidates[j].fact)[0]
            counts[f"llm_{int(llm_same)}_heuristic_{int(heuristic_same)}"] += 1
            if llm_same != heuristic_same and len(examples) < 20:
                examples.append({"a_id": j, "b_id": i, "llm_same": llm_same,
                                 "heuristic_same": heuristic_same,
                                 "a": candidates[j].fact, "b": candidates[i].fact})
    return {"pair_counts": dict(counts), "examples": examples,
            "note": "Pairwise heuristic comparison only; not quality or cluster accuracy"}


def exact_selection(path: str, available: set[str], inputs: set[str]) -> list[str]:
    """A locked input frame must not be filtered by extraction success."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = data.get("input_ids") if isinstance(data, dict) else None
    if (not isinstance(ids, list) or not ids
            or any(not isinstance(i, str) or not i for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Selection must contain unique nonempty input_ids")
    if set(ids) - available or set(ids) - inputs:
        raise ValueError("Selection has missing source records or source paragraphs")
    return sorted(ids)


async def run(args: argparse.Namespace) -> dict:
    inputs = {r["input_id"]: r for r in load_inputs(args.inputs)}
    records = [r for rd in args.run_dirs for r in load_run_records(Path(rd))]
    if any(r.get("mock") for r in records):
        raise ValueError("Mock extraction records are not experimental candidates")
    by_input = defaultdict(list)
    for c in load_candidates(records):
        by_input[c.source["input_id"]].append(c)
    ids = sorted(by_input)
    selection = getattr(args, "selection", None)
    if selection:
        if args.selection_run_dir or args.limit is not None:
            raise ValueError("Exact --selection cannot be combined with --limit or --selection-run-dir")
        available = {r["task"]["input_id"] for r in records}
        ids = exact_selection(selection, available, set(inputs))
    elif args.selection_run_dir:
        selected_ids = {r["task"]["input_id"] for r in load_run_records(Path(args.selection_run_dir))}
        if not selected_ids or selected_ids - set(ids):
            raise ValueError("Selection run is empty or has inputs missing from candidate runs")
        ids = sorted(selected_ids)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        # Selection depends only on IDs and seed; no output quality filtering.
        random.Random(args.seed).shuffle(ids)
        ids = sorted(ids[:args.limit])
    if not ids or any(i not in inputs for i in ids):
        raise ValueError("No candidates or missing source paragraph")
    cfg = AccessConfig.load(args.access, creds_file=args.creds_file,
                            openrouter=getattr(args, "openrouter", False))
    if args.model:
        cfg.model = args.model
    if cfg.missing():
        raise ValueError("Missing API settings: " + "; ".join(cfg.missing()))
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    params = {"temperature": 0, "max_tokens": args.max_tokens}
    if args.max_tokens < 1:
        raise ValueError("max-tokens must be positive")
    out_dir = Path(args.out_dir)
    result = {"algorithm": "llm_equivalence", "algorithm_version": VERSION,
              "params": {"model": cfg.model, "gen_params": params,
                         "route": cfg.routing_metadata,
                         "prompt_hash": hash_obj(prompt), "seed": args.seed,
                         "support_semantics": "equivalent complete assertions, not core-only support"},
              "selected_input_ids": ids, "results": {}, "failures": {}, "diagnostics": {},
              "decisions": [], "quality_status": "preliminary"}
    write_json(out_dir / "selection.json", {"seed": args.seed, "input_ids": ids})

    async with OpenAICompatClient(cfg) as client:
        permanent_error = False
        async def one(iid: str) -> None:
            nonlocal permanent_error
            candidates = by_input[iid]
            if not candidates:
                result["failures"][iid] = "No usable extraction candidates; no merger request made"
                return
            payload = {"title": inputs[iid]["title"], "paragraph_text": inputs[iid]["paragraph_text"],
                       "candidates": [{"id": i, "fact": c.fact} for i, c in enumerate(candidates)]}
            cache_spec = {"version": VERSION, "model": cfg.model, "prompt": prompt,
                            "params": params, "payload": payload,
                            "sources": [c.source for c in candidates]}
            if cfg.cache_route is not None:
                cache_spec["route"] = cfg.cache_route
            key = hash_obj(cache_spec)
            cache = out_dir / "responses" / f"{key}.json"
            record = json.loads(cache.read_text()) if cache.exists() else None
            if record is None or (args.retry_failures and record.get("status") != "ok"):
                response = await client.chat(cfg.model, [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ], params, 0, key)
                record = {"input_id": iid, "cache_key": key, "status": "raw",
                          "route": cfg.routing_metadata,
                          "response": asdict(response), "payload": payload}
                write_json(cache, record)  # raw persisted BEFORE parse/validation
            if record.get("status") != "ok":
                try:
                    response = record["response"]
                    if not response["ok"]:
                        raise ValueError("API call failed: " + str(response.get("error")))
                    choices = (response.get("raw_response") or {}).get("choices") or []
                    if choices and choices[0].get("finish_reason") == "length":
                        raise ValueError("Truncated response")
                    answer = json.loads(response["text"] or "")
                    record["groups"] = validate_partition(answer, len(candidates))
                    record["status"] = "ok"
                except (ValueError, KeyError, TypeError) as exc:
                    record["status"] = "failed"
                    record["error"] = cfg.redact(str(exc))
                write_json(cache, record)
            if record["status"] == "ok":
                groups = validate_partition({"groups": record["groups"]}, len(candidates))
                result["results"][iid] = materialize(groups, candidates)
                for fact in result["results"][iid]:
                    fact["dedup_cache_key"] = key
                result["diagnostics"][iid] = partition_disagreements(groups, candidates)
            else:
                result["failures"][iid] = record.get("error", "failed")
                if record.get("response", {}).get("error_kind") == "permanent":
                    permanent_error = True

        for start in range(0, len(ids), max(1, cfg.max_concurrency)):
            await asyncio.gather(*(one(i) for i in ids[start:start + max(1, cfg.max_concurrency)]))
            write_json(out_dir / "merged_union.json", result)
            if permanent_error:
                result["halted_on_permanent_error"] = True
                result["pending_input_ids"] = ids[start + max(1, cfg.max_concurrency):]
                write_json(out_dir / "merged_union.json", result)
                break
    vote = dict(result, results={i: [f for f in facts if f["n_support"] >= 2]
                               for i, facts in result["results"].items()})
    vote["algorithm"] = "llm_equivalence_vote2"
    write_json(out_dir / "merged_vote2.json", vote)
    return {"selected": len(ids), "ok": len(result["results"]), "failed": len(result["failures"])}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dirs", nargs="+", required=True)
    p.add_argument("--inputs", default="data/tuning/inputs.jsonl")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prompt", default="prompts/merger/llm_dedup_v1.txt")
    p.add_argument("--access", default=None)
    p.add_argument("--creds-file", default=None)
    p.add_argument("--openrouter", action="store_true", help="explicit OpenRouter route; no automatic fallback")
    p.add_argument("--model", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--selection", default=None, help="exact selection.json input_ids; no success filtering or subsampling")
    p.add_argument("--selection-run-dir", default=None, help="reuse exact input IDs from e.g. C2 run")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=6000)
    p.add_argument("--retry-failures", action="store_true")
    stats = asyncio.run(run(p.parse_args()))
    print(json.dumps(stats, indent=2))
    return int(stats["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
