"""Aggregation of independent extractor answers.

Algorithms (each versioned; every output fact carries full provenance):
- union_dedup   : union of candidates with semantic duplicate removal
- vote          : keep clusters whose support (#independent generations) >= threshold
- support_check : (optional, separate LLM stage) drop/flag candidates without text support

Semantic dedup rules: normalized subject/relation/object equality or near-equality
is a *candidate* for merging; qualifiers decide. Compatible extra qualifiers are
unioned only when the underlying claim is the same; contradictory qualifier values
(e.g. two different dates for the same relation) keep the facts separate — different
episodes are never fused into invented ranges. Negation/modality/attribution
qualifiers are part of identity: a negated claim never merges with an affirmed one.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .hashing import merger_cache_key

ALGORITHM_VERSION = "agg-1.0"

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def norm(s: str) -> str:
    return _WS.sub(" ", _PUNCT.sub("", s.strip().lower())).strip()


def similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


@dataclass
class Candidate:
    fact: dict[str, Any]
    source: dict[str, Any]  # run_id, model, prompt_hash, gen_index, input_id, cache_key

    @property
    def core(self) -> tuple[str, str, str]:
        return (self.fact["subject"], self.fact["relation"], self.fact["object"])


def load_candidates(records: list[dict[str, Any]]) -> list[Candidate]:
    cands: list[Candidate] = []
    for rec in records:
        if rec.get("status") != "ok" or not rec.get("parsed"):
            continue
        src = {
            "cache_key": rec.get("cache_key"),
            "input_id": rec["task"]["input_id"],
            "model": rec["task"]["model"],
            "prompt_hash": rec["task"]["prompt_hash"],
            "gen_index": rec["task"]["gen_index"],
        }
        if "route" in rec["task"]:
            src["route"] = rec["task"]["route"]
        for fact in rec["parsed"]["triplets"]:
            cands.append(Candidate(fact=fact, source=src))
    return cands


def load_run_records(run_dir: Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted((run_dir / "responses").glob("*.json")):
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


_IDENTITY_QUALIFIERS = {"polarity", "modality", "attribution", "condition", "scope"}


def _qual_map(fact: dict[str, Any]) -> dict[str, set[str]]:
    m: dict[str, set[str]] = defaultdict(set)
    for q in fact.get("qualifiers") or []:
        m[norm(q["relation"])].add(norm(q["object"]))
    return {k: v for k, v in m.items()}


def _compatible(qa: dict[str, set[str]], qb: dict[str, set[str]]) -> tuple[bool, list[str]]:
    """Two facts may be the same claim iff no shared qualifier relation carries
    contradictory values and identity qualifiers (negation/modality/attribution/
    condition/scope) are equal where present."""
    conflicts = []
    for rel in set(qa) | set(qb):
        if rel in _IDENTITY_QUALIFIERS:
            if qa.get(rel, set()) != qb.get(rel, set()):
                conflicts.append(rel)
        elif rel in qa and rel in qb and not (qa[rel] & qb[rel]):
            # disjoint values for the same relation: different episodes (e.g. dates)
            conflicts.append(rel)
    return (not conflicts), conflicts


def same_claim(a: dict[str, Any], b: dict[str, Any], threshold: float = 0.9) -> tuple[bool, list[str]]:
    ca = (norm(a["subject"]), norm(a["relation"]), norm(a["object"]))
    cb = (norm(b["subject"]), norm(b["relation"]), norm(b["object"]))
    if ca == cb:
        core_ok = True
    else:
        core_ok = all(similarity(x, y) >= threshold for x, y in zip(ca, cb))
    if not core_ok:
        return False, ["core"]
    ok, conflicts = _compatible(_qual_map(a), _qual_map(b))
    return ok, conflicts


def cluster_same_claim(cands: list[Candidate], threshold: float = 0.9) -> list[list[Candidate]]:
    """Fast semantic clustering of candidates (same semantics as repeated same_claim
    against the cluster representative).

    Two passes: exact normalized-core buckets first (O(n)), then fuzzy comparison
    against cluster representatives only. Never call this across different inputs —
    cluster per input_id first."""
    items = []
    for c in cands:
        f = c.fact
        core = (norm(f["subject"]), norm(f["relation"]), norm(f["object"]))
        items.append((c, core, _qual_map(f)))
    clusters: list[list[Any]] = []  # [rep_item, members]
    exact: dict[tuple[str, str, str], list[int]] = {}
    for item in items:
        c, core, quals = item
        placed = False
        for idx in exact.get(core, []):
            if _compatible(quals, clusters[idx][0][2])[0]:
                clusters[idx][1].append(c)
                placed = True
                break
        if placed:
            continue
        for rep, members in clusters:
            rcore = rep[1]
            if rcore != core and not all(similarity(x, y) >= threshold for x, y in zip(core, rcore)):
                continue
            if _compatible(quals, rep[2])[0]:
                members.append(c)
                placed = True
                break
        if not placed:
            clusters.append([item, [c]])
            exact.setdefault(core, []).append(len(clusters) - 1)
    return [members for _, members in clusters]


def merge_group(members: list[Candidate]) -> dict[str, Any]:
    """Merge same-claim candidates: union of qualifiers (normalized), richest core
    kept; full provenance preserved."""
    base = dict(members[0].fact)
    seen_q: dict[tuple[str, str], dict[str, Any]] = {}
    for m in members:
        for q in m.fact.get("qualifiers") or []:
            key = (norm(q["relation"]), norm(q["object"]))
            seen_q[key] = {"relation": q["relation"], "object": q["object"]}
    base["qualifiers"] = list(seen_q.values())
    support = sorted({(m.source["model"], m.source["prompt_hash"], m.source["gen_index"]) for m in members})
    return {
        "fact": base,
        "n_support": len(support),
        "support": [{"model": m, "prompt_hash": p, "gen_index": g} for m, p, g in support],
        "sources": [m.source for m in members],
    }


def aggregate(candidates: list[Candidate], algorithm: str = "union_dedup",
              vote_threshold: int = 1, similarity_threshold: float = 0.9,
              per_input: bool = True) -> dict[str, Any]:
    """Return merged facts grouped by input_id with provenance and decision log."""
    by_input: dict[str, list[Candidate]] = defaultdict(list)
    for c in candidates:
        by_input[c.source["input_id"]].append(c)
    if not per_input:
        by_input = {"__all__": candidates}

    merged_all: dict[str, Any] = {}
    decisions: list[dict[str, Any]] = []
    for iid, cands in by_input.items():
        clusters: list[list[Candidate]] = []
        for c in cands:
            placed = False
            for cl in clusters:
                ok, conflicts = same_claim(cl[0].fact, c.fact, similarity_threshold)
                if ok:
                    cl.append(c)
                    placed = True
                    break
                if conflicts and conflicts != ["core"]:
                    decisions.append({
                        "input_id": iid, "action": "kept_separate",
                        "reason": f"conflicting qualifiers: {conflicts}",
                        "a": cl[0].fact, "b": c.fact,
                    })
            if not placed:
                clusters.append([c])
        merged = [merge_group(cl) for cl in clusters]
        if algorithm == "vote":
            kept = [m for m in merged if m["n_support"] >= vote_threshold]
            for m in merged:
                if m["n_support"] < vote_threshold:
                    decisions.append({
                        "input_id": iid, "action": "dropped_by_vote",
                        "reason": f"n_support={m['n_support']} < {vote_threshold}",
                        "fact": m["fact"],
                    })
            merged = kept
        elif algorithm != "union_dedup":
            raise ValueError(f"unknown algorithm: {algorithm}")
        merged_all[iid] = merged
    return {
        "algorithm": algorithm,
        "algorithm_version": ALGORITHM_VERSION,
        "params": {"vote_threshold": vote_threshold, "similarity_threshold": similarity_threshold,
                   "per_input": per_input},
        "results": merged_all,
        "decisions": decisions,
    }


def agg_cache_inputs(records: list[dict[str, Any]]) -> list[str]:
    import hashlib
    return [hashlib.sha256((r.get("cache_key") or json.dumps(r.get("task"), sort_keys=True)).encode()).hexdigest()
            for r in records]


def save_aggregate(agg: dict[str, Any], records: list[dict[str, Any]], out_path: Path) -> str:
    key = merger_cache_key(agg_cache_inputs(records), agg["algorithm"], agg["algorithm_version"], agg["params"])
    out = dict(agg)
    out["merger_cache_key"] = key
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_path)
    return key
