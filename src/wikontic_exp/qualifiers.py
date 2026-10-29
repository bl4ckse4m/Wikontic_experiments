"""Qualifier ablation representations (Q0/Q1/Q2) built from ONE frozen fact set.

Q0: core triples only (qualifiers dropped)
Q1: same facts, hyper-relational (core + qualifiers)
Q2: same information reified as event/claim nodes with binary edges

Q1-vs-Q0 measures information loss from dropping qualifiers; Q1-vs-Q2 compares
representations of identical information. All three are *derived* from the same
frozen facts — no separate extraction is involved. Graph-only ablations must not
include source text or evidence quotes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_CONDITION_QUALIFIERS = {
    "point in time", "start time", "end time", "point in time", "inception",
    "polarity", "modality", "attribution", "condition", "approximation", "unit",
    "quantity", "place", "location", "scope",
}

_TIME = re.compile(r"\b(\d{4}|\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{4}|century|decade)\b", re.I)


def freeze_facts(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "facts" in data:
        return data
    if "results" in data:  # aggregation output: flatten merged groups into a frozen fact list
        facts = [g["fact"] for groups in data["results"].values() for g in groups]
        return {"facts": facts, "frozen_from": str(path), "merger_cache_key": data.get("merger_cache_key")}
    if "triplets" in data:
        return {"facts": data["triplets"], "frozen_from": str(path)}
    raise ValueError(f"unrecognized frozen fact file: {path}")


def to_q0(frozen: dict[str, Any]) -> dict[str, Any]:
    return {
        "representation": "Q0_core_triples",
        "items": [
            {"subject": f["subject"], "relation": f["relation"], "object": f["object"], "qualifiers": []}
            for f in frozen["facts"]
        ],
    }


def to_q1(frozen: dict[str, Any]) -> dict[str, Any]:
    return {
        "representation": "Q1_hyper_relational",
        "items": [
            {k: f[k] for k in ("subject", "relation", "object", "qualifiers") if k in f}
            for f in frozen["facts"]
        ],
    }


def to_q2(frozen: dict[str, Any]) -> dict[str, Any]:
    """Reification: one event/claim node per fact, binary edges for core and qualifiers."""
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for i, f in enumerate(frozen["facts"]):
        node = f"claim_{i}"
        nodes.append({"id": node, "type": "claim", "label": f"{f['subject']} | {f['relation']} | {f['object']}"})
        edges.append({"source": f["subject"], "relation": f["relation"], "target": f["object"], "via": node})
        for q in f.get("qualifiers") or []:
            edges.append({"source": node, "relation": q["relation"], "target": q["object"], "via": node})
    return {"representation": "Q2_reified_binary", "nodes": nodes, "edges": edges}


def information_preservation(frozen: dict[str, Any]) -> dict[str, Any]:
    """Which condition classes survive in each representation (deterministic audit)."""
    counts = {"time": 0, "polarity_modality_attribution": 0, "quantity_unit": 0, "place": 0, "other": 0}
    q0_kept = dict.fromkeys(counts, 0)
    q1_kept = dict.fromkeys(counts, 0)
    q2_kept = dict.fromkeys(counts, 0)

    def classify(rel: str, obj: str) -> str:
        r = rel.lower()
        if r in ("point in time", "start time", "end time", "inception", "date of birth", "date of death") or _TIME.search(obj):
            return "time"
        if r in ("polarity", "modality", "attribution", "condition", "scope"):
            return "polarity_modality_attribution"
        if r in ("unit", "quantity", "approximation") or re.search(r"\d", obj):
            return "quantity_unit"
        if r in ("place", "location", "country", "country of citizenship"):
            return "place"
        return "other"

    for f in frozen["facts"]:
        for q in f.get("qualifiers") or []:
            cls = classify(q["relation"], q["object"])
            counts[cls] += 1
            q1_kept[cls] += 1
            q2_kept[cls] += 1
            # Q0 drops all qualifiers by construction
        # values embedded in core objects survive everywhere
        for obj in (f.get("object"),):
            cls = classify(f["relation"], obj or "")
            if cls in ("time", "quantity_unit"):
                q0_kept[cls] += 1
                q1_kept[cls] += 1
                q2_kept[cls] += 1
    return {
        "note": "counts of condition-bearing values per class; Q0 keeps only values inside core objects",
        "total_in_qualifiers": counts,
        "q0_kept": q0_kept,
        "q1_kept": q1_kept,
        "q2_kept": q2_kept,
    }


def build_ablation_inputs(frozen_path: Path, out_dir: Path) -> dict[str, Any]:
    frozen = freeze_facts(frozen_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    reps = {"Q0": to_q0(frozen), "Q1": to_q1(frozen), "Q2": to_q2(frozen)}
    for name, rep in reps.items():
        (out_dir / f"{name}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = information_preservation(frozen)
    (out_dir / "information_preservation.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"representations": {k: str(out_dir / f"{k}.json") for k in reps}, "audit": audit}
