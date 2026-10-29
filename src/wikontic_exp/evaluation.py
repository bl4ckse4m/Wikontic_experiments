"""Evaluation: gold-based metrics and proxy metrics.

MuSiQue gives QA annotation, not a full gold of all facts. Until independent
annotation of the pre-selected paragraphs exists, all automatic numbers here are
**proxy** metrics and method choice is preliminary. Facts are matched one-to-one
semantically (normalized equality or similarity), so duplicated predictions can
never inflate recall of a single gold fact.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .aggregation import norm, similarity

_TOKEN = re.compile(r"[\w]+", re.UNICODE)


def tokens(s: str) -> set[str]:
    return set(_TOKEN.findall(s.lower()))


def text_support_proxy(fact: dict[str, Any], text: str) -> dict[str, Any]:
    """Proxy only: does the source text lexically contain the entity strings?"""
    tnorm = norm(text)
    def present(s: str) -> bool:
        n = norm(s)
        return bool(n) and (n in tnorm or (len(tokens(s)) > 0 and tokens(s) <= tokens(text)))
    s_ok, o_ok = present(fact["subject"]), present(fact["object"])
    quals = fact.get("qualifiers") or []
    q_ok = [present(q["object"]) for q in quals] if quals else []
    return {
        "subject_in_text": s_ok,
        "object_in_text": o_ok,
        "qualifiers_in_text": q_ok,
        "supported_lexically": s_ok and o_ok and all(q_ok),
    }


def match_facts(preds: list[dict[str, Any]], golds: list[dict[str, Any]],
                threshold: float = 0.9, full: bool = False) -> dict[str, Any]:
    """One-to-one greedy matching by similarity; ``full`` requires qualifiers to agree."""
    def score(p: dict[str, Any], g: dict[str, Any]) -> float:
        core = sum(similarity(norm(p[k]), norm(g[k])) for k in ("subject", "relation", "object")) / 3
        if not full:
            return core
        pq = {(norm(q["relation"]), norm(q["object"])) for q in p.get("qualifiers") or []}
        gq = {(norm(q["relation"]), norm(q["object"])) for q in g.get("qualifiers") or []}
        if not gq and not pq:
            q = 1.0
        elif not gq or not pq:
            q = 0.5
        else:
            inter = len(pq & gq)
            q = inter / max(len(pq), len(gq))
        return 0.7 * core + 0.3 * q

    pairs = sorted(
        ((score(p, g), pi, gi) for pi, p in enumerate(preds) for gi, g in enumerate(golds)),
        key=lambda x: -x[0],
    )
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches = []
    for s, pi, gi in pairs:
        if s < threshold or pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append({"pred_idx": pi, "gold_idx": gi, "score": s})
    return {
        "matches": matches,
        "matched_pred": sorted(used_p),
        "matched_gold": sorted(used_g),
        "n_pred": len(preds),
        "n_gold": len(golds),
    }


def prf(matched: int, n_pred: int, n_gold: int) -> dict[str, float]:
    p = matched / n_pred if n_pred else 0.0
    r = matched / n_gold if n_gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "matched": matched, "n_pred": n_pred, "n_gold": n_gold}


def qualifier_metrics(preds: list[dict[str, Any]], golds: list[dict[str, Any]],
                      core_matches: dict[str, Any]) -> dict[str, Any]:
    """Qualifier P/R (a) end-to-end over all facts, (b) given a matched core fact."""
    e2e = match_facts(preds, golds, threshold=0.9, full=True)
    given_core_tp = given_core_pred = given_core_gold = 0
    for m in core_matches["matches"]:
        p = preds[m["pred_idx"]]
        g = golds[m["gold_idx"]]
        pq = {(norm(q["relation"]), norm(q["object"])) for q in p.get("qualifiers") or []}
        gq = {(norm(q["relation"]), norm(q["object"])) for q in g.get("qualifiers") or []}
        given_core_tp += len(pq & gq)
        given_core_pred += len(pq)
        given_core_gold += len(gq)
    out = {
        "end_to_end_full_statement": prf(len(e2e["matches"]), e2e["n_pred"], e2e["n_gold"]),
        "given_core": prf(given_core_tp, given_core_pred, given_core_gold),
    }
    return out


def gold_eval(preds: list[dict[str, Any]], golds: list[dict[str, Any]],
              threshold: float = 0.9) -> dict[str, Any]:
    core = match_facts(preds, golds, threshold=threshold, full=False)
    full = match_facts(preds, golds, threshold=threshold, full=True)
    quals = qualifier_metrics(preds, golds, core)
    return {
        "core": prf(len(core["matches"]), core["n_pred"], core["n_gold"]),
        "full_statements": prf(len(full["matches"]), full["n_pred"], full["n_gold"]),
        "qualifiers": quals,
        "fabricated_or_distorted_proxy": 1.0 - (len(core["matches"]) / core["n_pred"] if core["n_pred"] else 0.0),
    }


def compactness(facts: list[dict[str, Any]], correct_ids: set[int] | None = None) -> dict[str, Any]:
    """Size of the *whole* serialization (core + qualifiers), tokens and unique
    correct semantic units per representation volume."""
    ser = json.dumps({"triplets": facts}, ensure_ascii=False)
    n_chars = len(ser)
    n_tokens_est = max(1, n_chars // 4)
    n_units = len(facts)
    n_correct = len(correct_ids) if correct_ids is not None else None
    out = {
        "n_facts": n_units,
        "serialization_chars": n_chars,
        "serialization_tokens_est": n_tokens_est,
        "facts_per_1k_chars": 1000.0 * n_units / n_chars if n_chars else 0.0,
    }
    if n_correct is not None:
        out["n_unique_correct_units"] = n_correct
        out["correct_units_per_1k_chars"] = 1000.0 * n_correct / n_chars if n_chars else 0.0
    return out


def proxy_report(records: list[dict[str, Any]], texts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Automatic (proxy) quality signals; NOT a substitute for gold annotation."""
    n = len(records)
    ok = [r for r in records if r.get("status") == "ok"]
    facts_by_input: dict[str, list[dict[str, Any]]] = {}
    for r in ok:
        iid = r["task"]["input_id"]
        facts_by_input.setdefault(iid, []).extend(r["parsed"]["triplets"])
    lex_supported = lex_total = 0
    empty_facts = 0
    for iid, facts in facts_by_input.items():
        text = texts.get(iid, {}).get("paragraph_text", "")
        for f in facts:
            lex_total += 1
            if text_support_proxy(f, text)["supported_lexically"]:
                lex_supported += 1
            if not (f.get("subject") and f.get("relation") and f.get("object")):
                empty_facts += 1
    status_counts: dict[str, int] = {}
    for r in records:
        status_counts[r.get("status", "?")] = status_counts.get(r.get("status", "?"), 0) + 1
    return {
        "note": "PROXY metrics only (no independent gold annotation yet)",
        "n_tasks": n,
        "status_counts": status_counts,
        "ok_rate": len(ok) / n if n else 0.0,
        "n_facts": lex_total,
        "lexical_support_rate": lex_supported / lex_total if lex_total else 0.0,
        "empty_field_facts": empty_facts,
        "unique_inputs_with_facts": len(facts_by_input),
    }


def load_gold(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["triplets"] if isinstance(data, dict) else data


def qa_em_f1(prediction: str, answers: list[str]) -> dict[str, float]:
    """Official-style QA EM/F1 (token F1), for the qualifier ablation only."""
    pred_toks = tokens(prediction)
    em = float(any(norm(prediction) == norm(a) for a in answers))
    best_f1 = 0.0
    for a in answers:
        at = tokens(a)
        if not pred_toks or not at:
            continue
        inter = len(pred_toks & at)
        if inter == 0:
            continue
        p = inter / len(pred_toks)
        r = inter / len(at)
        best_f1 = max(best_f1, 2 * p * r / (p + r))
    return {"em": em, "f1": best_f1}
