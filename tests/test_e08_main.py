"""Stage adapter tests; synthetic files only, no API/credential access."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from wikontic_exp.aggregation import load_candidates
from wikontic_exp.hashing import hash_obj
from wikontic_exp.llm_dedup import materialize

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("e08_stage_adapter", REPO / "scripts/e08_main.py")
e08 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e08)


def save(root, rel, value):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(e08.e08_preparation.json_bytes(value))


def save_rows(root, rel, rows):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(e08.e08_preparation.jsonl_bytes(rows))


@pytest.fixture
def calibration(tmp_path, monkeypatch):
    monkeypatch.setattr(e08.e08_preparation, "verify", lambda root: {"status": "verified"})
    for rel in (f"{e08.BASE}/calibration_decision.json", "scripts/e08_main.py"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((REPO / rel).read_bytes())
    save(tmp_path, f"{e08.BASE}/lock.json", {"synthetic": True})
    for kind in ("binary", "merger_audit"):
        binary = kind == "binary"
        key = "annotation_id" if binary else "audit_id"
        filename = "candidates.jsonl" if binary else "pairs.jsonl"
        save_rows(tmp_path, f"{e08.CAL}/{kind}/{filename}", [{key: f"id{i}"} for i in range(24)])
        task = tmp_path / f"{e08.CAL}/{kind}/TASK.md"
        task.parent.mkdir(parents=True, exist_ok=True)
        task.write_bytes((REPO / f"{e08.CAL}/{kind}/TASK.md").read_bytes())
        for family, filename in (("haiku", "judgments.jsonl"), ("sonnet", "judgments_sonnet.jsonl")):
            labels = [{key: f"id{i}", "annotator": f"Claude {family.title()} 5.5",
                       "status": "silver", "rationale": "synthetic",
                       **({"core_label": 1, "full_label": int(i > 0 or family == "sonnet"), "evidence": "text"}
                          if binary else {"equivalent": int(i > 0 or family == "sonnet")})}
                      for i in range(24)]
            save_rows(tmp_path, f"{e08.CAL}/{kind}/{filename}", labels)
    return tmp_path


def test_freeze_preserves_rubric_and_refuses_silent_changes(calibration):
    root = calibration
    assert e08.freeze(root)["additional_calibration_runs"] == 0
    assert e08.verify(root)["status"] == "verified"
    assert e08.freeze(root)["status"].startswith("already frozen")
    summary = e08.read(root, f"{e08.BASE}/calibration_summary.json")
    assert summary["agreement"]["full_label"]["agree"] == 23
    assert summary["agreement"]["equivalent"]["agree"] == 23
    for kind in ("binary", "merger_audit"):
        main = e08.main_task(root, kind).decode()
        original = (root / f"{e08.CAL}/{kind}/TASK.md").read_text()
        marker = "# Binary assessment" if kind == "binary" else "# Blind equivalence audit"
        rubric = original[original.index(marker):].replace(f"{e08.CAL}/{kind}/", f"{e08.MAIN}/{kind}/")
        assert main.endswith(rubric)
        assert "not the main evaluation" not in main
    file = root / f"{e08.CAL}/binary/judgments_sonnet.jsonl"
    file.write_bytes(file.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Stage lock mismatch"):
        e08.verify(root)


def test_freeze_rejects_incomplete_calibration_without_new_files(calibration):
    root = calibration
    rel = f"{e08.CAL}/binary/judgments_sonnet.jsonl"
    labels = e08.rows(root, rel)
    save_rows(root, rel, labels[:-1])
    with pytest.raises(ValueError, match="Incomplete calibration"):
        e08.freeze(root)
    assert not (root / f"{e08.BASE}/rubric_lock.json").exists()


@pytest.fixture
def merger(tmp_path):
    ids = [f"i{i:02d}" for i in range(50)]
    inputs = [{"input_id": iid, "title": iid, "paragraph_text": "synthetic text"} for iid in ids]
    save(tmp_path, f"{e08.BASE}/selection.json", {"input_ids": ids})
    save_rows(tmp_path, f"{e08.BASE}/inputs.jsonl", inputs)
    source_tasks, originals = {}, {iid: [] for iid in ids}
    for m in ("coverage", "baseline_fixed", "fewshot", "autoontik"):
        source_tasks[m] = []
        for iid in ids:
            key = hash_obj([m, iid])
            fact = {"subject": iid, "relation": "r", "object": m, "qualifiers": []}
            rec = {"status": "ok", "cache_key": key,
                   "task": {"input_id": iid, "model": "synthetic", "prompt_hash": m, "gen_index": 0},
                   "parsed": {"triplets": [fact]}}
            save(tmp_path, f"{e08.RUNS[m]}/responses/{key}.json", rec)
            source_tasks[m].append({"input_id": iid, "cache_key": key, "status": "ok", "n_facts": 1})
            if m != "autoontik":
                originals[iid].extend(load_candidates([rec]))
    save(tmp_path, f"{e08.BASE}/preflight.json", {"source_tasks": source_tasks})
    union = {"algorithm": "llm_equivalence", "algorithm_version": e08.llm_dedup.VERSION,
             "selected_input_ids": ids, "results": {}, "failures": {},
             "params": {"gen_params": {"temperature": 0, "max_tokens": 6000},
                        "model": "openai/gpt-oss-120b", "prompt_hash": e08.e08_preparation.V2_PROMPT_HASH,
                        "route": {"provider": "openrouter", "endpoint_hash": "synthetic"}, "seed": 42}}
    groups = [{"members": [0, 1], "representative": 0}, {"members": [2], "representative": 2}]
    for row in inputs:
        iid = row["input_id"]
        key = hash_obj(["merger", iid])
        rec = {"input_id": iid, "cache_key": key, "status": "ok", "groups": groups,
               "route": union["params"]["route"],
               "payload": {"title": iid, "paragraph_text": row["paragraph_text"],
                           "candidates": [{"id": i, "fact": c.fact} for i, c in enumerate(originals[iid])]}}
        save(tmp_path, f"{e08.MERGER}/responses/{key}.json", rec)
        fs = materialize(groups, originals[iid])
        for f in fs:
            f["dedup_cache_key"] = key
        union["results"][iid] = fs
    vote = copy.deepcopy(union)
    vote["algorithm"] = "llm_equivalence_vote2"
    vote["results"] = {i: [f for f in fs if f["n_support"] >= 2] for i, fs in union["results"].items()}
    save(tmp_path, e08.METHODS["llm_union"], union)
    save(tmp_path, e08.METHODS["llm_vote2"], vote)
    heuristic_union = copy.deepcopy(union)
    heuristic_union.update(algorithm="union_dedup", algorithm_version="agg-1.0")
    heuristic_vote = copy.deepcopy(vote)
    heuristic_vote.update(algorithm="vote", algorithm_version="agg-1.0")
    save(tmp_path, e08.METHODS["heuristic_union"], heuristic_union)
    save(tmp_path, e08.METHODS["heuristic_vote2"], heuristic_vote)
    return tmp_path


def test_validate_merger_and_sample_opaque_blind_pairs(merger):
    root = merger
    union, _, records = e08.merger_data(root)
    public, private = e08.pair_files(union, records)
    assert len(public) == 150  # all 50 proposed merges; 100 proposed separations
    assert private["population_pairs"] == {"True": 50, "False": 100}
    assert len({r["audit_id"] for r in public}) == 150
    assert all(len(r["audit_id"]) == 64 for r in public)
    assert all(set(r) == {"audit_id", "title", "paragraph_text", "a", "b"} for r in public)
    assert e08.pair_files(union, records) == (public, private)
    # No synthesized or edited representative may enter the main evaluation.
    union["results"]["i00"][0]["fact"]["object"] = "repaired"
    save(root, e08.METHODS["llm_union"], union)
    vote = e08.read(root, e08.METHODS["llm_vote2"])
    vote["results"]["i00"][0]["fact"]["object"] = "repaired"
    save(root, e08.METHODS["llm_vote2"], vote)
    with pytest.raises(ValueError, match="original representatives"):
        e08.merger_data(root)


def test_validate_merger_rejects_parameter_drift(merger):
    data = e08.read(merger, e08.METHODS["llm_union"])
    data["params"]["gen_params"]["max_tokens"] = 12000
    save(merger, e08.METHODS["llm_union"], data)
    with pytest.raises(ValueError, match="configuration/frame"):
        e08.merger_data(merger)


def test_prepare_main_judging_end_to_end(calibration, merger):
    root = calibration
    assert root == merger
    e08.freeze(root)
    result = e08.prepare_judging(root)
    assert result["model_calls"] == 0 and result["pairs_to_judge"] == 150
    assert not result["zero_population_methods"]
    availability = e08.read(root, f"{e08.BASE}/availability.json")
    assert set(availability) == set(e08.METHODS)
    assert all(a["n_evaluation_inputs"] == 50 for a in availability.values())
    verification = e08.verify(root)
    assert verification["prediction_lock_present"]
    assert verification["judgments"]["binary"]["present"] == 0
    assert verification["judgments"]["binary"]["expected"] == result["unique_facts_to_judge"]
    assert e08.prepare_judging(root)["status"].startswith("already prepared")
    candidates = e08.rows(root, f"{e08.MAIN}/binary/candidates.jsonl")
    assert all(set(c) == {"annotation_id", "input_id", "title", "paragraph_text", "fact"}
               for c in candidates)
    wrong = {"annotation_id": candidates[0]["annotation_id"], "core_label": 1, "full_label": 1,
             "evidence": "synthetic", "rationale": "synthetic", "status": "silver",
             "annotator": "Claude Haiku 5.5"}
    save_rows(root, f"{e08.MAIN}/binary/judgments.jsonl", [wrong])
    with pytest.raises(ValueError, match="wrong judge model"):
        e08.verify(root)


def test_empty_audit_is_an_explicit_empty_population():
    public, private = e08.pair_files({"results": {}}, {})
    assert public == [] and private["population_pairs"] == {"True": 0, "False": 0}


def test_main_labels_allow_missing_but_not_wrong_judge_or_duplicate_ids():
    candidates = [{"annotation_id": "a"}, {"annotation_id": "b"}]
    label = {"annotation_id": "a", "core_label": 1, "full_label": 0, "evidence": "text",
             "rationale": "synthetic", "annotator": "Claude Sonnet 5.5", "status": "silver"}
    assert len(e08.checked_labels(candidates, [label], pair=False, family="sonnet", complete=False)) == 1
    for invalid in ([label, label], [dict(label, annotator="Claude Haiku 5.5")],
                    [dict(label, full_label=True)]):
        with pytest.raises(ValueError):
            e08.checked_labels(candidates, invalid, pair=False, family="sonnet", complete=False)


def test_create_files_preserves_existing_user_work(tmp_path):
    save(tmp_path, "existing.json", {"keep": True})
    with pytest.raises(ValueError, match="not overwritten"):
        e08.create_files(tmp_path, {"new.json": b"new", "existing.json": b"replacement"})
    assert not (tmp_path / "new.json").exists()
    assert e08.read(tmp_path, "existing.json") == {"keep": True}
