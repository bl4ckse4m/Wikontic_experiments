"""E08 offline preparation: synthetic records only, no API or real credentials."""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import pytest

from wikontic_exp import e08_preparation as e08
from wikontic_exp.aggregation import Candidate, aggregate
from wikontic_exp.hashing import hash_obj, input_id, sha256_hex


def write_json(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(e08.json_bytes(value))


def write_jsonl(root, relative, rows):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(e08.jsonl_bytes(rows))


@pytest.fixture
def project(tmp_path, monkeypatch):
    rows = [{"title": f"Synthetic title {i}", "paragraph_text": f"Synthetic text {i}",
             "refs": [{"sample_id": f"qa{i}", "paragraph_idx": 0, "is_supporting": True}],
             "question": "DO_NOT_EXPORT_QA"} for i in range(300)]
    for row in rows:
        row["input_id"] = input_id(row["title"], row["paragraph_text"])
    inputs = e08.input_map(rows)
    dev = rows[0]["input_id"]
    excluded = [r["input_id"] for r in rows[:20]]
    monkeypatch.setattr(e08, "FACT_CASES", tuple((dev, i) for i in range(24)))
    monkeypatch.setattr(e08, "PAIR_CASES", tuple((dev, i, i + 24) for i in range(24)))
    monkeypatch.setattr(e08, "CALIBRATION_TITLES", {dev: inputs[dev]["title"]})
    prompt = "synthetic merger prompt"
    monkeypatch.setattr(e08, "V2_PROMPT_HASH", hash_obj(prompt))
    (tmp_path / "prompts/merger").mkdir(parents=True)
    (tmp_path / e08.V2_PROMPT).write_text(prompt)
    write_jsonl(tmp_path, "data/tuning/inputs.jsonl", rows)
    write_jsonl(tmp_path, "data/tuning/subset300_inputs.jsonl", rows)
    write_jsonl(tmp_path, "data/tuning/pilot12_inputs.jsonl", rows[:12])
    write_json(tmp_path, "reports/tuning_comparison.json", {"gold_kinds": {excluded[1]: "silver"}})
    write_json(tmp_path, "data/gold/gold_selection.json", {"items": [{"input_id": excluded[2]}]})
    # This file must NOT be parsed: its filename alone denotes prior exposure.
    gold = tmp_path / "data/gold/pending" / f"{excluded[3]}.gold.json"
    gold.parent.mkdir(parents=True)
    gold.write_text("NOT JSON; do not inspect old labels")
    write_jsonl(tmp_path, "runs/gptoss_pilot_synthetic/index.jsonl", [{"input_id": excluded[4]}])
    for run in (e08.V1_RUN, e08.V2_RUN):
        write_json(tmp_path, f"{run}/selection.json", {"input_ids": excluded})
    selected = e08.select_ids(inputs, dict.fromkeys(excluded))
    records = {}

    def make_run(name, run, wanted, n_generations=1):
        text = f"synthetic prompt {name}"
        rel_prompt = f"prompts/synthetic/{name}.txt"
        path = tmp_path / rel_prompt
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        method = {"prompt_path": rel_prompt, "prompt_sha256": sha256_hex(text),
                  "n_generations": n_generations, "schema_id": "wikontic_facts_v1",
                  "gen_params": {"temperature": 0.2, "max_tokens": 6000}}
        write_json(tmp_path, f"{run}/run_config.json", {
            "mock": False, "method": method, "access_model": "synthetic-model"})
        for iid in sorted(wanted):
            for g in range(n_generations):
                row = inputs[iid]
                task = {"input_id": iid, "input_hash": hash_obj({k: row[k] for k in ("title", "paragraph_text")}),
                        "gen_index": g, "model": "synthetic-model", "prompt_hash": sha256_hex(text),
                        "schema_id": method["schema_id"], "gen_params": method["gen_params"]}
                key = hash_obj({"run": run, "input_id": iid, "g": g})
                facts = [{"subject": row["title"], "relation": f"r{n}",
                          "object": f"object {name} {n}", "qualifiers": []}
                         for n in range(48 if iid == dev and name == "coverage" else 1)]
                status = "ok"
                if name == "coverage" and iid == selected[0]:
                    status, facts = "truncated", None
                elif name == "baseline_fixed" and iid == selected[1]:
                    facts = []
                rec = {"task": task, "status": status, "mock": False, "cache_key": key,
                       "parsed": {"triplets": facts} if facts is not None else None}
                records[(name, iid, g)] = rec
                write_json(tmp_path, f"{run}/responses/{key}.json", rec)
    for name, run in e08.RUNS.items():
        make_run(name, run, set(selected) | ({dev} if name != "autoontik" else set()))
    make_run("c2", "runs/gptoss_ens3_300", inputs, 3)
    candidates = []
    for iid in selected:
        for name in ("coverage", "baseline_fixed", "fewshot"):
            rec = records[(name, iid, 0)]
            if rec["status"] == "ok":
                source = {k: rec["task"][k] for k in ("input_id", "model", "prompt_hash", "gen_index")}
                source["cache_key"] = rec["cache_key"]
                candidates.extend(Candidate(f, source) for f in rec["parsed"]["triplets"])
    write_json(tmp_path, "runs/ensemble_xprompt/merged_union.json", aggregate(candidates))
    write_json(tmp_path, "runs/ensemble_xprompt/merged_vote2.json",
               aggregate(candidates, algorithm="vote", vote_threshold=2))
    original = [f for name in ("coverage", "baseline_fixed", "fewshot")
                for f in records[(name, dev, 0)]["parsed"]["triplets"]]
    key = "synthetic-development-merger"
    write_json(tmp_path, f"{e08.V2_RUN}/responses/{key}.json", {
        "input_id": dev, "cache_key": key, "status": "ok",
        "payload": {"title": inputs[dev]["title"], "paragraph_text": inputs[dev]["paragraph_text"],
                    "candidates": [{"id": i, "fact": f} for i, f in enumerate(original)]}})
    write_json(tmp_path, f"{e08.V2_RUN}/merged_union.json", {
        "params": {"prompt_hash": hash_obj(prompt)},
        "results": {dev: [{"dedup_cache_key": key}]}})
    repo = Path(__file__).resolve().parents[1]
    for rel in ("data/judging/ANNOTATION_TASK.md", "data/judging/MERGER_AUDIT_TASK.md"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((repo / rel).read_bytes())
    for rel in ("reports/E08_PROTOCOL.md", "schemas/facts_v1.json", "pyproject.toml",
                "scripts/prepare_merger_audit.py", "scripts/score_merger_audit.py",
                "src/wikontic_exp/synthetic.py"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic file\n")
    return tmp_path, selected, records


def test_selection_uses_only_ids_and_exclusions():
    frame = {f"id{i:03d}": {"status": "failed"} for i in range(100)}
    exclusions = {"id000": ["exposed"]}
    expected = sorted(set(frame) - set(exclusions))
    random.Random(42).shuffle(expected)
    assert e08.select_ids(frame, exclusions) == sorted(expected[:50])
    assert e08.select_ids(dict(reversed(list(frame.items()))), exclusions) == sorted(expected[:50])
    with pytest.raises(ValueError, match="eligible"):
        e08.select_ids(dict(list(frame.items())[:50]), exclusions)


def test_prepare_and_verify_preserve_failures_blinding_and_originals(project):
    root, selected, records = project
    source = root / e08.RUNS["coverage"] / "responses" / f"{records[('coverage', selected[0], 0)]['cache_key']}.json"
    before = source.read_bytes()
    result = e08.prepare(root)
    assert result["model_calls"] == 0 and result["selected"] == 50
    assert e08.verify(root)["status"] == "verified"
    selection = json.loads((root / e08.EXPERIMENT_DIR / "selection.json").read_bytes())
    assert selection["input_ids"] == selected
    inputs = (root / e08.EXPERIMENT_DIR / "inputs.jsonl").read_text()
    assert "DO_NOT_EXPORT_QA" not in inputs and "is_supporting" not in inputs
    preflight = json.loads((root / e08.EXPERIMENT_DIR / "preflight.json").read_bytes())
    assert preflight["source_tasks"]["coverage"][0]["status"] == "truncated"
    assert preflight["source_tasks"]["baseline_fixed"][1]["n_facts"] == 0
    for name, filename, fields in (
        ("binary", "candidates.jsonl", {"annotation_id", "title", "paragraph_text", "fact"}),
        ("merger_audit", "pairs.jsonl", {"audit_id", "title", "paragraph_text", "a", "b"}),
    ):
        folder = root / e08.CALIBRATION_DIR / name
        rows = [json.loads(line) for line in (folder / filename).read_text().splitlines()]
        assert len(rows) == 24 and all(set(row) == fields for row in rows)
        task = (folder / "TASK.md").read_text()
        assert f"{e08.CALIBRATION_DIR}/{name}/{filename}" in task
        assert "AutoOntik" not in task and "round1/" not in task
    assert source.read_bytes() == before
    with pytest.raises(ValueError, match="not overwritten"):
        e08.prepare(root)
    # New annotations do not mutate the immutable preparation lock.
    (root / e08.CALIBRATION_DIR / "binary/judgments.jsonl").write_text("unscored work\n")
    assert e08.verify(root)["status"] == "verified"


def test_lock_detects_modified_files_and_new_ambiguous_sources(project):
    root, selected, records = project
    e08.prepare(root)
    path = root / e08.EXPERIMENT_DIR / "selection.json"
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="Lock mismatch"):
        e08.verify(root)
    path.write_bytes(original)
    duplicate = copy.deepcopy(records[("coverage", selected[2], 0)])
    write_json(root, f"{e08.RUNS['coverage']}/responses/extra.json", duplicate)
    with pytest.raises(ValueError, match="discovery changed"):
        e08.verify(root)


def test_preflight_rejects_ambiguous_or_mock_sources_without_writing(project):
    root, selected, records = project
    duplicate = copy.deepcopy(records[("coverage", selected[2], 0)])
    duplicate["cache_key"] = "duplicate"
    rel = f"{e08.RUNS['coverage']}/responses/duplicate.json"
    write_json(root, rel, duplicate)
    with pytest.raises(ValueError, match="Ambiguous duplicate"):
        e08.prepare(root)
    assert not (root / e08.EXPERIMENT_DIR).exists()
    # Remove only this test's synthetic duplicate to exercise a separate failure.
    (root / rel).unlink()
    original = records[("coverage", selected[2], 0)]
    mock = dict(original, mock=True)
    write_json(root, f"{e08.RUNS['coverage']}/responses/{mock['cache_key']}.json", mock)
    with pytest.raises(ValueError, match="completed real"):
        e08.prepare(root)
    assert not (root / e08.EXPERIMENT_DIR).exists()


def test_preflight_rejects_changed_payload_and_wrong_heuristic_provenance(project):
    root, selected, records = project
    rel = f"{e08.V2_RUN}/responses/synthetic-development-merger.json"
    payload = json.loads((root / rel).read_bytes())
    changed = copy.deepcopy(payload)
    changed["payload"]["candidates"][0]["fact"]["object"] = "silently repaired"
    write_json(root, rel, changed)
    with pytest.raises(ValueError, match="differs from original"):
        e08.prepare(root)
    write_json(root, rel, payload)
    rel = "runs/ensemble_xprompt/merged_union.json"
    union = json.loads((root / rel).read_bytes())
    union["results"][selected[2]][0]["sources"][0]["cache_key"] = "wrong-run"
    write_json(root, rel, union)
    with pytest.raises(ValueError, match="different extraction source"):
        e08.prepare(root)
    assert not (root / e08.CALIBRATION_DIR).exists()


def test_lock_rejects_new_exposure_and_keeps_existing_annotation_work(project):
    root, _, _ = project
    e08.prepare(root)
    write_jsonl(root, "runs/gptoss_pilot_extra/index.jsonl", [{"input_id": "new-exposure"}])
    with pytest.raises(ValueError, match="discovery changed"):
        e08.verify(root)


def test_preparation_refuses_existing_calibration_without_touching_it(tmp_path):
    path = tmp_path / e08.CALIBRATION_DIR / "binary/judgments.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("keep me")
    with pytest.raises(ValueError, match="not overwritten"):
        e08.prepare(tmp_path)
    assert path.read_text() == "keep me"


def test_input_validation_and_path_boundary(tmp_path):
    row = {"input_id": input_id("A", "text"), "title": "A", "paragraph_text": "text"}
    with pytest.raises(ValueError, match="Duplicate input"):
        e08.input_map([row, row])
    with pytest.raises(ValueError, match="does not match"):
        e08.input_map([dict(row, paragraph_text="changed")])
    with pytest.raises(ValueError, match="project-relative"):
        e08.local_path(tmp_path, "../outside")
    (tmp_path / "external-link").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="leaves project"):
        e08.local_path(tmp_path, "external-link/anything")
