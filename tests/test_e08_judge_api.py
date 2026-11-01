"""Stateless judge transport: synthetic requests/responses, no paid calls or credentials."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("e08_api_test_adapter", REPO / "scripts/e08_judge_api.py")
api = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(api)


def save(root, rel, data):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(api.stage.e08_preparation.json_bytes(data))


def save_rows(root, rel, data):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(api.stage.e08_preparation.jsonl_bytes(data))


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(api.stage, "verify", lambda root: {"status": "verified"})
    for rel in ("scripts/e08_main.py", "scripts/e08_judge_api.py"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((REPO / rel).read_bytes())
    for kind in api.KINDS:
        rel = f"{api.stage.MAIN}/{kind}/TASK.md"
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((REPO / rel).read_bytes())
    fact = {"subject": "A", "relation": "r", "object": "B", "qualifiers": []}
    facts = [{"annotation_id": f"b{i}", "input_id": "private-input-id", "title": "A",
              "paragraph_text": "A r B.", "fact": dict(fact, object=f"B{i}"),
              "n_support": 99, "method": "DO_NOT_SEND_METHOD"} for i in range(4)]
    pairs = [{"audit_id": f"p{i}", "title": "A", "paragraph_text": "A r B.",
              "a": fact, "b": fact} for i in range(2)]
    save_rows(tmp_path, f"{api.stage.MAIN}/binary/candidates.jsonl", facts)
    save_rows(tmp_path, f"{api.stage.MAIN}/merger_audit/pairs.jsonl", pairs)
    save(tmp_path, f"{api.stage.BASE}/judging_summary.json", {"unique_facts_to_judge": 4, "pairs_to_judge": 2})
    save(tmp_path, f"{api.stage.BASE}/budget_status.json", {
        "reported_cumulative_spend_usd_approx": 1.54, "total_budget_usd": 5})
    for name in ("rubric_lock.json", "prediction_lock.json"):
        save(tmp_path, f"{api.stage.BASE}/{name}", {"synthetic": True})
    return tmp_path


def success(body, cost=0.001):
    data = json.loads(body["messages"][1]["content"])
    binary = body["response_format"]["json_schema"]["name"] == "e08_binary"
    key = "annotation_id" if binary else "audit_id"
    labels = [{key: r[key], "rationale": "synthetic",
               **({"core_label": 1, "full_label": 1, "evidence": "A r B."}
                  if binary else {"equivalent": 1})} for r in data["items"]]
    return {"status_code": 200, "error": None, "response": {
        "model": api.API_MODEL, "usage": {"cost": cost, "completion_tokens": 100},
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"judgments": labels})}}]}}


def test_prepare_is_blind_stateless_fixed_and_offline(project, monkeypatch):
    from wikontic_exp.api_client import AccessConfig
    monkeypatch.setattr(AccessConfig, "load", lambda *a, **kw: pytest.fail("Credential loading during offline prepare"))
    summary = api.prepare(project)
    assert summary["model_calls"] == 0
    plan, jobs = api.verify_plan(project)
    assert plan["workload"] == {"binary": 4, "merger_audit": 2, "requests": 2}
    assert api.prepare(project)["status"].startswith("already prepared")
    for job in jobs:
        body = job["request"]
        assert [m["role"] for m in body["messages"]] == ["system", "user"]
        assert body["model"] == api.API_MODEL and body["reasoning"] == {"effort": "low"}
        assert "tools" not in body and "models" not in body
        data = json.loads(body["messages"][1]["content"])
        assert set(data) == {"title", "paragraph_text", "items"}
        serialized = json.dumps(body)
        for private in ("DO_NOT_SEND_METHOD", "n_support", "private-input-id", "Read only", "calibration/judgments"):
            assert private not in serialized
        assert api.money(job["reservation_usd"]) > 0
    with pytest.raises(ValueError, match="change budget"):
        api.prepare(project, "3.10")


def test_one_attempt_cache_and_export_without_metadata_hallucinations(project, monkeypatch):
    api.prepare(project)
    plan, jobs = api.verify_plan(project)
    calls = []
    with api.exclusive_run(project):
        result = api.execute_jobs(project, plan, jobs, lambda body: (calls.append(body), success(body))[1])
    assert len(calls) == 2 and result["labels"] == {"binary": 4, "merger_audit": 2}
    assert api.money(result["reported_cost_usd"]) == api.money("0.002")
    for kind in api.KINDS:
        labels = api.stage.rows(project, f"{api.stage.MAIN}/{kind}/judgments.jsonl")
        assert all(r["status"] == "silver" and r["annotator"] == api.API_MODEL for r in labels)
    from wikontic_exp.api_client import AccessConfig
    monkeypatch.setattr(AccessConfig, "load", lambda *a, **kw: pytest.fail("Completed run loaded credentials"))
    resumed = api.run(project, None, True)
    assert resumed["labels"] == result["labels"]
    assert len(calls) == 2


def test_crash_after_reservation_is_never_replayed(project):
    api.prepare(project)
    plan, jobs = api.verify_plan(project)
    def crash(body):
        raise KeyboardInterrupt("synthetic interruption after the reservation")
    with pytest.raises(KeyboardInterrupt):
        api.execute_jobs(project, plan, jobs, crash)
    result = api.execute_jobs(project, plan, jobs, lambda body: pytest.fail("Uncertain request replayed"))
    assert result["halted"] and result["jobs"]["uncertain"] == 1
    assert api.money(result["accounted_usd_including_unknown_reservations"]) > 0


def test_budget_guard_stops_before_sending(project):
    api.prepare(project, "0.000001")
    plan, jobs = api.verify_plan(project)
    result = api.execute_jobs(project, plan, jobs, lambda body: pytest.fail("Exceeded budget"))
    assert result["halted"] and result["jobs"] == {"pending": 2}
    assert result["labels"] == {"binary": 0, "merger_audit": 0}


def test_missing_billed_cost_stops_with_conservative_debit(project):
    api.prepare(project)
    plan, jobs = api.verify_plan(project)
    calls = []
    def no_cost(body):
        calls.append(body)
        receipt = success(body)
        del receipt["response"]["usage"]["cost"]
        return receipt
    result = api.execute_jobs(project, plan, jobs, no_cost)
    assert result["halted"] and len(calls) == 1
    assert api.money(result["accounted_usd_including_unknown_reservations"]) > 0
    assert api.money(result["reported_cost_usd"]) == 0


def test_invalid_json_is_not_repaired_or_retried(project):
    api.prepare(project)
    plan, jobs = api.verify_plan(project)
    calls = []
    def first_bad(body):
        receipt = success(body)
        if not calls:
            receipt["response"]["choices"][0]["message"]["content"] = "```json\n{}\n```"
        calls.append(body)
        return receipt
    result = api.execute_jobs(project, plan, jobs, first_bad)
    assert len(calls) == 2 and result["jobs"] == {"failed": 1, "ok": 1}
    resumed = api.execute_jobs(project, plan, jobs, lambda body: pytest.fail("Failed request retried"))
    assert resumed["jobs"] == result["jobs"]


@pytest.mark.parametrize("choices", [None, [], [None], [{"finish_reason": "stop", "message": []}]])
def test_malformed_choice_is_a_retained_failure_not_a_crash(project, choices):
    api.prepare(project)
    plan, jobs = api.verify_plan(project)
    calls = []
    def malformed(body):
        receipt = success(body)
        if not calls:
            receipt["response"]["choices"] = choices
        calls.append(body)
        return receipt
    result = api.execute_jobs(project, plan, jobs, malformed)
    assert len(calls) == 2 and result["jobs"] == {"failed": 1, "ok": 1}
    assert not result["halted"]


def test_reservation_is_synced_before_send_and_records_cannot_be_overwritten(project, monkeypatch):
    api.prepare(project)
    plan, jobs = api.verify_plan(project)
    synced = []
    original_fsync = api.os.fsync
    def fsync(fd):
        synced.append(fd)
        return original_fsync(fd)
    monkeypatch.setattr(api.os, "fsync", fsync)
    def send(body):
        assert len(synced) >= 3
        return success(body)
    api.execute_jobs(project, plan, jobs, send)
    attempt_rel, _ = api.record_paths(jobs[0])
    original = api.stage.path(project, attempt_rel).read_bytes()
    with pytest.raises(FileExistsError):
        api.durable_record(project, attempt_rel, {"replacement": True})
    assert api.stage.path(project, attempt_rel).read_bytes() == original


def test_stale_export_temp_does_not_block_reconstruction(project):
    api.prepare(project)
    plan, jobs = api.verify_plan(project)
    stale = api.stage.path(project, f"{api.stage.MAIN}/binary/judgments.jsonl.api.tmp")
    stale.write_bytes(b"preserve old interrupted work")
    result = api.execute_jobs(project, plan, jobs, success)
    assert result["labels"] == {"binary": 4, "merger_audit": 2}
    assert stale.read_bytes() == b"preserve old interrupted work"


def test_existing_agent_labels_and_modified_api_outputs_are_preserved(project):
    rel = f"{api.stage.MAIN}/binary/judgments.jsonl"
    save_rows(project, rel, [{"annotation_id": "foreign-agent-result"}])
    original = api.stage.path(project, rel).read_bytes()
    with pytest.raises(ValueError, match="already exist"):
        api.prepare(project)
    assert api.stage.path(project, rel).read_bytes() == original


def test_export_never_overwrites_foreign_work(project):
    api.prepare(project)
    plan, jobs = api.verify_plan(project)
    api.execute_jobs(project, plan, jobs, success)
    rel = f"{api.stage.MAIN}/binary/judgments.jsonl"
    labels = api.stage.rows(project, rel)
    labels[0]["full_label"] = 0
    save_rows(project, rel, labels)
    original = api.stage.path(project, rel).read_bytes()
    with pytest.raises(ValueError, match="Foreign/modified"):
        api.sync_outputs(project, api.scan(project, jobs))
    assert api.stage.path(project, rel).read_bytes() == original


def test_lock_change_key_confirmation_and_concurrency(project):
    with pytest.raises(ValueError, match="confirm-key-limit"):
        api.run(project, None, False)
    api.prepare(project)
    with api.exclusive_run(project):
        with pytest.raises(ValueError, match="already active"):
            with api.exclusive_run(project):
                pass
    task = api.stage.path(project, f"{api.stage.MAIN}/binary/TASK.md")
    task.write_bytes(task.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Stage lock mismatch"):
        api.verify_plan(project)


def test_rubric_criteria_copied_and_complete_ids_required(project):
    api.prepare(project)
    _, jobs = api.verify_plan(project)
    for job in jobs:
        receipt = success(job["request"])
        assert len(api.decoded(job, receipt)) == len(job["ids"])
        data = json.loads(receipt["response"]["choices"][0]["message"]["content"])
        data["judgments"].pop()
        receipt["response"]["choices"][0]["message"]["content"] = json.dumps(data)
        with pytest.raises(ValueError, match="Incomplete"):
            api.decoded(job, receipt)
    original = api.stage.path(project, f"{api.stage.MAIN}/binary/TASK.md").read_text()
    criteria = original[original.index("For each row, assess"):original.index("Output schema (JSONL")].strip()
    assert criteria in api.rubric(project, "binary")


@pytest.mark.parametrize("invalid", ["NaN", "Infinity", "-1", True])
def test_invalid_budget_values(invalid):
    with pytest.raises(ValueError):
        api.money(invalid)
