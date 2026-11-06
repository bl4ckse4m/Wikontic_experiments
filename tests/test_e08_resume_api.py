"""Synthetic transport-only E08 continuation tests. No real credentials/network."""
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e08_resume_test", REPO / "scripts/e08_resume_api.py")
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)
REAL_LOAD = resume.load_api


def save(root, rel, data, rows=False):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in data) if rows else json.dumps(data))


def success(body):
    binary = body["response_format"]["json_schema"]["name"] == "e08_binary"
    key = "annotation_id" if binary else "audit_id"
    items = json.loads(body["messages"][1]["content"])["items"]
    labels = [{key: r[key], "rationale": "synthetic", **(
        {"core_label": 1, "full_label": 1, "evidence": "A r B."} if binary else {"equivalent": 1})} for r in items]
    return {"status_code": 200, "error": None, "response": {"model": "anthropic/claude-sonnet-5.5",
            "provider": "Anthropic", "usage": {"cost": 0.001},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"judgments": labels})}}]}}


def rejection():
    return {"status_code": 429, "error": None, "response": {"error": {"code": 429,
            "message": "OpenRouter could not verify available credits for this request in time. Retry shortly.",
            "metadata": {"limit_source": "openrouter_admission_control", "provider_name": None,
                         "headers": {"Retry-After": "0"}}}}}


@pytest.fixture
def project(tmp_path, monkeypatch):
    def fresh():
        api = REAL_LOAD()
        monkeypatch.setattr(api.stage, "verify", lambda root: {"status": "verified"})
        return api
    monkeypatch.setattr(resume, "load_api", fresh)
    api = fresh()
    for rel in ("scripts/e08_main.py", "scripts/e08_judge_api.py", "scripts/e08_resume_api.py",
                "tests/test_e08_resume_api.py", resume.AMENDMENT):
        p = tmp_path / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes((REPO / rel).read_bytes())
    for kind in api.KINDS:
        rel = f"{api.stage.MAIN}/{kind}/TASK.md"
        p = tmp_path / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes((REPO / rel).read_bytes())
    fact = {"subject": "A", "relation": "r", "object": "B", "qualifiers": []}
    save(tmp_path, f"{api.stage.MAIN}/binary/candidates.jsonl", [
        {"annotation_id": f"b{i}", "title": "A", "paragraph_text": "A r B." if i<2 else "C r D.", "fact": fact}
        for i in range(4)], rows=True)
    save(tmp_path, f"{api.stage.MAIN}/merger_audit/pairs.jsonl", [
        {"audit_id": f"p{i}", "title": "A", "paragraph_text": "A r B.", "a": fact, "b": fact}
        for i in range(2)], rows=True)
    save(tmp_path, f"{api.stage.BASE}/judging_summary.json", {"unique_facts_to_judge": 4, "pairs_to_judge": 2})
    for name in ("rubric_lock.json", "prediction_lock.json"):
        save(tmp_path, f"{api.stage.BASE}/{name}", {"synthetic": True})
    save(tmp_path, "key.json", {"limit_remaining": 5, "limit": 5})
    api.prepare(tmp_path, "key.json")
    plan,jobs = api.verify_plan(tmp_path)
    calls = []
    def send(body):
        calls.append(body)
        return success(body) if len(calls) == 1 else rejection()
    state = api.execute_jobs(tmp_path, plan, jobs, send,
                             lambda: {"status_code": 200, "accounting": {"limit_remaining": 5}})
    assert state["jobs"] == {"ok": 1, "failed": 1, "pending": 1}
    return tmp_path, api, jobs


def prepare(project):
    root,api,jobs = project
    resume.prepare(root, resume.DEFAULT_CONTINUATION, "key.json")
    adapted = resume.attach(root, resume.DEFAULT_CONTINUATION)
    plan,new = adapted.verify_plan(root)
    assert new == jobs
    return root,api,jobs,adapted,plan


def test_explicit_retry_preserves_original_ledger_and_never_rejudges_success(project):
    root,api,jobs,adapted,plan = prepare(project)
    old = {p.relative_to(root).as_posix(): p.read_bytes() for p in (root / resume.BASE_API).rglob("*.json")}
    calls=[]
    result = adapted.execute_jobs(root,plan,jobs,lambda b:(calls.append(b),success(b))[1],
                                  lambda:{"status_code":200,"accounting":{"limit_remaining":5}})
    assert len(calls)==2 and result["jobs"]=={"ok":3}
    assert result["labels"]=={"binary":4,"merger_audit":2}
    assert calls == [j["request"] for j in jobs[1:]]
    assert result["continuation"]["total_http_attempts"]==4
    assert result["reported_cost_usd"]=="0.003"
    assert adapted.money(result["accounted_usd_including_unknown_reservations"]) > adapted.money('0.003')
    assert result["missing_cost_jobs"]==["prior_attempt/"+jobs[1]["job_id"]]
    assert all((root/p).read_bytes()==b for p,b in old.items())
    again=adapted.execute_jobs(root,plan,jobs,lambda b:pytest.fail("A new attempt replayed"))
    assert again["jobs"]=={"ok":3}


def test_new429_is_terminal_and_not_replayed(project):
    root,api,jobs,adapted,plan=prepare(project)
    result=adapted.execute_jobs(root,plan,jobs,lambda b:rejection(),
                                lambda:{"status_code":200,"accounting":{"limit_remaining":5}})
    assert result["halted"] and result["jobs"]=={"ok":1,"failed":1,"pending":1}
    adapted.execute_jobs(root,plan,jobs,lambda b:pytest.fail("Second retry is forbidden"))
    assert (root/resume.BASE_API/'halt.json').exists()


def test_new_uncertain_attempt_never_replayed(project):
    root,api,jobs,adapted,plan=prepare(project)
    def crash(body): raise KeyboardInterrupt("simulated crash")
    with pytest.raises(KeyboardInterrupt):
        adapted.execute_jobs(root,plan,jobs,crash,
                             lambda:{"status_code":200,"accounting":{"limit_remaining":5}})
    result=adapted.execute_jobs(root,plan,jobs,lambda b:pytest.fail("Uncertain replay"))
    assert result["halted"] and result["jobs"]["uncertain"]==1


def test_duplicate_continuation_registration_forbidden(project):
    root,api,jobs,adapted,plan=prepare(project)
    with pytest.raises(ValueError,match="already authorized"):
        resume.prepare(root,"runs/another/api_judging","key.json")


def test_fresh_key_guard_still_blocks_before_post(project):
    root,api,jobs,adapted,plan=prepare(project)
    result=adapted.execute_jobs(root,plan,jobs,lambda b:pytest.fail("Key margin ignored"),
                                lambda:{"status_code":200,"accounting":{"limit_remaining":0.1}})
    assert result["halted"] and result["continuation"]["new_http_attempts"]==0


def test_continuation_cannot_include_successful_job(project):
    root,api,jobs,adapted,plan=prepare(project)
    path=root/resume.DEFAULT_CONTINUATION/'plan.json'
    cp=json.loads(path.read_bytes());cp['eligible_job_ids'].insert(0,jobs[0]['job_id']);path.write_text(json.dumps(cp))
    with pytest.raises(ValueError,match="allowlist includes"):
        resume.attach(root,resume.DEFAULT_CONTINUATION)


def test_model_answer_even_with429_is_not_retryable(project):
    root,api,jobs=project
    _,rel=api.record_paths(jobs[1]);receipt=api.stage.read(root,rel)
    receipt['response']['choices']=[]
    save(root,rel,receipt)
    with pytest.raises(ValueError,match="Only the observed"):
        resume.prepare(root,resume.DEFAULT_CONTINUATION,'key.json')


def test_new_malformed200_stops_before_next_post(project):
    root,api,jobs,adapted,plan=prepare(project)
    calls=[]
    def malformed(body):
        calls.append(body)
        receipt=success(body)
        receipt['response']['choices'][0]['message']['content']='not JSON'
        return receipt
    result=adapted.execute_jobs(root,plan,jobs,malformed,
                                lambda:{"status_code":200,"accounting":{"limit_remaining":5}})
    assert len(calls)==1 and result['halted']
    assert result['jobs']=={'ok':1,'failed':1,'pending':1}
    result=adapted.execute_jobs(root,plan,jobs,lambda b:pytest.fail('Malformed response replay or further POST'))
    assert result['jobs']=={'ok':1,'failed':1,'pending':1}
