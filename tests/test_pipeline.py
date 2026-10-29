"""Mandatory checks (handoff §10) as automated tests."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from wikontic_exp import aggregation, data_prep, evaluation
from wikontic_exp.api_client import AccessConfig, APICallResult, OpenAICompatClient, TransientAPIError
from wikontic_exp.hashing import extraction_cache_key
from wikontic_exp.runner import MethodConfig, Runner, TaskSpec
from wikontic_exp.schema import SchemaError, parse_extraction


# ---------------------------------------------------------------- id integrity

def test_ids_preserved_and_qa_disjoint(tmp_path: Path):
    pool = [
        {"id": f"s{i}", "paragraphs": [
            {"idx": 0, "title": f"T{i}", "paragraph_text": f"text {i} unique", "is_supporting": True},
            {"idx": 1, "title": f"D{i}", "paragraph_text": f"distractor {i}", "is_supporting": False},
        ], "question": "q", "answer": "a", "answer_aliases": [], "question_decomposition": []}
        for i in range(60)
    ]
    t, c, info = data_prep.sample_split(pool, tuning_n=10, collection_n=20, seed=42)
    assert len(t) == 10 and len(c) == 20
    assert not (set(r["id"] for r in t) & set(r["id"] for r in c))
    assert data_prep.qa_id_overlap(t, c) == []


def test_paragraph_duplicates_kept_with_all_refs():
    recs = [
        {"id": "a", "paragraphs": [{"idx": 0, "title": "X", "paragraph_text": "same", "is_supporting": True}]},
        {"id": "b", "paragraphs": [{"idx": 0, "title": "X", "paragraph_text": "same", "is_supporting": False}]},
    ]
    uniq = data_prep.build_unique_inputs(recs)
    assert len(uniq) == 1
    (u,) = uniq.values()
    assert sorted(r.sample_id for r in u.refs) == ["a", "b"]


def test_extraction_inputs_have_no_question_or_answer(tmp_path: Path):
    recs = [{"id": "a", "paragraphs": [{"idx": 0, "title": "X", "paragraph_text": "y", "is_supporting": True}],
             "question": "SECRET_Q", "answer": "SECRET_A", "answer_aliases": ["z"], "question_decomposition": [{"k": 1}]}]
    uniq = data_prep.build_unique_inputs(recs)
    p = tmp_path / "inputs.jsonl"
    data_prep.write_inputs_jsonl(p, uniq)
    raw = p.read_text()
    assert "SECRET" not in raw
    keys = set(json.loads(raw.splitlines()[0]).keys())
    assert keys == {"input_id", "title", "paragraph_text", "refs"}


# ---------------------------------------------------------------- retries

class _FlakyTransport:
    """Stand-in client: transient failures first, then success; then permanent error."""

    def __init__(self, transient_failures: int = 2):
        self.transient_failures = transient_failures
        self.calls = 0

    async def chat(self, model, messages, gen_params, gen_index, task_id):
        self.calls += 1
        if self.calls <= self.transient_failures:
            return APICallResult(ok=False, raw_response=None, text=None, usage=None, model=model,
                                 latency_s=0.0, attempts=self.calls, error="http 503",
                                 error_kind="transient_exhausted")
        return APICallResult(ok=True, raw_response={"choices": [{"message": {"content": '{"triplets": []}'}}]},
                             text='{"triplets": []}', usage=None, model=model, latency_s=0.0,
                             attempts=self.calls)


def test_transient_then_success_and_permanent_failure(tmp_path: Path):
    method = MethodConfig(name="m", prompt_path=str(tmp_path / "p.txt"), model="m",
                          gen_params={"temperature": 0}, n_generations=1)
    (tmp_path / "p.txt").write_text("prompt")
    runner = Runner(method, tmp_path / "run", access=None, mock=True)
    flaky = _FlakyTransport(transient_failures=2)
    key = runner.task_key(TaskSpec("i1", "t", "p", 0))
    task = TaskSpec("i1", "t", "p", 0)
    messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]

    async def _run():
        res = None
        for _ in range(3):  # as done by the retrying client: retries share one task key
            res = await flaky.chat("m", messages, {}, 0, key)
        return runner._build_record(task, key, res)

    rec = asyncio.run(_run())
    assert rec.status == "ok" and rec.attempts == 3 and flaky.calls == 3

    perm = APICallResult(ok=False, raw_response=None, text=None, usage=None, model="m",
                         latency_s=0.0, attempts=1, error="http 401", error_kind="permanent", status_code=401)
    rec = runner._build_record(task, key, perm)
    assert rec.status == "api_permanent_error"

    transient_exhausted = APICallResult(ok=False, raw_response=None, text=None, usage=None, model="m",
                                        latency_s=0.0, attempts=5, error="http 503",
                                        error_kind="transient_exhausted")
    rec = runner._build_record(task, key, transient_exhausted)
    assert rec.status == "api_error"


def test_http_client_retries_only_transient(monkeypatch):
    """429/5xx are retried; 400/401 fail fast (no infinite retries)."""
    import tenacity

    monkeypatch.setattr(tenacity, "wait_exponential_jitter", lambda **kw: (lambda rs: 0.0))
    cfg = AccessConfig(api_base="http://x/v1", api_key="k", model="m", max_retries=3, max_concurrency=1)
    calls = {"n": 0}

    class FakeResp:
        def __init__(self, status, text="", headers=None):
            self.status_code = status
            self.text = text
            self.headers = headers or {}

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    async def fake_post(self, url, json=None):
        calls["n"] += 1
        if calls["n"] < 3:
            return FakeResp(503, "busy")
        return FakeResp(200)

    async def fake_post_perm(self, url, json=None):
        calls["n"] += 1
        return FakeResp(400, "bad params")

    import httpx
    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    async def run_ok():
        async with OpenAICompatClient(cfg) as c:
            return await c.chat("m", [{"role": "user", "content": "x"}], {}, 0, "t")

    res = asyncio.run(run_ok())
    assert res.ok and calls["n"] == 3 and res.attempts == 3

    calls["n"] = 0
    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post_perm)

    async def run_perm():
        async with OpenAICompatClient(cfg) as c:
            return await c.chat("m", [{"role": "user", "content": "x"}], {}, 0, "t")

    res = asyncio.run(run_perm())
    assert not res.ok and res.error_kind == "permanent" and calls["n"] == 1


# ---------------------------------------------------------------- parsing failures

def test_invalid_json_is_failure_not_empty_success():
    with pytest.raises(SchemaError):
        parse_extraction('{"triplets": [{"subject": "broken"')
    try:
        parse_extraction("not json at all")
    except SchemaError as e:
        assert "JSON" in str(e) or "json" in str(e)


def test_missing_usage_ok_and_missing_triplets_fails():
    res = parse_extraction('{"triplets": [{"subject": "a", "relation": "r", "object": "b", "qualifiers": []}]}')
    assert len(res.facts) == 1
    with pytest.raises(SchemaError):
        parse_extraction('{"facts": []}')


# ---------------------------------------------------------------- cache / resume / generations

def test_independent_generations_have_distinct_cache_keys():
    base = dict(input_hash="h", model="m", prompt_hash="p", schema_id="s", gen_params={"temperature": 0.7})
    k0 = extraction_cache_key(**base, gen_index=0)
    k1 = extraction_cache_key(**base, gen_index=1)
    assert k0 != k1
    # a retry of the same generation keeps the key
    assert k0 == extraction_cache_key(**base, gen_index=0)


def test_resume_and_failure_paths(tmp_path: Path):
    prompt = tmp_path / "p.txt"
    prompt.write_text("prompt text")
    method = MethodConfig(name="m", prompt_path=str(prompt), model="mock-model",
                          gen_params={"temperature": 0}, n_generations=1)
    runner = Runner(method, tmp_path / "run", access=None, mock=True)
    inputs = [{"input_id": f"i{i}", "title": f"t{i}", "paragraph_text": f"p{i}"} for i in range(20)]
    stats1 = asyncio.run(runner.run(inputs, resume=True))
    assert stats1["total"] == 20 and stats1["computed"] + stats1["failed"] == 20
    stats2 = asyncio.run(runner.run(inputs, resume=True))
    assert stats2["cached"] == 20 and stats2["computed"] == 0
    # corrupted cache file -> recomputed, no crash
    key = runner.task_key(TaskSpec("i0", "t0", "p0", 0))
    runner.response_path(key).write_text("{corrupt")
    stats3 = asyncio.run(runner.run(inputs, resume=True))
    assert stats3["computed"] + stats3["failed"] >= 1


def test_mock_produces_invalid_json_and_api_error_paths(tmp_path: Path):
    prompt = tmp_path / "p.txt"
    prompt.write_text("prompt")
    method = MethodConfig(name="m", prompt_path=str(prompt), model="mock",
                          gen_params={"temperature": 0}, n_generations=1)
    runner = Runner(method, tmp_path / "run", access=None, mock=True)
    inputs = [{"input_id": f"i{i:02d}", "title": "t", "paragraph_text": f"unique text {i}"} for i in range(60)]
    stats = asyncio.run(runner.run(inputs))
    assert stats["failed"] > 0  # deterministic mock slices trigger failure paths
    statuses = set()
    for p in (tmp_path / "run" / "responses").glob("*.json"):
        statuses.add(json.loads(p.read_text())["status"])
    assert "ok" in statuses
    assert statuses & {"invalid_json", "api_permanent_error", "api_error", "schema_error", "truncated"}


# ---------------------------------------------------------------- aggregation

def _fact(s, r, o, quals=None):
    return {"subject": s, "relation": r, "object": o, "qualifiers": quals or []}


def test_merge_paraphrases_but_keep_distinct_episodes():
    cands = [
        aggregation.Candidate(_fact("Marie Curie", "award received", "Nobel Prize in Physics",
                                    [{"relation": "point in time", "object": "1903"}]),
                              {"input_id": "x", "model": "m", "prompt_hash": "p", "gen_index": 0, "cache_key": "a"}),
        aggregation.Candidate(_fact("Marie Curie", "award received", "Nobel Prize in Physics",
                                    [{"relation": "point in time", "object": "1903"}]),
                              {"input_id": "x", "model": "m", "prompt_hash": "p", "gen_index": 1, "cache_key": "b"}),
        aggregation.Candidate(_fact("Marie Curie", "award received", "Nobel Prize in Physics",
                                    [{"relation": "point in time", "object": "1911"}]),
                              {"input_id": "x", "model": "m", "prompt_hash": "p", "gen_index": 0, "cache_key": "c"}),
        aggregation.Candidate(_fact("marie  curie", "award received", "nobel prize in physics",
                                    [{"relation": "point in time", "object": "1903"}]),
                              {"input_id": "x", "model": "m", "prompt_hash": "p", "gen_index": 2, "cache_key": "d"}),
    ]
    agg = aggregation.aggregate(cands, algorithm="union_dedup")
    groups = agg["results"]["x"]
    # paraphrase merges with 1903 group; the 1911 episode stays separate
    assert len(groups) == 2
    supports = sorted(g["n_support"] for g in groups)
    assert supports == [1, 3]
    # conflicting date is logged, not fused into a range
    assert any(d["action"] == "kept_separate" for d in agg["decisions"])


def test_vote_threshold():
    cands = [
        aggregation.Candidate(_fact("a", "r", "b"), {"input_id": "x", "model": "m", "prompt_hash": "p", "gen_index": i, "cache_key": str(i)})
        for i in range(2)
    ] + [aggregation.Candidate(_fact("a", "r", "c"), {"input_id": "x", "model": "m", "prompt_hash": "p", "gen_index": 5, "cache_key": "z"})]
    agg = aggregation.aggregate(cands, algorithm="vote", vote_threshold=2)
    facts = [g["fact"]["object"] for g in agg["results"]["x"]]
    assert facts == ["b"]
    assert any(d["action"] == "dropped_by_vote" for d in agg["decisions"])


def test_negated_and_affirmed_never_merge():
    a = _fact("X", "built", "Y", [{"relation": "polarity", "object": "negative"}])
    b = _fact("X", "built", "Y", [])
    ok, conflicts = aggregation.same_claim(a, b)
    assert not ok


# ---------------------------------------------------------------- evaluation matching

def test_one_to_one_matching_prevents_recall_inflation():
    gold = [_fact("a", "r", "b")]
    preds = [_fact("a", "r", "b"), _fact("a", "r", "b"), _fact("a", "r", "b")]
    m = evaluation.match_facts(preds, gold, threshold=0.9)
    assert len(m["matches"]) == 1
    assert evaluation.prf(len(m["matches"]), m["n_pred"], m["n_gold"])["recall"] == 1.0


def test_qa_em_f1():
    assert evaluation.qa_em_f1("1960", ["1960"])["em"] == 1.0
    assert evaluation.qa_em_f1("in 1960", ["1960"])["f1"] == 0.6666666666666666
