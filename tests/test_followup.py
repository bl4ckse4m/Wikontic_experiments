"""Offline regression tests. Only synthetic temporary credentials are read."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pytest

from wikontic_exp.aggregation import Candidate
from wikontic_exp.api_client import AccessConfig, APICallResult, _load_dotenv
from wikontic_exp.binary_judging import allocation, prepare, score, wilson
from wikontic_exp.llm_dedup import materialize, validate_partition


def test_external_credentials_runtime_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in list(__import__("os").environ):
        if name.upper().startswith(("LLM_", "WIKONTIC_", "OPENAI_")) or name.upper() in {"MODEL", "MODEL_NAME", "API_KEY", "API_BASE"}:
            monkeypatch.delenv(name)
    # Fake file ONLY, never the user's HOME or real credentials.
    creds = tmp_path / "fake-creds"
    creds.write_text('export llm_url="https://synthetic.invalid/v1"\nllm_key: synthetic-key\n')
    (tmp_path / ".env").write_text("api_base=stale\napi_key=stale\nmodel_name=test-model\n")
    cfg = AccessConfig.load(creds_file=creds)
    assert cfg.api_base == "https://synthetic.invalid/v1"
    assert cfg.api_key == "synthetic-key" and cfg.model == "test-model"
    assert "synthetic-key" not in repr(cfg)
    assert "synthetic.invalid" not in repr(cfg)
    assert cfg.redact("synthetic-key https://synthetic.invalid/v1") == "[REDACTED] [REDACTED]"
    monkeypatch.setenv("API_KEY", "explicit-env")
    assert AccessConfig.load(creds_file=creds).api_key == "explicit-env"
    with pytest.raises(ValueError, match="does not exist"):
        AccessConfig.load(creds_file=tmp_path / "missing")


def test_colon_credentials_with_equals_in_value(tmp_path):
    path = tmp_path / "synthetic"
    path.write_text("llm_url: https://synthetic.invalid/v1?q=x=y\nllm_key: 'fake=='\n")
    assert _load_dotenv(path) == {"LLM_URL": "https://synthetic.invalid/v1?q=x=y", "LLM_KEY": "fake=="}


@pytest.mark.parametrize("answer", [
    {"groups": [{"members": [0, 0], "representative": 0}]},
    {"groups": [{"members": [0], "representative": 0}]},
    {"groups": [{"members": [0, 2], "representative": 0}]},
    {"groups": [{"members": [0, 1], "representative": 3}]},
    {"groups": [{"members": [False, 1], "representative": 1}]},
    {"groups": []},
])
def test_partition_rejects_lost_or_invented_candidates(answer):
    with pytest.raises(ValueError):
        validate_partition(answer, 2)


def test_representative_is_not_rewritten_or_qualifier_union():
    a = {"subject": "A", "relation": "r", "object": "B", "qualifiers": []}
    b = dict(a, relation="paraphrase")
    source = {"model": "m", "prompt_hash": "p", "gen_index": 0, "input_id": "i"}
    cands = [Candidate(a, source), Candidate(b, dict(source, prompt_hash="q")), Candidate(a, source)]
    groups = validate_partition({"groups": [{"members": [0, 1, 2], "representative": 1}]}, 3)
    merged = materialize(groups, cands)[0]
    assert merged["fact"] == b
    assert merged["n_support"] == 2  # duplicate within one response is not another vote
    assert len(merged["sources"]) == 3


def test_strata_allocation_and_intervals():
    assert allocation({"a": 90, "b": 10}, 20) == {"a": 18, "b": 2}
    assert allocation({"a": 2, "b": 1}, 10) == {"a": 2, "b": 1}
    with pytest.raises(ValueError):
        allocation({"a": 1, "b": 1}, 1)
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi


def test_blind_sampling_deterministic_and_weighted_scoring(tmp_path):
    fact = {"subject": "A", "relation": "r", "object": "B", "qualifiers": []}
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text(json.dumps({"input_id": "i", "title": "A", "paragraph_text": "A r B.",
                                 "question": "DO_NOT_LEAK", "refs": []}) + "\n")
    merged = tmp_path / "merged.json"
    merged.write_text(json.dumps({"results": {"i": [
        {"fact": fact, "n_support": 1},
        {"fact": fact, "n_support": 1},
        {"fact": dict(fact, object="C"), "n_support": 2},
    ]}}))
    out = tmp_path / "annotation"
    args = argparse.Namespace(out_dir=str(out), inputs=str(inputs), methods=[f"hidden_method={merged}"],
                              n=3, seed=42)
    summary = prepare(args)
    assert summary["unique_facts_to_judge"] == 2
    public = (out / "candidates.jsonl").read_text()
    assert "hidden_method" not in public and "DO_NOT_LEAK" not in public
    assert "n_support" not in public and "source" not in public
    args.out_dir = str(tmp_path / "repeat")
    prepare(args)
    assert (Path(args.out_dir) / "candidates.jsonl").read_text() == public
    with pytest.raises(ValueError, match="not overwritten"):
        prepare(args)
    candidates = [json.loads(line) for line in public.splitlines()]
    judgments = tmp_path / "judgments.jsonl"
    judgments.write_text("")
    scoring = argparse.Namespace(manifest=str(out / "sampling_private.json"), judgments=str(judgments),
                                 out=str(tmp_path / "score.json"))
    incomplete = score(scoring)
    assert incomplete["methods"]["hidden_method"]["missing"] == 3
    assert "full_label" not in incomplete["methods"]["hidden_method"]
    labels = [dict(annotation_id=r["annotation_id"], core_label=int(r["fact"]["object"] == "B"),
                   full_label=int(r["fact"]["object"] == "B"), status="silver", annotator="test",
                   evidence="A r B.", rationale="synthetic test") for r in candidates]
    judgments.write_text("".join(json.dumps(r) + "\n" for r in labels))
    result = score(scoring)
    assert result["methods"]["hidden_method"]["full_label"]["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == "not measured"
    labels[0]["full_label"] = True
    judgments.write_text("".join(json.dumps(r) + "\n" for r in labels))
    with pytest.raises(ValueError, match="integers"):
        score(scoring)


def test_llm_dedup_cache_raw_checkpoint_and_explicit_failure(tmp_path, monkeypatch):
    from wikontic_exp import llm_dedup

    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text(json.dumps({"input_id": "i", "title": "A", "paragraph_text": "A r B.",
                                 "question": "NO_QA_LEAK"}) + "\n")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("cluster originals")
    fact = {"subject": "A", "relation": "r", "object": "B", "qualifiers": []}
    records = [{"status": "ok", "cache_key": "original", "mock": False,
                "task": {"input_id": "i", "model": "test", "prompt_hash": "p", "gen_index": 0},
                "parsed": {"triplets": [fact, fact]}}]
    monkeypatch.setattr(llm_dedup, "load_run_records", lambda path: records)
    # Stub configuration entirely; NEVER load external credentials in this test.
    cfg = AccessConfig("https://synthetic.invalid", "synthetic", "test", max_concurrency=1)
    monkeypatch.setattr(AccessConfig, "load", classmethod(lambda cls, *a, **kw: cfg))
    calls = []

    class FakeClient:
        def __init__(self, cfg):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

        async def chat(self, model, messages, params, gen_index, key):
            assert "NO_QA_LEAK" not in json.dumps(messages)
            calls.append(key)
            return APICallResult(
                ok=True, raw_response={"choices": [{"finish_reason": "length" if params["max_tokens"] == 1 else "stop"}]},
                text='{"groups":[{"members":[0,1],"representative":0}]}',
                usage=None, model=model, latency_s=0, attempts=1)

    monkeypatch.setattr(llm_dedup, "OpenAICompatClient", FakeClient)
    original_validate = llm_dedup.validate_partition
    def check_raw_checkpoint(answer, n):
        # The API result must already be durable when validation starts.
        assert list((tmp_path / "dedup" / "responses").glob("*.json"))
        return original_validate(answer, n)
    monkeypatch.setattr(llm_dedup, "validate_partition", check_raw_checkpoint)
    args = argparse.Namespace(inputs=str(inputs), run_dirs=["synthetic"], limit=None,
                              selection_run_dir=None, seed=42, access=None, creds_file=None,
                              model=None, prompt=str(prompt), max_tokens=6000,
                              out_dir=str(tmp_path / "dedup"), retry_failures=False)
    assert asyncio.run(llm_dedup.run(args)) == {"selected": 1, "ok": 1, "failed": 0}
    assert asyncio.run(llm_dedup.run(args))["ok"] == 1 and len(calls) == 1
    args.max_tokens = 1  # parameter changes cannot reuse previous valid response
    assert asyncio.run(llm_dedup.run(args))["failed"] == 1 and len(calls) == 2
    result = json.loads((tmp_path / "dedup" / "merged_union.json").read_text())
    assert result["results"] == {} and "Truncated" in result["failures"]["i"]
