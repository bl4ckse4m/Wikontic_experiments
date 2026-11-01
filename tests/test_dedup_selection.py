"""Exact locked merger input selection, using synthetic API/configuration doubles."""
from __future__ import annotations

import argparse
import asyncio
import json

import pytest

from wikontic_exp import llm_dedup
from wikontic_exp.api_client import AccessConfig, APICallResult


def test_exact_selection_rejects_duplicates_and_missing_inputs(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"input_ids": ["b", "a"]}))
    assert llm_dedup.exact_selection(str(path), {"a", "b"}, {"a", "b"}) == ["a", "b"]
    for ids in (["a", "a"], [], [False], ["c"]):
        path.write_text(json.dumps({"input_ids": ids}))
        with pytest.raises(ValueError):
            llm_dedup.exact_selection(str(path), {"a", "b"}, {"a", "b"})
    path.write_text(json.dumps({"input_ids": ["a"]}))
    with pytest.raises(ValueError, match="source paragraphs"):
        llm_dedup.exact_selection(str(path), {"a"}, {"b"})


def test_locked_frame_is_not_intersected_with_successes(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text("".join(json.dumps({"input_id": iid, "title": iid, "paragraph_text": iid}) + "\n"
                              for iid in ("a", "b", "outside")))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"input_ids": ["b", "a"]}))
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("synthetic prompt")
    fact = {"subject": "A", "relation": "r", "object": "B", "qualifiers": []}
    records = [{"mock": False, "status": status, "cache_key": iid,
                "task": {"input_id": iid, "model": "test", "prompt_hash": "p", "gen_index": 0},
                "parsed": {"triplets": [fact]} if status == "ok" else None}
               for iid, status in (("a", "ok"), ("b", "truncated"), ("outside", "ok"))]
    monkeypatch.setattr(llm_dedup, "load_run_records", lambda path: records)
    cfg = AccessConfig("https://synthetic.invalid/v1", "synthetic-key", "test", max_concurrency=1)
    monkeypatch.setattr(AccessConfig, "load", classmethod(lambda cls, *a, **kw: cfg))
    calls = []

    class Client:
        def __init__(self, cfg):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def chat(self, model, messages, params, gen_index, key):
            calls.append(json.loads(messages[1]["content"])["title"])
            return APICallResult(ok=True, text='{"groups":[{"members":[0],"representative":0}]}',
                                 raw_response={"choices": [{"finish_reason": "stop"}]},
                                 model=model, latency_s=0, attempts=1, usage=None)

    monkeypatch.setattr(llm_dedup, "OpenAICompatClient", Client)
    args = argparse.Namespace(inputs=str(inputs), run_dirs=["synthetic"], selection=str(selection),
                              selection_run_dir=None, limit=None, seed=42, access=None, creds_file=None,
                              model=None, prompt=str(prompt), max_tokens=6000, retry_failures=False,
                              out_dir=str(tmp_path / "out"))
    assert asyncio.run(llm_dedup.run(args)) == {"selected": 2, "ok": 1, "failed": 1}
    assert calls == ["a"]
    merged = json.loads((tmp_path / "out/merged_union.json").read_text())
    assert merged["selected_input_ids"] == ["a", "b"]
    assert "b" in merged["failures"] and "outside" not in merged["results"]
    for field, value in (("limit", 1), ("selection_run_dir", "other")):
        setattr(args, field, value)
        with pytest.raises(ValueError, match="cannot be combined"):
            asyncio.run(llm_dedup.run(args))
        setattr(args, field, None)
