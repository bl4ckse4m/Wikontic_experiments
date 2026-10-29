"""Offline OpenRouter opt-in tests; never read real credentials or call an API."""
from __future__ import annotations

import os

import pytest

from wikontic_exp.api_client import AccessConfig
from wikontic_exp.cli import build_parser
from wikontic_exp.hashing import extraction_cache_key, hash_obj
from wikontic_exp.runner import MethodConfig, Runner, TaskSpec


@pytest.fixture
def credentials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.upper().startswith(("OPENROUTER_", "OPENAI_", "LLM_", "WIKONTIC_")) or key.upper() in {"API_KEY", "API_BASE", "MODEL", "MODEL_NAME"}:
            monkeypatch.delenv(key)
    path = tmp_path / "synthetic-creds"
    path.write_text("llm_url=https://university.invalid/v1\nllm_key=fake-university\n"
                    "openrouter_key=fake-router\n")
    return path


def test_openrouter_requires_explicit_opt_in(credentials, monkeypatch):
    monkeypatch.setenv("WIKONTIC_MODEL", "Openai/Gpt-oss-120b")
    primary = AccessConfig.load(creds_file=credentials)
    router = AccessConfig.load(creds_file=credentials, openrouter=True)
    assert primary.provider == "primary" and primary.api_key == "fake-university"
    assert primary.model == "Openai/Gpt-oss-120b"
    assert router.provider == "openrouter" and router.api_key == "fake-router"
    assert router.api_base == "https://openrouter.ai/api/v1"
    assert router.model == "openai/gpt-oss-120b"
    assert not router.missing()
    assert primary.cache_route is None
    assert router.cache_route["provider"] == "openrouter"
    assert "fake-router" not in repr(router)
    assert "fake-router" not in str(router.routing_metadata)


def test_missing_openrouter_key_never_borrows_primary_key(credentials):
    credentials.write_text("llm_url=https://university.invalid/v1\nllm_key=fake-university\n")
    cfg = AccessConfig.load(creds_file=credentials, openrouter=True)
    assert cfg.api_key == ""
    assert "openrouter_key / OPENROUTER_API_KEY" in cfg.missing()


def test_openrouter_overrides_are_separate(credentials, monkeypatch):
    with credentials.open("a") as f:
        f.write("openrouter_model=custom/model\nopenrouter_url=https://router.invalid/v1/\n")
    monkeypatch.setenv("API_KEY", "fake-stale-primary")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-explicit-router")
    cfg = AccessConfig.load(creds_file=credentials, openrouter=True)
    assert cfg.api_key == "fake-explicit-router"
    assert cfg.model == "custom/model" and cfg.api_base == "https://router.invalid/v1"


def test_route_separates_cache_even_for_identical_model(credentials, tmp_path):
    primary = AccessConfig.load(creds_file=credentials)
    router = AccessConfig.load(creds_file=credentials, openrouter=True)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("extract")
    method = MethodConfig("m", str(prompt), "same/model", {"max_tokens": 6000})
    a = Runner(method, tmp_path / "a", primary)
    b = Runner(method, tmp_path / "b", router)
    task = TaskSpec("i", "title", "text", 0)
    assert a.task_key(task) != b.task_key(task)
    legacy_payload = {"input_hash": task.input_hash, "model": method.model,
                      "prompt_hash": method.prompt_hash, "schema_id": method.schema_id,
                      "gen_params": method.gen_params, "gen_index": 0}
    assert a.task_key(task) == hash_obj(legacy_payload)[:32]
    assert b.task_key(task) == extraction_cache_key(**legacy_payload, route=router.cache_route)


def test_cli_openrouter_flags():
    parser = build_parser()
    assert parser.parse_args(["probe", "--openrouter"]).openrouter
    assert not parser.parse_args(["probe"]).openrouter
    args = parser.parse_args(["run", "--openrouter", "--method", "m.json",
                              "--inputs", "i.jsonl", "--run-dir", "r"])
    assert args.openrouter
