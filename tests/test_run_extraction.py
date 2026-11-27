import json

import httpx
import pytest

from scripts import run_extraction as runner


@pytest.fixture
def run(tmp_path, monkeypatch):
    ids = ["0000000000000001", "0000000000000002"]
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text("".join(
        runner.dump({"input_id": iid, "title": "Title", "paragraph_text": iid})
        for iid in ids
    ), encoding="utf-8")
    sample = tmp_path / "sample.json"
    sample.write_text(runner.dump({"input_ids": ids[:1]}), encoding="utf-8")
    out = tmp_path / "run"
    monkeypatch.setattr(runner, "INPUTS", inputs)
    monkeypatch.setattr(runner, "_read_credentials", lambda _: ("https://example.invalid/v1", "test-key"))

    def invoke(*args):
        monkeypatch.setattr("sys.argv", [
            "run_extraction.py", "--model", "qwen", "--all", "--sample-first", str(sample),
            "--prompt", "baseline_fixed", "--out-dir", str(out), *args,
        ])
        runner.main()

    return out, ids, invoke


def response(finish="stop", *, model=runner.MODELS["qwen"], content='{"triplets":[]}', status=200):
    return httpx.Response(status, json={
        "model": model,
        "choices": [{"finish_reason": finish, "message": {"content": content}}],
        "usage": {"prompt_tokens": 10},
    })


def test_preview_does_not_read_credentials_or_write(run, monkeypatch, capsys):
    out, _, invoke = run
    def unexpected(*args, **kwargs):
        pytest.fail("Пробный план не должен использовать API или ключ")

    monkeypatch.setattr(runner, "_read_credentials", unexpected)
    monkeypatch.setattr(runner, "chat_response", unexpected)
    invoke()
    assert not out.exists()
    assert "2 абзацев × 1 промпта = 2 пар до повторов" in capsys.readouterr().out


def test_only_length_retries_and_resume_skips_completed(run, monkeypatch, capsys):
    out, ids, invoke = run
    calls = []
    replies = iter([response("length"), response(), response()])

    def post(model, messages, **kwargs):
        calls.append((model, messages, kwargs))
        return next(replies)

    monkeypatch.setattr(runner, "chat_response", post)
    invoke("--execute", "--limit", "1")
    assert [c[2]["max_tokens"] for c in calls] == [6000, 12000]
    assert calls[0][:2] == calls[1][:2]
    assert all(c[2]["temperature"] == 0.2 for c in calls)
    job = out / "jobs" / "baseline_fixed" / ids[0]
    old_request = (job / "attempt_1/request.json").read_bytes()
    old_result = (job / "attempt_1/result.json").read_bytes()
    assert json.loads(old_result)["status"] == "retry_length"
    invoke("--execute")
    assert [c[2]["max_tokens"] for c in calls] == [6000, 12000, 6000]
    assert (job / "attempt_1/request.json").read_bytes() == old_request
    assert (job / "attempt_1/result.json").read_bytes() == old_result
    answers = [json.loads(line) for line in (out / "answers.jsonl").read_bytes().splitlines()]
    assert [(r["input_id"], r["status"]) for r in answers] == [(iid, "ok") for iid in ids]
    assert answers[0]["parameters"]["max_tokens"] == 12000
    assert (out / answers[0]["source"]).is_file()
    assert b"test-key" not in (out / "config.json").read_bytes() + old_request + old_result
    assert "2/2 (100.0%)" in capsys.readouterr().err


def test_second_length_is_terminal(run, monkeypatch):
    out, _, invoke = run
    count = []

    def post(*args, **kwargs):
        count.append(kwargs["max_tokens"])
        return response("length")

    monkeypatch.setattr(runner, "chat_response", post)
    invoke("--execute", "--limit", "1")
    invoke("--execute")
    assert count == [6000, 12000, 6000, 12000]
    rows = (out / "answers.jsonl").read_bytes().splitlines()
    assert all(json.loads(row)["status"] == "truncated" for row in rows)


@pytest.mark.parametrize("failure", ["timeout", "http", "model"])
def test_failure_does_not_retry_even_on_resume(run, monkeypatch, failure):
    out, ids, invoke = run
    count = []

    def post(*args, **kwargs):
        count.append(kwargs["max_tokens"])
        if failure == "timeout":
            raise httpx.ReadTimeout("test-key")
        if failure == "http":
            return response(status=429)
        return response("length", model="wrong-model")

    monkeypatch.setattr(runner, "chat_response", post)
    with pytest.raises(SystemExit):
        invoke("--execute")
    with pytest.raises(SystemExit):
        invoke("--execute")
    assert count == [6000]
    body = (out / "jobs" / "baseline_fixed" / ids[0] / "attempt_1/result.json").read_bytes()
    assert b"test-key" not in body
    assert json.loads(body)["status"] == "halted"


def test_changed_configuration_is_rejected(run, monkeypatch):
    _, _, invoke = run
    monkeypatch.setattr(runner, "chat_response", lambda *args, **kwargs: response())
    invoke("--execute", "--limit", "1")
    monkeypatch.setattr(runner, "chat_response", lambda *args, **kwargs: pytest.fail("Новый POST"))
    with pytest.raises(SystemExit, match="Настройки запуска изменились"):
        invoke("--execute", "--temperature", "0.3")
