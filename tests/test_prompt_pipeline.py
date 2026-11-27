import json

import httpx
import pytest

from scripts import run_dedup, run_extraction, run_judging, score_judgments
from wikontic_exp import experiment_io, recorded_calls
from wikontic_exp.dedup import merge_triplets
from wikontic_exp.judging import parse_judge_labels
from wikontic_exp.scoring import summarize


G = run_extraction.MODELS["gpt-oss"]
B = "Institute/New-Model"
FACT = {"subject": "Анна", "relation": "открыла", "object": "музей", "qualifiers": []}
FALSE_FACT = {**FACT, "object": "стадион"}
INSTITUTE = "https://institute.invalid/v1"


def invoke(module, monkeypatch, *args):
    monkeypatch.setattr("sys.argv", [module.__name__, *map(str, args)])
    module.main()


def response(model, answer, *, fence=False):
    text = json.dumps(answer, ensure_ascii=False)
    return httpx.Response(200, json={
        "model": model, "id": "fake-generation", "usage": {"cost": 0.001},
        "choices": [{"finish_reason": "stop", "message": {
            "content": f"```json\n{text}\n```" if fence else text,
        }}],
    })


def test_extraction_to_prompt_and_ensemble_tables(tmp_path, monkeypatch):
    ids = ["0000000000000001", "0000000000000002"]
    source = tmp_path / "inputs.jsonl"
    source.write_text("".join(run_extraction.dump({
        "input_id": i, "title": i, "paragraph_text": "Анна открыла музей.",
    }) for i in ids), encoding="utf-8")
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps({"input_ids": ids}), encoding="utf-8")
    monkeypatch.setattr(run_extraction, "INPUTS", source)
    monkeypatch.setattr(experiment_io, "INPUTS", source)
    for m in (run_extraction, run_dedup):
        monkeypatch.setattr(m, "_read_credentials", lambda _: (INSTITUTE, "institute-test-key"))
    monkeypatch.setattr(run_judging, "read_openrouter_key", lambda _: "claude-test-key")
    extraction_calls = []

    def extract(model, messages, **kwargs):
        assert kwargs["base_url"] == INSTITUTE
        extraction_calls.append(model)
        correct_prompt = messages[0]["content"] == (
            run_extraction.ROOT / run_extraction.PROMPTS["baseline_fixed"]
        ).read_text(encoding="utf-8")
        return response(model, {"triplets": [FACT] if correct_prompt else [FACT, FALSE_FACT]})

    monkeypatch.setattr(run_extraction, "chat_response", extract)
    files = {}
    for model, flag, value in ((G, "--model", "gpt-oss"), (B, "--model-id", B)):
        out = tmp_path / value.replace("/", "_")
        invoke(run_extraction, monkeypatch, flag, value, "--sample", sample, "--out-dir", out, "--execute")
        files[model] = out / "answers.jsonl"
    assert extraction_calls == [G] * 8 + [B] * 8

    grounds, dedups = [], []

    def post(model, messages, **kwargs):
        user = json.loads(messages[1]["content"])
        assert set(user) == {"title", "paragraph_text", "candidates"}
        if model.startswith("anthropic/"):
            assert kwargs["base_url"] == run_judging.JUDGE_URL and kwargs["api_key"] == "claude-test-key"
            grounds.append(user)
            labels = [{"id": r["id"], "label": int(r["triplet"] == FACT)} for r in user["candidates"]]
            return response(model, {"labels": labels}, fence=True)
        assert model in (G, B) and kwargs["base_url"] == INSTITUTE
        dedups.append((model, messages))
        assert user["candidates"] == [{"id": 0, "triplet": FACT}, {"id": 1, "triplet": FACT}]
        return response(model, {"groups": [{"members": [0, 1], "representative": 1}]}, fence=True)

    monkeypatch.setattr(recorded_calls, "chat_response", post)
    for i, model in enumerate((G, B)):
        out = tmp_path / f"judgments_{i}.jsonl"
        args = ("--sample", sample, "--answers", files[model], "--out", out)
        invoke(run_judging, monkeypatch, *args)
        assert not out.exists()
        n = len(grounds)
        invoke(run_judging, monkeypatch, *args, "--execute", "--limit", "2")
        assert len(grounds) == n + 2
        invoke(run_judging, monkeypatch, *args, "--execute")
        invoke(run_judging, monkeypatch, *args, "--execute")
        assert len(grounds) == n + 8
        one = experiment_io.read_jsonl(out)[0]
        assert one["usage"]["cost"] == 0.001 and one["raw_text"].startswith("```json")
        assert "response_body" in json.loads((out.parent / one["result_file"]).read_bytes())
        report = tmp_path / f"score_{i}.json"
        invoke(score_judgments, monkeypatch, "--sample", sample, "--answers", files[model],
               "--judgments", out, "--out", report)
        stats = json.loads(report.read_bytes())
        assert stats["complete"]
        assert stats["methods"]["baseline_fixed"]["precision_common"] == 1
        assert stats["methods"]["coverage"]["precision_common"] == 0.5

    combined = []
    for model, label in ((G, "merge_g"), (B, "merge_b")):
        out = tmp_path / f"{label}.jsonl"
        invoke(run_dedup, monkeypatch, "--sample", sample, "--left", files[G],
               "--left-method", "baseline_fixed", "--right", files[B], "--right-method", "baseline_fixed",
               "--model-id", model, "--method", label, "--out", out, "--execute")
        combined.append(("--entry", label, label, out))
        for row in experiment_io.read_jsonl(out):
            assert row["triplets"] == [FACT]
            assert row["groups"] == [{"members": [0, 1], "representative": 1}]
    assert dedups[0][1] == dedups[2][1] and dedups[1][1] == dedups[3][1]

    entries = ("--entry", "single_g", "baseline_fixed", files[G],
               "--entry", "single_b", "baseline_fixed", files[B], *combined[0], *combined[1])
    judgments = tmp_path / "ensemble_judgments.jsonl"
    report = tmp_path / "ensemble_score.json"
    invoke(run_judging, monkeypatch, "--sample", sample, *entries, "--out", judgments, "--execute")
    invoke(score_judgments, monkeypatch, "--sample", sample, *entries, "--judgments", judgments, "--out", report)
    result = json.loads(report.read_bytes())
    assert result["complete"] and set(result["methods"]) == {"single_g", "single_b", "merge_g", "merge_b"}


def test_timeout_and_incomplete_attempt_are_not_resent(tmp_path, monkeypatch):
    out = tmp_path / "judgments.jsonl"
    body = {"model": "judge", "messages": []}
    tasks = [{"body": body, "metadata": {"input_id": "a", "method": "x"}}]
    calls = []

    def timeout(*args, **kwargs):
        calls.append(1)
        raise httpx.ReadTimeout("test-key")

    monkeypatch.setattr(recorded_calls, "chat_response", timeout)
    kwargs = dict(stage="test", base_url=INSTITUTE, api_key="test-key", timeout=120,
                  limit=None, parse=None, version="test")
    with pytest.raises(SystemExit, match="ReadTimeout"):
        recorded_calls.run_recorded(out, tasks, **kwargs)
    with pytest.raises(SystemExit, match="повтора нет"):
        recorded_calls.run_recorded(out, tasks, **kwargs)
    (tmp_path / "judgments.jsonl.jobs/000001/result.json").unlink()
    with pytest.raises(SystemExit, match="Незавершённая попытка"):
        recorded_calls.run_recorded(out, tasks, **kwargs)
    assert calls == [1] and b"test-key" not in out.read_bytes()


def test_partial_scores_and_invalid_labels():
    answers = [dict(input_id=i, method=m, status="ok", model=G, triplets=[FACT])
               for i in ("a", "b") for m in ("baseline_fixed", "coverage")]
    judged = [dict(a, judge_model="claude", extraction_status="ok", labels=[1])
              for a in answers[:3]]
    result = summarize(answers, judged, baseline="baseline_fixed")
    assert not result["complete"]
    assert result["common_input_ids"] == ["a"]
    assert result["methods"]["coverage"]["counts"] == {"judged": 1, "not_judged": 1}
    with pytest.raises(ValueError, match="одна метка"):
        summarize(answers, [dict(judged[0], labels=[])], baseline="baseline_fixed")


@pytest.mark.parametrize("raw", [
    '{"labels":[]}', '{"labels":[{"id":1,"label":true}]}',
    '{"labels":[{"id":1,"label":1},{"id":1,"label":0}]}',
    'prose\n```json\n{"labels":[{"id":1,"label":1}]}\n```',
])
def test_bad_labels_are_not_repaired(raw):
    with pytest.raises(ValueError):
        parse_judge_labels(raw, 1)


@pytest.mark.parametrize("raw", [
    '{"keep_ids":[0,1]}',
    '{"groups":[]}',
    '{"groups":[{"members":[0],"representative":0}]}',
    '{"groups":[{"members":[0,2],"representative":0}]}',
    '{"groups":[{"members":[0,0,1],"representative":0}]}',
    '{"groups":[{"members":[0,1],"representative":0},{"members":[1],"representative":1}]}',
    '{"groups":[{"members":[0],"representative":1},{"members":[1],"representative":1}]}',
    '{"groups":[{"members":[true,0],"representative":0}]}',
    '{"groups":[{"members":[0,1],"representative":true}]}',
    '{"groups":[{"members":[0,1],"representative":1,"triplet":{}}]}',
    '{"groups":[{"members":[0,1],"representative":0,"representative":1}]}',
])
def test_dedup_rejects_invalid_partition(raw):
    with pytest.raises(ValueError):
        merge_triplets(raw, [FACT, FALSE_FACT])


def test_dedup_keeps_representatives_without_rewriting():
    paraphrase = {**FACT, "subject": "музей", "relation": "был открыт", "object": "Анна"}
    raw = '{"groups":[{"members":[1],"representative":1},{"members":[2,0],"representative":2}]}'
    groups, triplets = merge_triplets(raw, [FACT, FALSE_FACT, paraphrase])
    assert groups == [{"members": [0, 2], "representative": 2},
                      {"members": [1], "representative": 1}]
    assert triplets[0] is paraphrase and triplets[1] is FALSE_FACT


def test_dedup_distinguishes_missing_and_empty_extraction():
    left = [{"input_id": "a", "model": G, "method": "baseline_fixed", "status": "ok",
             "source": None, "triplets": []}]
    right = [{**left[0], "model": B, "status": "missing", "triplets": None}]
    params = dict(model=G, method="merge_g", prompt="test", temperature=0.2, max_tokens=6000)
    task, = run_dedup.build_tasks(left, right, {}, **params)
    assert task["body"] is None and task["without_call"]["status"] == "no_extraction"
    task, = run_dedup.build_tasks(left, [{**right[0], "status": "ok", "triplets": []}], {}, **params)
    assert task["body"] is None
    assert task["without_call"] == {"status": "ok", "triplets": [], "groups": []}
