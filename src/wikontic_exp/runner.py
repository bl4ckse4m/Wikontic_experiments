"""Extraction runner: one independent single-pass extractor per call.

Every task is stored immediately (resume-safe). The cache key includes the
*independent generation* index so intentional ensemble repetitions do not
collapse, while retries of one task do not create new entries. Raw responses
are kept before parsing; parse/schema failures are first-class statuses.
``--mock`` runs the whole pipeline deterministically and marks every record as
mock: it verifies plumbing only and is never an experimental result.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import __version__
from .api_client import AccessConfig, APICallResult, OpenAICompatClient
from .hashing import canonical_json, extraction_cache_key, hash_obj, sha256_hex
from .schema import SCHEMA_ID, SchemaError, parse_extraction


@dataclass
class MethodConfig:
    name: str
    prompt_path: str
    model: str
    gen_params: dict[str, Any]
    n_generations: int = 1
    schema_id: str = SCHEMA_ID
    description: str = ""

    @property
    def prompt_text(self) -> str:
        return Path(self.prompt_path).read_text(encoding="utf-8")

    @property
    def prompt_hash(self) -> str:
        return sha256_hex(self.prompt_text)

    @classmethod
    def load(cls, path: str | Path, model_default: str = "") -> "MethodConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=data["name"],
            prompt_path=data["prompt"],
            model=data.get("model") or model_default,
            gen_params=data.get("gen_params") or {},
            n_generations=int(data.get("n_generations", 1)),
            schema_id=data.get("schema_id", SCHEMA_ID),
            description=data.get("description", ""),
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "prompt_path": self.prompt_path,
            "prompt_sha256": self.prompt_hash,
            "model": self.model,
            "gen_params": self.gen_params,
            "n_generations": self.n_generations,
            "schema_id": self.schema_id,
            "description": self.description,
            "wikontic_exp_version": __version__,
        }


@dataclass
class TaskSpec:
    input_id: str
    title: str
    paragraph_text: str
    gen_index: int

    @property
    def input_hash(self) -> str:
        return sha256_hex(canonical_json({"title": self.title, "paragraph_text": self.paragraph_text}))


def render_messages(prompt_text: str, title: str, paragraph_text: str) -> list[dict[str, str]]:
    """Extraction request: title + paragraph only. No question/answer/decomposition."""
    user = f"### Input:\n  Text: \"{title}\\n\\n{paragraph_text}\"\n### Output:"
    return [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": user},
    ]


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class TaskRecord:
    task: dict[str, Any]
    status: str
    raw_response: Any
    raw_text: str | None
    parsed: dict[str, Any] | None
    service: dict[str, Any]
    usage: dict[str, Any] | None
    token_estimate: dict[str, int]
    attempts: int
    latency_s: float
    error: str | None
    error_kind: str | None
    mock: bool
    created_utc: str
    cache_key: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


class MockClient:
    """Deterministic offline double. Produces schema-valid and (by hash) invalid
    outputs so failure paths are exercised. Never a substitute for experiments."""

    def __init__(self, cfg: AccessConfig):
        self.cfg = cfg

    async def __aenter__(self) -> "MockClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def chat(self, model: str, messages: list[dict[str, str]], gen_params: dict[str, Any],
                   gen_index: int, task_id: str) -> APICallResult:
        t0 = time.monotonic()
        seed = hashlib.sha256(f"{task_id}|{gen_index}".encode()).hexdigest()
        if seed[0] == "f":  # deterministic slice of invalid JSON for failure-path testing
            text = '{"triplets": [{"subject": "broken"'
            return APICallResult(ok=True, raw_response={"mock": True, "choices": [{"message": {"content": text}}]},
                                 text=text, usage=None, model=model, latency_s=time.monotonic() - t0,
                                 attempts=1, extra={"mock": True})
        if seed[0] == "e":  # deterministic permanent API error path
            return APICallResult(ok=False, raw_response=None, text=None, usage=None, model=model,
                                 latency_s=time.monotonic() - t0, attempts=1,
                                 error="mock permanent error", error_kind="permanent", status_code=400)
        user = messages[1]["content"]
        title = user.split("Text: \"", 1)[1].split("\n", 1)[0][:40]
        payload = {
            "triplets": [
                {"subject": title or "subject", "relation": "mock relation",
                 "object": f"mock object {seed[:6]}",
                 "qualifiers": [{"relation": "point in time", "object": "2020"}],
                 "subject_type": "mock", "object_type": "mock"}
            ]
        }
        text = json.dumps(payload)
        return APICallResult(ok=True,
                             raw_response={"mock": True, "choices": [{"message": {"content": text}}]},
                             text=text,
                             usage={"prompt_tokens": estimate_tokens(user), "completion_tokens": estimate_tokens(text)},
                             model=model, latency_s=time.monotonic() - t0, attempts=1,
                             extra={"mock": True})


def load_inputs(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
            if limit is not None and len(out) >= limit:
                break
    return out


class Runner:
    def __init__(self, method: MethodConfig, run_dir: Path, access: AccessConfig | None, mock: bool = False):
        self.method = method
        self.run_dir = Path(run_dir)
        self.access = access
        self.mock = mock
        self.responses_dir = self.run_dir / "responses"
        self.responses_dir.mkdir(parents=True, exist_ok=True)

    def task_key(self, task: TaskSpec) -> str:
        return extraction_cache_key(
            input_hash=task.input_hash,
            model=self.method.model,
            prompt_hash=self.method.prompt_hash,
            schema_id=self.method.schema_id,
            gen_params=self.method.gen_params,
            gen_index=task.gen_index,
            route=self.access.cache_route if self.access and not self.mock else None,
        )

    def response_path(self, cache_key: str) -> Path:
        return self.responses_dir / f"{cache_key}.json"

    def existing(self, cache_key: str) -> TaskRecord | None:
        p = self.response_path(cache_key)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            data.pop("_extra", None)
            return TaskRecord(**data)
        except Exception:  # noqa: BLE001  # corrupted file -> recompute
            return None

    def write_run_config(self, inputs_path: str, extra: dict[str, Any]) -> None:
        cfg = {
            "run_dir": str(self.run_dir),
            "method": self.method.snapshot(),
            "inputs_path": str(inputs_path),
            "mock": self.mock,
            "access_model": self.method.model,
            "access_route": self.access.routing_metadata if self.access and not self.mock else {"provider": "mock"},
            "created_utc": datetime.now(timezone.utc).isoformat(),
            **extra,
        }
        (self.run_dir / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    async def run(self, inputs: Iterable[dict[str, Any]], resume: bool = True,
                  retry_failures: bool = False) -> dict[str, Any]:
        tasks: list[TaskSpec] = []
        for rec in inputs:
            for g in range(self.method.n_generations):
                tasks.append(TaskSpec(input_id=rec["input_id"], title=rec["title"],
                                      paragraph_text=rec["paragraph_text"], gen_index=g))

        stats = {"total": len(tasks), "cached": 0, "computed": 0, "failed": 0}
        index_path = self.run_dir / "index.jsonl"
        sem_stats: dict[str, int] = {}

        client: Any
        if self.mock:
            client = MockClient(self.access or AccessConfig(api_base="mock", api_key="mock", model=self.method.model))
        else:
            assert self.access is not None, "access config required unless --mock"
            client = OpenAICompatClient(self.access)

        pending: list[TaskSpec] = []
        for t in tasks:
            key = self.task_key(t)
            rec = self.existing(key) if resume else None
            if rec is not None and (rec.status == "ok" or not retry_failures):
                stats["cached"] += 1
                continue
            pending.append(t)

        async with client:
            async def _one(t: TaskSpec) -> TaskRecord:
                key = self.task_key(t)
                messages = render_messages(self.method.prompt_text, t.title, t.paragraph_text)
                res = await client.chat(
                    model=self.method.model, messages=messages, gen_params=self.method.gen_params,
                    gen_index=t.gen_index, task_id=key,
                )
                rec = self._build_record(t, key, res)
                self._save(rec)
                return rec

            for chunk_start in range(0, len(pending), max(1, self.access.max_concurrency if self.access else 8)):
                chunk = pending[chunk_start:chunk_start + (self.access.max_concurrency if self.access else 8)]
                results = await asyncio.gather(*[_one(t) for t in chunk])
                for rec in results:
                    sem_stats[rec.status] = sem_stats.get(rec.status, 0) + 1
                    if rec.status != "ok":
                        stats["failed"] += 1
                    else:
                        stats["computed"] += 1
                    with open(index_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "cache_key": rec.cache_key, "input_id": rec.task["input_id"],
                            "gen_index": rec.task["gen_index"], "status": rec.status,
                        }, ensure_ascii=False) + "\n")
        stats["status_counts"] = sem_stats
        (self.run_dir / "status_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        return stats

    def _build_record(self, t: TaskSpec, key: str, res: APICallResult) -> TaskRecord:
        task_meta = {
            "input_id": t.input_id, "input_hash": t.input_hash, "gen_index": t.gen_index,
            "model": self.method.model, "prompt_hash": self.method.prompt_hash,
            "prompt_path": self.method.prompt_path, "schema_id": self.method.schema_id,
            "gen_params": self.method.gen_params,
        }
        task_meta["route"] = self.access.routing_metadata if self.access and not self.mock else {"provider": "mock"}
        usage = res.usage
        est = {
            "input_tokens_est": estimate_tokens(canonical_json(task_meta) + (res.text or "")),
            "output_tokens_est": estimate_tokens(res.text or ""),
        }
        if not res.ok:
            status = "api_error" if res.error_kind != "permanent" else "api_permanent_error"
            return TaskRecord(task=task_meta, status=status, raw_response=res.raw_response, raw_text=res.text,
                              parsed=None, service={}, usage=usage, token_estimate=est, attempts=res.attempts,
                              latency_s=res.latency_s, error=res.error, error_kind=res.error_kind,
                              mock=self.mock, created_utc=datetime.now(timezone.utc).isoformat(), cache_key=key)
        raw_text = res.text or ""
        finish_reason = None
        try:
            finish_reason = res.raw_response["choices"][0].get("finish_reason")
        except Exception:  # noqa: BLE001
            pass
        try:
            parsed = parse_extraction(raw_text)
        except SchemaError as e:
            status = "truncated" if finish_reason == "length" else ("invalid_json" if "JSON" in str(e) or "json" in str(e) else "schema_error")
            return TaskRecord(task=task_meta, status=status, raw_response=res.raw_response, raw_text=raw_text,
                              parsed=None, service={}, usage=usage, token_estimate=est, attempts=res.attempts,
                              latency_s=res.latency_s, error=str(e), error_kind="parse", mock=self.mock,
                              created_utc=datetime.now(timezone.utc).isoformat(), cache_key=key)
        return TaskRecord(task=task_meta, status="ok", raw_response=res.raw_response, raw_text=raw_text,
                          parsed=parsed.to_target_dict(), service=parsed.service, usage=usage,
                          token_estimate=est, attempts=res.attempts, latency_s=res.latency_s,
                          error=None, error_kind=None, mock=self.mock,
                          created_utc=datetime.now(timezone.utc).isoformat(), cache_key=key)

    def _save(self, rec: TaskRecord) -> None:
        tmp = self.responses_dir / f".{rec.cache_key}.tmp"
        tmp.write_text(rec.to_json(), encoding="utf-8")
        tmp.replace(self.response_path(rec.cache_key))  # atomic within the filesystem


def aggregate_usage(run_dir: Path) -> dict[str, Any]:
    total = {"calls_ok": 0, "calls_failed": 0, "attempts": 0, "latency_s": 0.0,
             "prompt_tokens": 0, "completion_tokens": 0, "prompt_tokens_est": 0,
             "completion_tokens_est": 0, "usage_present": 0}
    for p in sorted((run_dir / "responses").glob("*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        total["attempts"] += rec.get("attempts", 0)
        total["latency_s"] += rec.get("latency_s", 0.0)
        if rec.get("status") == "ok":
            total["calls_ok"] += 1
        else:
            total["calls_failed"] += 1
        u = rec.get("usage")
        if u:
            total["usage_present"] += 1
            total["prompt_tokens"] += u.get("prompt_tokens", 0) or 0
            total["completion_tokens"] += u.get("completion_tokens", 0) or 0
        te = rec.get("token_estimate") or {}
        total["prompt_tokens_est"] += te.get("input_tokens_est", 0)
        total["completion_tokens_est"] += te.get("output_tokens_est", 0)
    return total
