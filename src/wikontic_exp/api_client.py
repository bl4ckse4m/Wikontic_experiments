"""Async OpenAI-compatible API client with bounded retries.

Endpoint capabilities are *probed*, not assumed: usage and reasoning fields may
be absent. Retries cover transient network errors, timeouts, 429 and selected
5xx; auth/parameter errors fail fast. SDK double-retries are avoided by using
httpx directly (single retry budget, owned by tenacity here).
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import tenacity

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


class APIError(Exception):
    def __init__(self, message: str, status_code: int | None = None, permanent: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.permanent = permanent


class TransientAPIError(Exception):
    """Retryable: network/timeout/429/selected 5xx. Carries Retry-After if present."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


def _load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Minimal .env parser (KEY=VALUE or KEY: VALUE, keys case-insensitive).
    Values are never logged or printed."""
    out: dict[str, str] = {}
    p = Path(path).expanduser()
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(.*)", line)
        if match:
            k, v = match.groups()
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                v = v[1:-1]
            out[k.upper()] = v
    return out


def _first_env(names: tuple[str, ...], env: dict[str, str], dotenv: dict[str, str]) -> str | None:
    for n in names:
        if env.get(n):
            return env[n]
    for n in names:
        if n.upper() in dotenv:
            return dotenv[n.upper()]
    return None


@dataclass
class AccessConfig:
    api_base: str = field(repr=False)
    api_key: str = field(repr=False)
    model: str
    answerer_model: str | None = None
    timeout_s: float = 120.0
    max_concurrency: int = 8
    max_retries: int = 5
    provider: str = "primary"

    @property
    def routing_metadata(self) -> dict[str, str]:
        return {"provider": self.provider,
                "endpoint_hash": hashlib.sha256(self.api_base.encode()).hexdigest()}

    @property
    def cache_route(self) -> dict[str, str] | None:
        # Preserve legacy primary cache keys. Explicit OpenRouter never reuses them.
        return self.routing_metadata if self.provider != "primary" else None

    @classmethod
    def load(cls, path: str | Path | None = None,
             creds_file: str | Path | None = None,
             openrouter: bool = False) -> "AccessConfig":
        data: dict[str, Any] = {}
        if path is None:
            path = Path("configs/access.local.yaml")
        if yaml is not None and Path(path).exists():
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        models = data.get("models") or {}
        # Read only at runtime. Never source/execute a credentials file.
        env = {k.upper(): v for k, v in os.environ.items()}
        selected = creds_file or env.get("LLM_CREDS_FILE")
        external_path = Path(selected or "~/.config/wikontic/creds").expanduser()
        if selected and not external_path.is_file():
            raise ValueError("Configured credentials file does not exist")
        dotenv = {**_load_dotenv(), **_load_dotenv(external_path)}
        api_base = _first_env(("LLM_URL", "WIKONTIC_API_BASE", "API_BASE", "OPENAI_API_BASE", "OPENAI_BASE_URL"), env, {}) or _first_env(("LLM_URL", "WIKONTIC_API_BASE", "API_BASE", "OPENAI_API_BASE", "OPENAI_BASE_URL"), {}, dotenv) or data.get("api_base") or ""
        api_key = _first_env(("LLM_KEY", "WIKONTIC_API_KEY", "API_KEY", "OPENAI_API_KEY"), env, {}) or _first_env(("LLM_KEY", "WIKONTIC_API_KEY", "API_KEY", "OPENAI_API_KEY"), {}, dotenv) or data.get("api_key") or ""
        model = _first_env(("WIKONTIC_MODEL", "MODEL_NAME", "MODEL"), env, dotenv) or models.get("extractor") or ""
        answerer = _first_env(("WIKONTIC_ANSWERER_MODEL", "ANSWERER_MODEL"), env, dotenv) or models.get("answerer")
        if openrouter:
            # Explicit opt-in only. Never borrow the university/OpenAI key or model.
            api_base = _first_env(("OPENROUTER_URL", "OPENROUTER_BASE_URL"), env, dotenv) or "https://openrouter.ai/api/v1"
            api_key = _first_env(("OPENROUTER_API_KEY", "OPENROUTER_KEY"), env, dotenv) or ""
            model = _first_env(("OPENROUTER_MODEL",), env, dotenv) or "openai/gpt-oss-120b"
            answerer = _first_env(("OPENROUTER_ANSWERER_MODEL",), env, dotenv)
        return cls(
            api_base=api_base.rstrip("/"),
            api_key=api_key,
            model=model,
            answerer_model=answerer,
            timeout_s=float(_first_env(("WIKONTIC_TIMEOUT_S",), env, dotenv) or data.get("timeout_s", 120)),
            max_concurrency=int(_first_env(("WIKONTIC_MAX_CONCURRENCY",), env, dotenv) or data.get("max_concurrency", 8)),
            max_retries=int(_first_env(("WIKONTIC_MAX_RETRIES",), env, dotenv) or data.get("max_retries", 5)),
            provider="openrouter" if openrouter else "primary",
        )

    def redact(self, message: str) -> str:
        for value in (self.api_key, self.api_base):
            if value:
                message = message.replace(value, "[REDACTED]")
        return message

    def missing(self) -> list[str]:
        miss = []
        if not self.api_base or "<university-endpoint>" in self.api_base:
            miss.append("OPENROUTER_URL" if self.provider == "openrouter" else "llm_url in credentials file / WIKONTIC_API_BASE")
        if not self.api_key or self.api_key == "<key>":
            miss.append("openrouter_key / OPENROUTER_API_KEY" if self.provider == "openrouter" else "llm_key in credentials file / WIKONTIC_API_KEY")
        if not self.model or self.model.startswith("<"):
            miss.append("WIKONTIC_MODEL / models.extractor (exact model id from the access configuration)")
        return miss


@dataclass
class APICallResult:
    ok: bool
    raw_response: dict[str, Any] | None
    text: str | None
    usage: dict[str, Any] | None
    model: str
    latency_s: float
    attempts: int
    error: str | None = None
    error_kind: str | None = None  # transient_exhausted | permanent | http | timeout
    status_code: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _retryable_status(code: int) -> bool:
    return code in (408, 425, 429, 500, 502, 503, 504, 522, 524)


class OpenAICompatClient:
    def __init__(self, cfg: AccessConfig):
        self.cfg = cfg
        self._sem = asyncio.Semaphore(max(1, cfg.max_concurrency))
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "OpenAICompatClient":
        # httpx retries disabled: the retry budget is tenacity's alone
        self._client = httpx.AsyncClient(
            base_url=self.cfg.api_base,
            timeout=httpx.Timeout(self.cfg.timeout_s),
            headers={"Authorization": f"Bearer {self.cfg.api_key}"},
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def probe(self) -> dict[str, Any]:
        """Check endpoint capabilities without assuming OpenAI optionality.
        Never prints api_base/api_key values."""
        assert self._client is not None
        info: dict[str, Any] = {"api_base_configured": bool(self.cfg.api_base),
                                "api_key_configured": bool(self.cfg.api_key),
                                "provider": self.cfg.provider,
                                "model": self.cfg.model}
        try:
            r = await self._client.get("/models", timeout=20.0)
            info["models_status"] = r.status_code
            if r.status_code == 200:
                try:
                    data = r.json()
                    ids = [m.get("id") for m in data.get("data", [])] if isinstance(data, dict) else []
                    info["models"] = ids
                    info["model_listed"] = self.cfg.model in ids
                except Exception as e:  # noqa: BLE001
                    info["models_parse_error"] = self.cfg.redact(str(e))
        except Exception as e:  # noqa: BLE001
            info["models_error"] = self.cfg.redact(str(e))
        try:
            res = await self.chat(
                model=self.cfg.model,
                messages=[{"role": "user", "content": "Reply with the single word: ok"}],
                gen_params={"temperature": 0, "max_tokens": 512},
                gen_index=0,
                task_id="probe",
            )
            info["chat_ok"] = res.ok
            info["chat_error"] = res.error
            info["chat_has_usage"] = res.usage is not None
            info["chat_text"] = (res.text or "")[:80]
            info["chat_raw_keys"] = sorted(res.raw_response.keys()) if res.raw_response else None
        except Exception as e:  # noqa: BLE001
            info["chat_exception"] = self.cfg.redact(str(e))
        return info

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        gen_params: dict[str, Any],
        gen_index: int,
        task_id: str,
    ) -> APICallResult:
        assert self._client is not None, "use as async context manager"
        payload: dict[str, Any] = {"model": model, "messages": messages}
        payload.update(gen_params or {})
        started = time.monotonic()
        attempts_holder = {"n": 0}

        async def _attempt() -> dict[str, Any]:
            attempts_holder["n"] += 1
            async with self._sem:  # held only for the request, released during backoff
                try:
                    r = await self._client.post("/chat/completions", json=payload)
                except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as e:
                    raise TransientAPIError(f"transient network: {e}") from e
                if _retryable_status(r.status_code):
                    retry_after = None
                    if r.status_code == 429:
                        ra = r.headers.get("Retry-After")
                        if ra is not None:
                            try:
                                retry_after = min(float(ra), 60.0)
                            except ValueError:
                                retry_after = None
                    raise TransientAPIError(f"retryable http {r.status_code}: {r.text[:200]}", retry_after=retry_after)
                if r.status_code >= 400:
                    raise APIError(f"http {r.status_code}: {r.text[:400]}", status_code=r.status_code, permanent=True)
                try:
                    return r.json()
                except Exception as e:  # noqa: BLE001
                    raise APIError(f"non-JSON response: {e}", status_code=r.status_code, permanent=True) from e

        wait_exp = tenacity.wait_exponential_jitter(initial=1, max=30, jitter=2)

        async def _wait(retry_state: tenacity.RetryCallState) -> float:
            outcome = retry_state.outcome
            delay = wait_exp(retry_state)
            exc = outcome.exception() if outcome is not None else None
            if isinstance(exc, TransientAPIError) and exc.retry_after is not None:
                delay = max(delay, exc.retry_after)
            return delay

        retryer = tenacity.AsyncRetrying(
            stop=tenacity.stop_after_attempt(self.cfg.max_retries),
            wait=_wait,
            retry=tenacity.retry_if_exception_type(TransientAPIError),
            reraise=True,
        )
        try:
            async for attempt in retryer:
                with attempt:
                    data = await _attempt()
        except APIError as e:
            return APICallResult(
                ok=False, raw_response=None, text=None, usage=None, model=model,
                latency_s=time.monotonic() - started, attempts=attempts_holder["n"],
                error=self.cfg.redact(str(e)), error_kind="permanent", status_code=e.status_code,
            )
        except TransientAPIError as e:
            return APICallResult(
                ok=False, raw_response=None, text=None, usage=None, model=model,
                latency_s=time.monotonic() - started, attempts=attempts_holder["n"],
                error=self.cfg.redact(str(e)), error_kind="transient_exhausted",
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            return APICallResult(
                ok=False, raw_response=None, text=None, usage=None, model=model,
                latency_s=time.monotonic() - started, attempts=attempts_holder["n"],
                error=self.cfg.redact(str(e)), error_kind="unknown",
            )

        text = None
        usage = None
        try:
            choice = data["choices"][0]
            msg = choice.get("message") or {}
            text = msg.get("content")
            if text is None:
                text = choice.get("text")
        except Exception as e:  # noqa: BLE001
            return APICallResult(
                ok=False, raw_response=data, text=None, usage=None, model=model,
                latency_s=time.monotonic() - started, attempts=attempts_holder["n"],
                error=f"response shape error: {e}", error_kind="http",
            )
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        return APICallResult(
            ok=True, raw_response=data, text=text, usage=usage, model=model,
            latency_s=time.monotonic() - started, attempts=attempts_holder["n"],
        )
