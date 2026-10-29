"""Canonical hashing used for input identity and cache keys."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json(obj))


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def input_id(title: str, paragraph_text: str) -> str:
    """Stable identifier of an extraction input (title + paragraph)."""
    return hashlib.sha256((title + "\x00" + paragraph_text).encode("utf-8")).hexdigest()[:16]


def extraction_cache_key(
    input_hash: str,
    model: str,
    prompt_hash: str,
    schema_id: str,
    gen_params: dict[str, Any],
    gen_index: int,
    route: dict[str, str] | None = None,
) -> str:
    """Cache key of one extraction task.

    ``gen_index`` identifies an *independent generation* of the ensemble: two
    intentional repeated generations of the same task must not collapse into one
    cache entry. A retry of the same task reuses the same key.
    """
    payload = {
        "input_hash": input_hash,
        "model": model,
        "prompt_hash": prompt_hash,
        "schema_id": schema_id,
        "gen_params": gen_params,
        "gen_index": gen_index,
    }
    if route is not None:
        payload["route"] = route
    return hash_obj(payload)[:32]


def merger_cache_key(input_response_hashes: list[str], algorithm: str, algorithm_version: str, params: dict[str, Any]) -> str:
    payload = {
        "input_response_hashes": sorted(input_response_hashes),
        "algorithm": algorithm,
        "algorithm_version": algorithm_version,
        "params": params,
    }
    return hash_obj(payload)[:32]
