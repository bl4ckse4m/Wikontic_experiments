"""Извлечение из одного абзаца с форматами запросов и разбора прошлых прогонов."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .api_client import chat


ParseMode = Literal["legacy", "pilot"]
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|```$", re.MULTILINE)
_CORE_FIELDS = ("subject", "relation", "object")
_TYPE_FIELDS = ("subject_type", "object_type")
_PILOT_FIELDS = {*_CORE_FIELDS, "qualifiers", *_TYPE_FIELDS}


@dataclass
class ExtractionResult:
    raw_text: str
    triplets: list[dict[str, Any]]


class ExtractionParseError(ValueError):
    """Ответ получен, но не принят парсером. Оригинал доступен в raw_text."""

    def __init__(self, message: str, raw_text: str):
        super().__init__(message)
        self.raw_text = raw_text


def render_messages(prompt_text: str, title: str, paragraph_text: str) -> list[dict[str, str]]:
    """Extraction request: title + paragraph only. No question/answer/decomposition."""
    user = f"### Input:\n  Text: \"{title}\\n\\n{paragraph_text}\"\n### Output:"
    return [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": user},
    ]


def _legacy_json_object(raw_text: str) -> dict:
    # 0974f43: schema.py — кодовый блок или первый JSON-объект после вводного текста.
    text = _FENCE_RE.sub("", raw_text).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start == -1:
            raise
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("Верхний уровень JSON должен быть объектом")
    return obj


def _as_str(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field}: ожидалась строка")
    return value


def _legacy_triplet(fact: Any, index: int) -> dict:
    # Тот же набор полей, что у parse_extraction(...).to_target_dict().
    field = f"triplets[{index}]"
    if not isinstance(fact, dict):
        raise ValueError(f"{field}: ожидался объект")
    out = {key: _as_str(fact.get(key), f"{field}.{key}") for key in _CORE_FIELDS}

    qualifiers = fact.get("qualifiers")
    if qualifiers is None:
        qualifiers = []
    if not isinstance(qualifiers, list):
        raise ValueError(f"{field}.qualifiers: ожидался список")
    out["qualifiers"] = []
    for i, qualifier in enumerate(qualifiers):
        group = f"{field}.qualifiers[{i}]"
        if not isinstance(qualifier, dict):
            raise ValueError(f"{group}: ожидался объект")
        out["qualifiers"].append({
            key: _as_str(qualifier.get(key), f"{group}.{key}")
            for key in ("relation", "object")
        })

    claim_ids = fact.get("claim_ids")
    if claim_ids is not None and (
        not isinstance(claim_ids, list) or not all(isinstance(i, int) for i in claim_ids)
    ):
        raise ValueError(f"{field}.claim_ids: ожидался список целых чисел")
    for key in _TYPE_FIELDS:
        if fact.get(key) is not None:
            out[key] = _as_str(fact[key], f"{field}.{key}")
    return out


def _strict_json(raw_text: str) -> Any:
    # Пилот gpt-oss/Qwen: e11_journal.strict_json.
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Повторяющийся ключ JSON")
            result[key] = value
        return result

    def constant(_value):
        raise ValueError("Константа не входит в JSON")

    return json.loads(raw_text, object_pairs_hook=pairs, parse_constant=constant)


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _pilot_triplets(raw_text: str) -> list[dict[str, Any]]:
    # e11_experiment.extraction_answer и проверка e11_measurement.claim_payload.
    obj = _strict_json(raw_text)
    if not isinstance(obj, dict) or set(obj) != {"triplets"} or not isinstance(obj["triplets"], list):
        raise ValueError("Ожидался объект с единственным полем triplets (список)")
    for i, fact in enumerate(obj["triplets"]):
        field = f"triplets[{i}]"
        if (
            not isinstance(fact, dict)
            or set(fact) - _PILOT_FIELDS
            or not all(_nonempty_text(fact.get(key)) for key in _CORE_FIELDS)
            or not isinstance(fact.get("qualifiers"), list)
            or any(not isinstance(fact[key], str) for key in _TYPE_FIELDS if key in fact)
        ):
            raise ValueError(f"{field}: неверные поля триплета")
        for j, qualifier in enumerate(fact["qualifiers"]):
            if (
                not isinstance(qualifier, dict)
                or set(qualifier) != {"relation", "object"}
                or not all(_nonempty_text(value) for value in qualifier.values())
            ):
                raise ValueError(f"{field}.qualifiers[{j}]: неверные поля уточнения")
    return obj["triplets"]


def _check_mode(parse_mode: ParseMode) -> None:
    if parse_mode not in ("legacy", "pilot"):
        raise ValueError("parse_mode должен быть 'legacy' или 'pilot'")


def parse_triplets(raw_text: str, *, parse_mode: ParseMode) -> list[dict[str, Any]]:
    """Разобрать сохранённый текст без API: legacy для сравнения промптов, pilot для пары моделей."""
    _check_mode(parse_mode)
    try:
        if parse_mode == "pilot":
            return _pilot_triplets(raw_text)
        obj = _legacy_json_object(raw_text)
        triplets = obj.get("triplets")
        if not isinstance(triplets, list):
            raise ValueError("Поле triplets отсутствует или не является списком")
        return [_legacy_triplet(fact, i) for i, fact in enumerate(triplets)]
    except ValueError as exc:
        raise ExtractionParseError(str(exc), raw_text) from exc


def extract_triplets(
    title: str,
    paragraph_text: str,
    *,
    prompt_path: str | Path,
    model: str,
    base_url: str,
    api_key: str,
    parse_mode: ParseMode,
    temperature: float = 0.2,
    max_tokens: int = 6000,
    timeout: float = 120,
    **parameters: Any,
) -> ExtractionResult:
    """Один вызов модели; никаких записей на диск и автоматических повторов."""
    _check_mode(parse_mode)
    if not base_url or not api_key:
        raise ValueError("Передайте адрес и ключ одного сервиса")
    prompt_text = Path(prompt_path).read_text(encoding="utf-8")
    raw_text = chat(
        model,
        render_messages(prompt_text, title, paragraph_text),
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
        **parameters,
    )
    return ExtractionResult(raw_text, parse_triplets(raw_text, parse_mode=parse_mode))
