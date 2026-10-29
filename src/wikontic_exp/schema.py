"""Fact representation contract: wikontic_facts_v1.

Contract (see schemas/facts_v1.json):
    {"triplets": [ {"subject","relation","object","qualifiers":[{relation,object}],
                    "subject_type"?,"object_type"?,"claim_ids"?} ] }

``claim_ids`` is a *service* field (evidence linkage for the claimwise variant);
it is stored separately from the student target. ``subject_type``/``object_type``
are kept for comparability with Wikontic.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

SCHEMA_ID = "wikontic_facts_v1"

_SERVICE_FIELDS = ("claim_ids",)


class SchemaError(ValueError):
    pass


@dataclass
class Qualifier:
    relation: str
    object: str

    def to_dict(self) -> dict[str, str]:
        return {"relation": self.relation, "object": self.object}


@dataclass
class Fact:
    subject: str
    relation: str
    object: str
    qualifiers: list[Qualifier] = field(default_factory=list)
    subject_type: str | None = None
    object_type: str | None = None
    claim_ids: list[int] | None = None

    def to_dict(self, include_service: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "qualifiers": [q.to_dict() for q in self.qualifiers],
        }
        if self.subject_type is not None:
            d["subject_type"] = self.subject_type
        if self.object_type is not None:
            d["object_type"] = self.object_type
        if include_service and self.claim_ids is not None:
            d["claim_ids"] = self.claim_ids
        return d

    def core_key(self) -> tuple[str, str, str]:
        return (self.subject, self.relation, self.object)


@dataclass
class ExtractionResult:
    facts: list[Fact]
    schema_id: str = SCHEMA_ID
    service: dict[str, Any] = field(default_factory=dict)  # e.g. {"claims": [...]}

    def to_target_dict(self) -> dict[str, Any]:
        """Student-facing target: service fields excluded."""
        return {"triplets": [f.to_dict(include_service=False) for f in self.facts]}


_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|```$", re.MULTILINE)


def strip_code_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def _as_str(v: Any, ctx: str) -> str:
    if not isinstance(v, str):
        raise SchemaError(f"{ctx}: expected string, got {type(v).__name__}")
    return v


def parse_qualifier(obj: Any, ctx: str) -> Qualifier:
    if not isinstance(obj, dict):
        raise SchemaError(f"{ctx}: qualifier must be an object")
    return Qualifier(relation=_as_str(obj.get("relation"), ctx + ".relation"),
                     object=_as_str(obj.get("object"), ctx + ".object"))


def parse_fact(obj: Any, idx: int) -> Fact:
    ctx = f"triplets[{idx}]"
    if not isinstance(obj, dict):
        raise SchemaError(f"{ctx}: expected object")
    for req in ("subject", "relation", "object"):
        if req not in obj:
            raise SchemaError(f"{ctx}: missing required key '{req}'")
    quals_raw = obj.get("qualifiers", [])
    if quals_raw is None:
        quals_raw = []
    if not isinstance(quals_raw, list):
        raise SchemaError(f"{ctx}.qualifiers: expected list")
    quals = [parse_qualifier(q, f"{ctx}.qualifiers[{i}]") for i, q in enumerate(quals_raw)]
    claim_ids = obj.get("claim_ids")
    if claim_ids is not None:
        if not isinstance(claim_ids, list) or not all(isinstance(c, int) for c in claim_ids):
            raise SchemaError(f"{ctx}.claim_ids: expected list of ints")
    st = obj.get("subject_type")
    ot = obj.get("object_type")
    if st is not None:
        st = _as_str(st, ctx + ".subject_type")
    if ot is not None:
        ot = _as_str(ot, ctx + ".object_type")
    return Fact(
        subject=_as_str(obj["subject"], ctx + ".subject"),
        relation=_as_str(obj["relation"], ctx + ".relation"),
        object=_as_str(obj["object"], ctx + ".object"),
        qualifiers=quals,
        subject_type=st,
        object_type=ot,
        claim_ids=claim_ids,
    )


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Find and decode the first JSON object in a model response (no repair beyond fence stripping)."""
    text = strip_code_fences(raw_text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        raise SchemaError(f"top-level JSON must be an object, got {type(obj).__name__}")
    except json.JSONDecodeError as first_err:
        # tolerate leading prose *around* an object, but do not modify the object itself
        start = text.find("{")
        if start == -1:
            raise SchemaError(f"no JSON object found: {first_err}") from first_err
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError as e:
            raise SchemaError(f"invalid JSON: {e}") from first_err
        if not isinstance(obj, dict):
            raise SchemaError("top-level JSON must be an object")
        return obj


def parse_extraction(raw_text: str) -> ExtractionResult:
    obj = extract_json_object(raw_text)
    if "triplets" not in obj:
        raise SchemaError("missing key 'triplets'")
    triplets = obj["triplets"]
    if not isinstance(triplets, list):
        raise SchemaError("'triplets' must be a list")
    facts = [parse_fact(t, i) for i, t in enumerate(triplets)]
    service: dict[str, Any] = {}
    if isinstance(obj.get("claims"), list):
        service["claims"] = obj["claims"]
    return ExtractionResult(facts=facts, schema_id=SCHEMA_ID, service=service)
