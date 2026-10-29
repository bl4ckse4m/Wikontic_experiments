"""Dataset preparation: MuSiQue sampling, dedup, manifests, anti-leak export.

Extraction unit (fixed): one ``(title, paragraph_text)`` pair together with its
title. All paragraphs are used, distractors included. Questions / answers /
decompositions / is_supporting never enter extraction inputs.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from .hashing import file_sha256, input_id

DATASET_NAME = "musique_ans"
DATASET_VERSION = "v1.0"
DATASET_URL = "https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view"
DATASET_REPO = "https://github.com/stonybrooknlp/musique"
DEFAULT_SPLIT = "train"
DEFAULT_FILE = "musique_raw/data/musique_ans_v1.0_train.jsonl"
SEED = 42
TUNING_N = 100
COLLECTION_N = 1000


@dataclass
class ParagraphRef:
    sample_id: str
    paragraph_idx: int
    is_supporting: bool
    title: str


@dataclass
class UniqueInput:
    input_id: str
    title: str
    paragraph_text: str
    refs: list[ParagraphRef] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "input_id": self.input_id,
            "title": self.title,
            "paragraph_text": self.paragraph_text,
            "refs": [
                {
                    "sample_id": r.sample_id,
                    "paragraph_idx": r.paragraph_idx,
                    "is_supporting": r.is_supporting,
                }
                for r in self.refs
            ],
        }


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_records(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def build_unique_inputs(records: Iterable[dict[str, Any]]) -> dict[str, UniqueInput]:
    """Deduplicate identical (title, paragraph_text) pairs, keeping all refs."""
    uniq: dict[str, UniqueInput] = {}
    for rec in records:
        for p in rec["paragraphs"]:
            iid = input_id(p["title"], p["paragraph_text"])
            if iid not in uniq:
                uniq[iid] = UniqueInput(input_id=iid, title=p["title"], paragraph_text=p["paragraph_text"])
            uniq[iid].refs.append(
                ParagraphRef(
                    sample_id=rec["id"],
                    paragraph_idx=p["idx"],
                    is_supporting=bool(p.get("is_supporting", False)),
                    title=p["title"],
                )
            )
    return uniq


def sample_split(
    pool: list[dict[str, Any]],
    tuning_n: int = TUNING_N,
    collection_n: int = COLLECTION_N,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Deterministic sampling: seed-shuffled pool -> tuning, then collection with
    no QA-id overlap and no (title, paragraph_text) overlap with tuning.

    Within-collection paragraph reuse is allowed (MuSiQue shares paragraphs across
    QA records): duplicates are handled at extraction-input level by deduplication,
    and every shared input keeps all refs. Any residual tuning/collection overlap
    is reported explicitly, never hidden."""
    ids = sorted(r["id"] for r in pool)
    by_id = {r["id"]: r for r in pool}
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)

    tuning_ids = shuffled[:tuning_n]
    tuning = [by_id[i] for i in tuning_ids]
    tuning_para_keys = _paragraph_keys(tuning)

    collection: list[dict[str, Any]] = []
    shared_inputs_detail: list[dict[str, Any]] = []
    rejected_for_tuning_overlap = 0
    for sid in shuffled[tuning_n:]:
        rec = by_id[sid]
        keys = _paragraph_keys([rec])
        shared = sorted(keys & tuning_para_keys)
        if shared:
            rejected_for_tuning_overlap += 1
            continue
        collection.append(rec)
        if len(collection) >= collection_n:
            break

    c_uniq = build_unique_inputs(collection)
    t_uniq = build_unique_inputs(tuning)
    cross = sorted(set(t_uniq) & set(c_uniq))
    for iid in cross:
        shared_inputs_detail.append({"input_id": iid, "present_in": ["tuning", "collection"]})

    info = {
        "seed": seed,
        "algorithm": (
            "sorted QA ids, random.Random(seed).shuffle, first tuning_n as tuning; "
            "then scan the remainder in shuffled order and take records with no "
            "(title, paragraph_text) overlap with tuning until collection_n records; "
            "within-collection paragraph reuse is allowed and deduplicated at input level"
        ),
        "pool_size": len(pool),
        "tuning_n_requested": tuning_n,
        "collection_n_requested": collection_n,
        "collection_n_obtained": len(collection),
        "rejected_for_tuning_paragraph_overlap": rejected_for_tuning_overlap,
        "paragraph_slots_collection": sum(len(r["paragraphs"]) for r in collection),
        "paragraph_slots_tuning": sum(len(r["paragraphs"]) for r in tuning),
    }
    return tuning, collection, {**info, "_shared_inputs_detail": shared_inputs_detail}


def _paragraph_keys(records: Iterable[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for rec in records:
        for p in rec["paragraphs"]:
            keys.add(input_id(p["title"], p["paragraph_text"]))
    return keys


def qa_id_overlap(tuning: list[dict[str, Any]], collection: list[dict[str, Any]]) -> list[str]:
    t = {r["id"] for r in tuning}
    return sorted(t & {r["id"] for r in collection})


def paragraph_overlap(tuning: list[dict[str, Any]], collection: list[dict[str, Any]]) -> list[str]:
    return sorted(_paragraph_keys(tuning) & _paragraph_keys(collection))


def manifest(
    records: list[dict[str, Any]],
    unique_inputs: dict[str, UniqueInput],
    split_file: str,
    info_extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": {
            "name": DATASET_NAME,
            "version": DATASET_VERSION,
            "source_url": DATASET_URL,
            "repo": DATASET_REPO,
            "split": DEFAULT_SPLIT,
            "split_file": split_file,
            "split_file_sha256": file_sha256(split_file),
        },
        "extraction_unit": "title + paragraph_text (one call per unit); all paragraphs incl. distractors",
        "n_qa_records": len(records),
        "qa_ids": [r["id"] for r in records],
        "n_unique_paragraph_inputs": len(unique_inputs),
        "n_supporting_paragraph_refs": sum(
            1 for u in unique_inputs.values() for r in u.refs if r.is_supporting
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **info_extra,
    }


def write_inputs_jsonl(path: Path, unique_inputs: dict[str, UniqueInput]) -> None:
    """Extraction inputs: NO question/answer/decomposition/is_supporting at top level
    (refs keep is_supporting only as *evaluation* metadata; the extractor never sees it)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for iid in sorted(unique_inputs):
            rec = unique_inputs[iid].to_record()
            out = {"input_id": rec["input_id"], "title": rec["title"], "paragraph_text": rec["paragraph_text"], "refs": rec["refs"]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def write_qa_sidecar(path: Path, records: list[dict[str, Any]]) -> None:
    """QA metadata kept aside for evaluation only (never sent to the extractor)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(
                json.dumps(
                    {
                        "id": r["id"],
                        "question": r["question"],
                        "answer": r.get("answer"),
                        "answer_aliases": r.get("answer_aliases"),
                        "question_decomposition": r.get("question_decomposition"),
                        "answerable": r.get("answerable"),
                        "paragraphs": [
                            {"idx": p["idx"], "title": p["title"], "is_supporting": p.get("is_supporting")}
                            for p in r["paragraphs"]
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def prepare(data_dir: Path, split_file: str = DEFAULT_FILE, seed: int = SEED,
            tuning_n: int = TUNING_N, collection_n: int = COLLECTION_N) -> dict[str, Any]:
    pool = read_records(split_file)
    tuning, collection, info = sample_split(pool, tuning_n, collection_n, seed)
    shared_detail = info.pop("_shared_inputs_detail", [])
    t_uniq = build_unique_inputs(tuning)
    c_uniq = build_unique_inputs(collection)

    data_dir.mkdir(parents=True, exist_ok=True)
    write_inputs_jsonl(data_dir / "tuning" / "inputs.jsonl", t_uniq)
    write_inputs_jsonl(data_dir / "collection" / "inputs.jsonl", c_uniq)
    write_qa_sidecar(data_dir / "tuning" / "qa_sidecar.jsonl", tuning)
    write_qa_sidecar(data_dir / "collection" / "qa_sidecar.jsonl", collection)

    overlap_ids = qa_id_overlap(tuning, collection)
    overlap_paras = paragraph_overlap(tuning, collection)

    t_man = manifest(tuning, t_uniq, split_file, {**info, "stage": "tuning",
                                                 "qa_id_overlap_with_collection": overlap_ids,
                                                 "paragraph_overlap_with_collection": overlap_paras,
                                                 "collection_records_with_shared_paragraphs": shared_detail})
    c_man = manifest(collection, c_uniq, split_file, {**info, "stage": "collection",
                                                     "qa_id_overlap_with_tuning": overlap_ids,
                                                     "paragraph_overlap_with_tuning": overlap_paras,
                                                     "records_with_shared_paragraphs": shared_detail})
    (data_dir / "tuning" / "tuning_manifest.json").write_text(
        json.dumps(t_man, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (data_dir / "collection" / "collection_manifest.json").write_text(
        json.dumps(c_man, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"tuning": t_man, "collection": c_man}


def select_gold_sample(inputs_path: Path, out_path: Path, n: int = 30, seed: int = 7) -> dict[str, Any]:
    """Pre-select gold-annotation inputs BEFORE any system comparison: stratified by
    length and by supporting/distractor status (diagnostic split, fixed independently
    of model outputs)."""
    inputs = [json.loads(l) for l in inputs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for rec in inputs:
        rec["_len"] = len(rec["paragraph_text"])
        rec["_supporting"] = any(r["is_supporting"] for r in rec["refs"])
    inputs_sorted = sorted(inputs, key=lambda r: r["_len"])
    third = max(1, len(inputs_sorted) // 3)
    strata = {
        "short": inputs_sorted[:third],
        "medium": inputs_sorted[third:2 * third],
        "long": inputs_sorted[2 * third:],
    }
    rng = random.Random(seed)
    picked: list[dict[str, Any]] = []
    per = max(1, n // 6)
    for name, bucket in strata.items():
        supp = [r for r in bucket if r["_supporting"]]
        dist = [r for r in bucket if not r["_supporting"]]
        for sub, tag in ((supp, "supporting"), (dist, "distractor")):
            if not sub:
                continue
            take = min(per, len(sub))
            for rec in rng.sample(sub, take):
                picked.append({
                    "input_id": rec["input_id"],
                    "title": rec["title"],
                    "stratum": f"{name}/{tag}",
                    "paragraph_chars": rec["_len"],
                })
    out = {
        "purpose": "independent gold annotation of source paragraphs (proxy eval until annotated)",
        "selection_seed": seed,
        "n_requested": n,
        "n_selected": len(picked),
        "stratification": "paragraph length thirds x supporting/distractor; chosen before any model comparison",
        "items": sorted(picked, key=lambda x: x["input_id"]),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
