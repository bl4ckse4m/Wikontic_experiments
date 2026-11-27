"""Общий формат списков триплетов и небольшие утилиты для CLI."""

import hashlib
import json
from pathlib import Path

from .extraction import parse_triplets

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "data/tuning/inputs.jsonl"

# Исторический answers_complete.jsonl не хранит model в самих строках.
_HISTORICAL_GPTOSS = {
    "baseline_fixed": "Openai/Gpt-oss-120b",
    "coverage": "Openai/Gpt-oss-120b",
    "fewshot": "Openai/Gpt-oss-120b",
    "autoontik": "openai/gpt-oss-120b",
}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def load_sources(path=None):
    rows = read_jsonl(path or INPUTS)
    sources = {r["input_id"]: r for r in rows}
    if len(sources) != len(rows):
        raise ValueError("Повторный input_id в текстах")
    return sources


def select_ids(sources, *, sample=None, ids=None):
    if sample is not None:
        ids = json.loads(Path(sample).read_bytes())["input_ids"]
    if ids is None:
        ids = list(sources)
    if (not isinstance(ids, list) or not ids or not all(isinstance(i, str) and i in sources for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Нужен непустой список разных input_id из исходных текстов")
    return ids


def load_answers(path, *, default_method=None, source_model=None):
    """Прочитать prepared/extraction/combined JSONL без переопределения выданных триплетов."""
    path = Path(path)
    historical = path.resolve() == ROOT / "runs/prompt_comparison/answers_complete.jsonl"
    table = {}
    for row in read_jsonl(path):
        iid, method = row["input_id"], row.get("method", default_method)
        key = iid, method
        if not isinstance(method, str) or not method or key in table:
            raise ValueError(f"Пустой метод или повторная пара: {key}")
        model = row.get("model")
        if model is None and historical:
            model = _HISTORICAL_GPTOSS.get(method)
        model = model or source_model
        if not isinstance(model, str) or not model:
            raise ValueError(f"Нет ID модели для {key}; для старого файла укажите --source-model")
        status = row["status"]
        triplets = row.get("triplets")
        if status == "ok":
            if "triplets" not in row:
                triplets = parse_triplets(row["raw_text"], parse_mode="legacy")
            if not isinstance(triplets, list) or not all(isinstance(t, dict) for t in triplets):
                raise ValueError(f"Неверный список триплетов: {key}")
        else:
            triplets = None
        source = row.get("source") or row.get("source_file")
        if source is None and row.get("result_file"):
            source = str(path.parent / row["result_file"])
        table[key] = {"input_id": iid, "method": method, "model": model, "status": status,
                      "triplets": triplets, "source": source}
    return table


def check_labels(labels, n):
    if (not isinstance(labels, list) or len(labels) != n
            or any(type(label) is not int or label not in (0, 1) for label in labels)):
        raise ValueError("Нужна ровно одна метка 0/1 на каждый триплет")


def planned_lists(table, ids):
    """Матрица абзац × метод, включая явно отсутствующие выдачи."""
    methods = sorted({method for _, method in table})
    if not methods:
        raise ValueError("Файл выдач пуст")
    models = {method: {r["model"] for r in table.values() if r["method"] == method} for method in methods}
    if any(len(group) != 1 for group in models.values()):
        raise ValueError("Для каждого метода в файле должен быть один точный ID модели")
    return [
        table.get((iid, method), {"input_id": iid, "method": method, "model": next(iter(models[method])),
                                 "status": "missing", "triplets": None, "source": None})
        for iid in ids for method in methods
    ]


def comparison_lists(ids, *, path=None, entries=None, source_model=None):
    """Либо один файл всех промптов, либо отдельные именованные выдачи."""
    if path is not None:
        return planned_lists(load_answers(path, source_model=source_model), ids)
    result = {}
    labels = set()
    for label, method, filename in entries:
        if not label or label in labels:
            raise ValueError("Названия сравниваемых выдач должны быть разными")
        labels.add(label)
        table = {key: row for key, row in load_answers(filename).items() if key[1] == method}
        if not table:
            raise ValueError(f"В {filename} нет выдачи {method}")
        for row in planned_lists(table, ids):
            result[row["input_id"], label] = {**row, "method": label}
    return planned_lists(result, ids)
