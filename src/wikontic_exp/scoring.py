"""Precision по оценкам Claude, без вызовов модели и без оценки дублей."""

from collections import Counter
from random import Random

from .experiment_io import check_labels


def ratio(k, n):
    return k / n if n else None


def paired_interval(left, right, ids, *, repeats=2000):
    """Описательный bootstrap разницы micro-Precision по общим абзацам."""
    if len(ids) < 2:
        return None
    rng = Random(42)
    bounds = []
    for _ in range(repeats):
        sampled = rng.choices(ids, k=len(ids))
        a_n = sum(left[i][1] for i in sampled)
        b_n = sum(right[i][1] for i in sampled)
        if not a_n or not b_n:
            return None
        bounds.append(sum(left[i][0] for i in sampled) / a_n
                      - sum(right[i][0] for i in sampled) / b_n)
    bounds.sort()
    return [bounds[int(0.025 * repeats)], bounds[int(0.975 * repeats) - 1]]


def summarize(answers, judgments, *, baseline):
    indexed = {}
    expected = {(r["input_id"], r["method"]): r for r in answers}
    judges = {r["judge_model"] for r in judgments}
    if len(judges) > 1:
        raise ValueError("Оценки разных судей нельзя смешивать в одной таблице")
    for row in judgments:
        key = row["input_id"], row["method"]
        if key not in expected or key in indexed:
            raise ValueError(f"Вне выборки или повторная оценка: {key}")
        source = expected[key]
        if (row["model"], row["extraction_status"], row["triplets"]) != (
            source["model"], source["status"], source["triplets"],
        ):
            raise ValueError(f"Оценка относится к другой выдаче: {key}")
        indexed[key] = row

    rows = {}
    labels_by_method = {}
    ids = sorted({r["input_id"] for r in answers})
    for method in sorted({r["method"] for r in answers}):
        counts = Counter()
        measured = {}
        all_candidates = 0
        for iid in ids:
            source = expected[iid, method]
            if source["status"] != "ok":
                counts["no_extraction"] += 1
                continue
            n = len(source["triplets"])
            all_candidates += n
            if not n:
                counts["empty"] += 1
                measured[iid] = (0, 0)
                continue
            verdict = indexed.get((iid, method))
            if verdict is None:
                counts["not_judged"] += 1
            elif verdict["status"] != "ok":
                counts["failed_judgment"] += 1
            else:
                check_labels(verdict["labels"], n)
                counts["judged"] += 1
                measured[iid] = (sum(verdict["labels"]), n)

        observed = counts["empty"] + sum(
            expected[iid, method]["status"] == "ok" and bool(expected[iid, method]["triplets"])
            for iid in ids
        )
        rows[method] = {
            "model": expected[ids[0], method]["model"], "counts": dict(counts),
            "candidates_all_observed": all_candidates, "observed_extractions": observed,
            "candidates_per_observed_paragraph": ratio(all_candidates, observed),
        }
        labels_by_method[method] = measured

    common = sorted(set.intersection(*(set(g) for g in labels_by_method.values())))
    for method, row in rows.items():
        measured = labels_by_method[method]
        k, n = (sum(measured[i][j] for i in common) for j in (0, 1))
        row.update(supported_common=k, candidates_common=n, precision_common=ratio(k, n),
                   candidates_per_common_paragraph=ratio(n, len(common)))
    if baseline in rows:
        reference = rows[baseline]["precision_common"]
        for method, row in rows.items():
            precision = row["precision_common"]
            if method != baseline and precision is not None and reference is not None:
                row["delta_vs_baseline"] = precision - reference
                row["delta_ci95"] = paired_interval(labels_by_method[method], labels_by_method[baseline], common)

    return {"judge_model": next(iter(judges), None), "baseline": baseline,
            "planned_paragraphs": len(ids), "common_input_ids": common,
            "complete": common == ids, "methods": rows}
