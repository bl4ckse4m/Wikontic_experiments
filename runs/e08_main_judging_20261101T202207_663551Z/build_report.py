"""Assemble descriptive E08 results from frozen samples and the existing scorers.

No inference, resampling, matcher changes, label edits or Git operations.
"""
from datetime import datetime, timezone
import importlib.util
import hashlib
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
SPEC = importlib.util.spec_from_file_location("e08_report_stage", ROOT / "scripts/e08_main.py")
stage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage)


def read(rel):
    return json.loads((ROOT / rel).read_bytes())


def create(path, obj):
    if path.exists():
        if json.loads(path.read_bytes()) != obj:
            raise ValueError(f"Existing report artifact differs: {path.name}")
        return
    with path.open("x", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def usage(records):
    out = {"retained_response_records": len(records), "prompt_tokens": 0,
           "completion_tokens": 0, "reported_cost_usd": 0.0, "cost_records": 0,
           "prompt_usage_records": 0, "completion_usage_records": 0,
           "recorded_attempts": 0, "attempts_records": 0,
           "recorded_latency_s": 0.0, "latency_records": 0}
    for rec in records:
        result = rec.get("response", rec)
        for name, total, count in (("attempts", "recorded_attempts", "attempts_records"),
                                   ("latency_s", "recorded_latency_s", "latency_records")):
            value = result.get(name)
            if type(value) in (int, float) and value >= 0:
                out[total] += value
                out[count] += 1
        u = result.get("usage")
        if not isinstance(u, dict):
            raw = result.get("raw_response")
            u = raw.get("usage", {}) if isinstance(raw, dict) else {}
        for key in ("prompt_tokens", "completion_tokens"):
            if type(u.get(key)) is int:
                out[key] += u[key]
                out[key.removesuffix("_tokens") + "_usage_records"] += 1
        if type(u.get("cost")) in (int, float):
            out["reported_cost_usd"] += u["cost"]
            out["cost_records"] += 1
    if not out["cost_records"]:
        out["reported_cost_usd"] = None
    out["note"] = "Measured usage only, never summed with estimates; retained historical records do not reconstruct replaced/retried attempts. Missing cost is not zero."
    return out


def verify_score_provenance():
    provenance = json.loads((RUN / "score_provenance.json").read_bytes())
    for rel, digest in provenance["files_sha256"].items():
        if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Stale score inputs or outputs: {rel}")
    required = [f"{stage.MAIN}/{kind}/judgments.jsonl" for kind in ("binary", "merger_audit")]
    required += [str((RUN / name).relative_to(ROOT)) for name in
                 ("binary_precision.json", "merger_audit.json", "api_final_status.json")]
    if not set(required) <= set(provenance["files_sha256"]):
        raise ValueError("Scoring provenance lacks labels or scores")
    return provenance


def verify_score_counts(binary, pairs, api, private, verification):
    for kind in ("binary", "merger_audit"):
        if api["labels"][kind] != verification["judgments"][kind]["present"]:
            raise ValueError("API status differs from current labels")
    ids = {r["annotation_id"] for r in stage.rows(ROOT, f"{stage.MAIN}/binary/judgments.jsonl")}
    for name, method in private["methods"].items():
        q = binary["methods"][name]
        missing = sum(r["annotation_id"] not in ids for r in method["samples"])
        if (q["missing"] != missing or q["sampled"] != len(method["samples"])
                or q["population"] != method["population"]
                or (missing and ("full_label" in q or "core_label" in q))
                or (not missing and ("full_label" not in q or "core_label" not in q))):
            raise ValueError(f"Binary score counts/withholding mismatch: {name}")
    pair_ids = {r["audit_id"] for r in stage.rows(ROOT, f"{stage.MAIN}/merger_audit/judgments.jsonl")}
    pair_manifest = read(f"{stage.MAIN}/merger_audit/audit_private.json")
    for same, name in ((True, "proposed_merge"), (False, "proposed_separate")):
        rows = [r for r in pair_manifest["pairs"] if r["llm_same"] == same]
        missing = sum(r["audit_id"] not in pair_ids for r in rows)
        q = pairs["strata"][name]
        if (q["missing"] != missing or q["sampled"] != len(rows)
                or (missing and "error_rate" in q) or (rows and not missing and "error_rate" not in q)):
            raise ValueError(f"Pair score counts/withholding mismatch: {name}")


def fmt_precision(result, field):
    if field not in result:
        return "withheld"
    r = result[field]
    lo, hi = r["approx_interval95"]
    return f"{r['precision']:.3f} [{lo:.3f}; {hi:.3f}]"


def main():
    provenance = verify_score_provenance()
    verification = stage.verify(ROOT)
    binary = json.loads((RUN / "binary_precision.json").read_bytes())
    pairs = json.loads((RUN / "merger_audit.json").read_bytes())
    api = json.loads((RUN / "api_final_status.json").read_bytes())
    availability = read(f"{stage.BASE}/availability.json")
    selection = read(f"{stage.BASE}/selection.json")["input_ids"]
    private = read(f"{stage.MAIN}/binary/sampling_private.json")
    verify_score_counts(binary, pairs, api, private, verification)
    smoke = json.loads((RUN / "smoke_check.json").read_bytes())
    plan = read("data/experiments/e08/api_judging/plan.json")
    if smoke["status"] != "technical_smoke_passed" or not plan["operational_key_safety"]:
        raise ValueError("Missing verified smoke or key safety evidence")
    attempts = list((ROOT / "data/experiments/e08/api_judging/records").glob("*.attempt.json"))
    checks = list((ROOT / "data/experiments/e08/api_judging/key_checks").glob("*.json"))
    if len(checks) < len(attempts):
        raise ValueError("Fewer key checks than attempts")
    tasks = read(f"{stage.BASE}/preflight.json")["source_tasks"]
    single_records = {}
    for name, rel in stage.RUNS.items():
        single_records[name] = [read(f"{rel}/responses/{r['cache_key']}.json") for r in tasks[name]]
    merger_records = [json.loads(p.read_bytes()) for p in sorted((ROOT / stage.MERGER / "responses").glob("*.json"))]
    methods = {}
    names = ("baseline_fixed", "coverage", "fewshot", "autoontik", "heuristic_union",
             "heuristic_vote2", "llm_union", "llm_vote2")
    for name in names:
        source = private["methods"][name]["path"]
        rows = [r for r in stage.binary_judging.prediction_rows(Path(source)) if r["input_id"] in selection]
        by_input = {iid: [] for iid in selection}
        for row in rows:
            by_input[row["input_id"]].append(row["fact"])
        chars = sum(len(json.dumps({"triplets": fs}, ensure_ascii=False)) for fs in by_input.values())
        quality = binary["methods"][name]
        if name in single_records:
            inputs_usage = usage(single_records[name])
            call_count = len(single_records[name])
        else:
            inputs_usage = usage([r for m in ("coverage", "baseline_fixed", "fewshot") for r in single_records[m]])
            call_count = sum(len(single_records[m]) for m in ("coverage", "baseline_fixed", "fewshot"))
        row = {"availability": availability[name], "quality": quality,
               "output_occurrences_per_input": len(rows) / len(selection),
               "serialized_output_chars_total": chars, "serialized_output_chars_per_input": chars / len(selection),
               "retained_extraction_records": call_count, "extraction_usage": inputs_usage}
        if name.startswith("llm_"):
            row["merger_usage"] = usage(merger_records)
        if "full_label" in quality:
            row["estimated_full_supported_occurrences_per_input"] = len(rows) * quality["full_label"]["precision"] / len(selection)
        methods[name] = row
    complete = all(v["missing"] == 0 for v in verification["judgments"].values())
    result = {"experiment": "E08", "generated_utc": provenance["created_utc"],
              "status": "complete silver evaluation" if complete else "incomplete; no winner",
              "run_id": RUN.name, "quality_status": "exploratory LLM-assisted silver, no human validation",
              "recall": "not measured", "verification": verification, "methods": methods,
               "pair_audit": pairs, "judge_execution": api, "score_provenance": provenance,
               "execution_evidence": {"smoke_job_id": smoke["job_id"], "attempts": len(attempts), "key_checks": len(checks)},
              "primary_contrasts": [["autoontik", "baseline_fixed"], ["llm_union", "heuristic_union"]],
              "inference": "Descriptive only; intervals do not account for paragraph/shared-label dependence or judge bias. No superiority p-values or corpus certification."}
    create(RUN / "metrics.json", result)
    lines = ["# E08 — сравнение готовых пайплайнов", "",
             f"Статус: **{result['status']}**. Run: `{RUN.name}`.",
             f"UTC сборки отчёта: {result['generated_utc']}.", "",
             "## Условия", "",
             "50 фиксированных tuning-абзацев MuSiQue-Ans v1.0 train. Четыре single-pass метода и четыре варианта кросс-промпт ансамбля; существующие predictions не перегенерировались.",
             "Stateless OpenRouter API, `anthropic/claude-sonnet-5.5`, reasoning=low, только Anthropic без fallback; два сообщения system/user без tools и истории. 886 уникальных фактов и 200 пар, один проход в фиксированном порядке.",
             "Source-first gold и человеческой проверки нет. Это **judged precision**, не независимая точность и не recall. Calibration в агентном интерфейсе не валидирует API-harness.", "",
             "## Общая таблица", "",
             "Интервалы — исходные approximate strata-Wilson/Bonferroni envelopes, не строгие тесты превосходства. В availability остаются все 50 входов; ошибки/пустые выдачи не исключаются.", "",
             "| Метод | ok/failed/pending | Empty ok | Inputs с facts / 50 | Occurrences N | Размечено / sampled | Full P [interval] | Core P [interval] | N / input | Оценка supported occurrences / input | Chars / input |",
             "|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|"]
    for name in names:
        r = methods[name]; a = r["availability"]; q = r["quality"]
        supported = r.get("estimated_full_supported_occurrences_per_input")
        counts = "/".join(str(a["status_counts"].get(k, 0)) for k in ("ok", "failed", "pending"))
        supported_text = f"{supported:.2f}" if supported is not None else "withheld"
        lines.append(f"| {name} | {counts} | {a['n_successful_empty']} | {a['n_inputs_with_predictions']} | {a['population']} | {q['sampled']-q['missing']}/{q['sampled']} | {fmt_precision(q,'full_label')} | {fmt_precision(q,'core_label')} | {r['output_occurrences_per_input']:.2f} | {supported_text} | {r['serialized_output_chars_per_input']:.0f} |")
    lines += ["", "Supported occurrences — оценка числа правильных **выходных вхождений**, включая смысловые дубли; не уникальные факты и не покрытие. Chars — одна и та же JSON-сериализация исходных fact fields, суммарно по всем входам.",
              "", "## Отдельный аудит LLM-merger v2", ""]
    for name, r in pairs["strata"].items():
        if "error_rate" in r:
            lo, hi = r["wilson95"]
            lines.append(f"- {name}: ошибок {r['errors']}/{r['sampled']}, rate={r['error_rate']:.3f}, Wilson=[{lo:.3f}; {hi:.3f}], population pairs={r['population_pairs']}.")
        else:
            lines.append(f"- {name}: sampled={r['sampled']}, missing={r['missing']}; оценка withheld.")
    lines += ["", "Пары отбирались pair-uniform внутри proposed-merge/separate на успешных merger partitions. Это условные ошибки эквивалентности, не extraction recall. Эвристический partition отдельно по этим парам не оценивался, поэтому аудит сам по себе не доказывает превосходства v2.",
              "", "## Исполнение и ресурсы", "",
              f"- Состояния API batches: `{json.dumps(api['jobs'],ensure_ascii=False)}`; labels: `{json.dumps(api['labels'],ensure_ascii=False)}`.",
              f"- Reported judge cost: ${api['reported_cost_usd']}; jobs с неизвестной стоимостью: {len(api['missing_cost_jobs'])}.",
              f"- Reported judge usage: `{json.dumps(api['usage_totals'])}`; provider metadata: `{json.dumps(api['providers'])}`.",
              f"- Execution ledger: {len(attempts)} attempts и {len(checks)} key checks; frozen key guard margin=${plan['operational_key_safety']['margin_usd']}. Лимит ключа не повышался; стоимость не критерий выбора.",
              "- Single methods — 1 extraction/вход; heuristic ensemble — 3; LLM ensemble — 3 extraction + 1 merger/вход. Source usage по выбранным входам — в metrics.json; старые replacements/retries не полностью реконструированы, отсутствующая цена не равна нулю.",
              "- В metrics.json отдельно сохранены доступные attempts и latency по retained records; сумма latency не равна wall-clock при исторической параллельной обработке. Union/vote2 повторно используют одни extraction/merger ресурсы: строки методов нельзя суммировать как независимые расходы.",
              "", "## Изменения исполнения до labels и ограничения", "",
              "API delivery заменяет файловые агентные сессии. Научные денежные caps и provider price filter удалены; введена отдельная техническая защита остатка ключа. Содержание rubric, samples, batches/order, модель и output limits сохранены; hashes и проверка эквивалентности запросов находятся в run artifacts.",
              "Первый batch основной очереди использован как технический smoke, затем продолжены только неотправленные batches. Дополнительных quality-conditioned retries, нового extractor/merger или ручных repairs labels нет.",
              "Результаты ограничены tuning subset, historical route/retry differences (включая AutoOntik), непроверенным silver judge, общими labels и зависимостью фактов внутри абзаца. Target precision≥0.95 не сертифицирована даже при высокой точечной оценке. Старые matcher-зависимые P/R/F1 не используются как доказательство улучшения.",
              "", "## Артефакты", "",
              f"- `runs/{RUN.name}/metrics.json`, `binary_precision.json`, `merger_audit.json`, `api_final_status.json`.",
              "- `data/experiments/e08/api_judging/`: фиксированный plan/requests, append-only attempts/responses и sanitized key checks.",
              "- `data/judging/e08/{binary,merger_audit}/judgments.jsonl`: сырые содержательные labels из сохранённых API-ответов.",
              "- Исходные selection/rubric/prediction locks сохранены. Публичный model catalog, forecast, версии, команды, hashes исходного/адаптированного кода и stdout/stderr — в run directory.",
              "", "## Граница решения", "",
              "При неполных labels E08 не завершён, победитель не выбирается. При полных labels основные contrasts — AutoOntik vs baseline_fixed и llm_union vs heuristic_union; близкие точечные оценки не образуют надёжный рейтинг. Автоматического запуска collection-1000 или расширения эксперимента нет."]
    report = ROOT / "reports/E08_RESULTS.md"
    text = "\n".join(lines) + "\n"
    if report.exists():
        if report.read_text(encoding="utf-8") != text:
            raise ValueError("Existing report differs; do not overwrite")
    else:
        with report.open("x", encoding="utf-8") as f:
            f.write(text)
    print(json.dumps({"status": result["status"], "report": str(report.relative_to(ROOT)), "metrics": str((RUN/'metrics.json').relative_to(ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
