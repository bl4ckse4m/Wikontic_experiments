"""Offline E08 continuation report, adapted from the original run's builder.

No inference, resampling, matcher changes, label edits or Git operations.
"""
import importlib.util
import hashlib
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
BASE_RUN = ROOT / "runs/e08_main_judging_20261101T202207_663551Z"
BASE_API = ROOT / "data/experiments/e08/api_judging"
CONTINUATION = RUN / "api_judging"
REPORT = ROOT / "reports/E08_RESULTS_RESUMED.md"
SPEC = importlib.util.spec_from_file_location("e08_resumed_report_stage", ROOT / "scripts/e08_main.py")
stage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage)


def read(rel):
    return json.loads((ROOT / rel).read_bytes())


def create_text(path, text):
    data = text.encode("utf-8")
    if path.exists():
        if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(data).digest():
            raise ValueError(f"Existing report artifact differs: {path.name}")
        return
    with path.open("xb") as stream:
        stream.write(data)


def create(path, obj):
    create_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


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
                 ("binary_precision.json", "merger_audit.json", "api_final_status.json", "smoke_check.json",
                  "build_report.py", "finalize_results.py", "resume_smoke_check.json")]
    required += [str((CONTINUATION / "plan.json").relative_to(ROOT)),
                 str((BASE_API / "plan.json").relative_to(ROOT)),
                 str((BASE_API / "requests.jsonl").relative_to(ROOT)),
                 str((BASE_API / "halt.json").relative_to(ROOT)),
                 str((BASE_RUN / "smoke_check.json").relative_to(ROOT)),
                 "scripts/e08_resume_api.py", "src/wikontic_exp/binary_judging.py",
                 "scripts/score_merger_audit.py"]
    if not set(required) <= set(provenance["files_sha256"]):
        raise ValueError("Scoring provenance lacks labels, scores or continuation history")
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


def execution_evidence(api):
    continuation = api["continuation"]
    original_attempts = list((BASE_API / "records").glob("*.attempt.json"))
    new_attempts = list((CONTINUATION / "records").glob("*.attempt.json"))
    original_checks = list((BASE_API / "key_checks").glob("*.json"))
    new_checks = list((CONTINUATION / "key_checks").glob("*.json"))
    if (len(original_attempts) != 70 or continuation["original_http_attempts"] != 70
            or len(new_attempts) != continuation["new_http_attempts"]
            or len(new_attempts) > 31 or len(original_checks) < len(original_attempts)
            or len(new_checks) < len(new_attempts)
            or not continuation["previous_failure_cost_unknown"]
            or f"prior_attempt/{continuation['retry_job_id']}" not in api["missing_cost_jobs"]):
        raise ValueError("Continuation history or unknown-cost accounting mismatch")
    total = len(original_attempts) + len(new_attempts)
    if sum(api["jobs"].values()) != 100:
        raise ValueError("Latest job states must cover the original 100 jobs")
    if api["jobs"].get("ok") == 100 and total != 101:
        raise ValueError("100 successful jobs require 101 retained HTTP attempts")
    return {"original_http_attempts": len(original_attempts), "new_http_attempts": len(new_attempts),
            "total_http_attempts": total, "maximum_http_attempts": 101,
            "original_key_checks": len(original_checks), "new_key_checks": len(new_checks),
            "original_failed_attempt_retained": True, "previous_failure_cost_unknown": True}


def primary_contrasts(methods):
    contrasts = []
    for left, right in (("autoontik", "baseline_fixed"), ("llm_union", "heuristic_union")):
        row = {"left": left, "right": right, "direction": "left minus right",
               "interpretation": "Descriptive point differences only; no superiority inferred"}
        for field in ("full_label", "core_label"):
            a, b = methods[left]["quality"], methods[right]["quality"]
            row[field + "_precision_difference"] = (
                a[field]["precision"] - b[field]["precision"] if field in a and field in b else None)
        contrasts.append(row)
    return contrasts


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
    plan = json.loads((BASE_API / "plan.json").read_bytes())
    if (smoke["status"] != "technical_smoke_passed" or not plan["operational_key_safety"]
            or (RUN / "smoke_check.json").read_bytes() != (BASE_RUN / "smoke_check.json").read_bytes()):
        raise ValueError("Missing original initial smoke or key safety evidence")
    evidence = execution_evidence(api)
    evidence["original_smoke_job_id"] = smoke["job_id"]
    evidence["smoke_scope"] = "Copied initial original smoke, not a new continuation smoke or quality validation"
    resumed_smoke = json.loads((RUN / "resume_smoke_check.json").read_bytes())
    if (resumed_smoke["status"] != "continuation_technical_smoke_passed"
            or resumed_smoke["job_id"] != api["continuation"]["retry_job_id"]
            or resumed_smoke["labels"] != 32):
        raise ValueError("Continuation transport smoke does not match the authorized retry")
    evidence["continuation_smoke_job_id"] = resumed_smoke["job_id"]
    resources = json.loads((RUN / "http_resources.json").read_bytes())
    key_after = json.loads((RUN / "key_status_after.json").read_bytes())
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
        records = single_records[name] if name in single_records else [
            r for m in ("coverage", "baseline_fixed", "fewshot") for r in single_records[m]]
        row = {"availability": availability[name], "quality": quality,
               "output_occurrences_per_input": len(rows) / len(selection),
               "serialized_output_chars_total": chars, "serialized_output_chars_per_input": chars / len(selection),
               "retained_extraction_records": len(records), "extraction_usage": usage(records)}
        if name.startswith("llm_"):
            row["merger_usage"] = usage(merger_records)
        if "full_label" in quality:
            row["estimated_full_supported_occurrences_per_input"] = len(rows) * quality["full_label"]["precision"] / len(selection)
        methods[name] = row
    complete = all(v["missing"] == 0 for v in verification["judgments"].values())
    contrasts = primary_contrasts(methods)
    result = {"experiment": "E08", "generated_utc": provenance["created_utc"],
              "status": "complete silver evaluation; no superiority inferred" if complete else "incomplete; no winner",
              "run_id": RUN.name, "quality_status": "exploratory LLM-assisted silver, no human validation",
              "recall": "not measured", "verification": verification, "methods": methods,
              "pair_audit": pairs, "judge_execution": api, "score_provenance": provenance,
               "execution_evidence": evidence, "http_resources": resources,
               "key_accounting_after": key_after, "primary_contrasts": contrasts,
              "inference": "Descriptive only; intervals do not account for paragraph/shared-label dependence or judge bias. No superiority p-values or corpus certification."}
    create(RUN / "metrics.json", result)
    lines = ["# E08 — результаты после разрешённого продолжения", "",
             f"Статус: **{result['status']}**. Run: `{RUN.name}`.",
             f"UTC сборки отчёта: {result['generated_utc']}.", "",
             "## Условия", "",
             "50 фиксированных tuning-абзацев MuSiQue-Ans v1.0 train. Четыре single-pass метода и четыре варианта кросс-промпт ансамбля; существующие predictions не перегенерировались.",
             "Stateless OpenRouter API, `anthropic/claude-sonnet-5.5`, reasoning=low, только Anthropic без fallback; system/user без tools и истории. Те же 886 уникальных фактов / 200 пар, rubric, batches и порядок.",
             "Для каждого ID сохраняется одна содержательная оценка. Разрешён один повтор исходного HTTP429 без labels и 30 ранее неотправленных jobs: максимум 101 HTTP-попытка на 100 научных batches, не повторная оценка успешных ответов.",
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
    lines += ["", "Supported occurrences — оценка числа правильных **выходных вхождений**, включая смысловые дубли; не уникальные факты и не покрытие. Chars — одинаковая JSON-сериализация fact fields по всем входам.",
              "", "## Основные contrasts", "", "Разность — первый метод минус второй; только описательные точечные оценки, без вывода о превосходстве."]
    for contrast in contrasts:
        values = []
        for field in ("full_label", "core_label"):
            value = contrast[field + "_precision_difference"]
            values.append(f"{field}: {value:+.4f}" if value is not None else f"{field}: withheld")
        lines.append(f"- {contrast['left']} vs {contrast['right']}: " + "; ".join(values) + ".")
    lines += ["", "## Отдельный аудит LLM-merger v2", ""]
    for name, r in pairs["strata"].items():
        if "error_rate" in r:
            lo, hi = r["wilson95"]
            lines.append(f"- {name}: ошибок {r['errors']}/{r['sampled']}, rate={r['error_rate']:.3f}, Wilson=[{lo:.3f}; {hi:.3f}], population pairs={r['population_pairs']}.")
        else:
            lines.append(f"- {name}: sampled={r['sampled']}, missing={r['missing']}; оценка withheld.")
    lines += ["", "Пары отбирались pair-uniform внутри proposed-merge/separate на успешных merger partitions. Это условные ошибки эквивалентности, не extraction recall. Эвристический partition отдельно по этим парам не оценивался; аудит не доказывает превосходства v2.",
              "", "## Исполнение и ресурсы", "",
              f"- Последние состояния 100 API jobs: `{json.dumps(api['jobs'], ensure_ascii=False)}`; labels: `{json.dumps(api['labels'], ensure_ascii=False)}`. Это не счётчик исторических HTTP-попыток.",
              f"- История HTTP: {evidence['original_http_attempts']} исходных + {evidence['new_http_attempts']} новых = {evidence['total_http_attempts']} (максимум 101). Исходный failed HTTP429 сохранён даже при latest ok после разрешённого повтора.",
              f"- Reported judge cost: ${api['reported_cost_usd']}; неизвестная стоимость: {len(api['missing_cost_jobs'])} записей. Conservative accounting с резервами: ${api['accounted_usd_including_unknown_reservations']} — не фактический счёт.",
              "- Исходный admission-control HTTP429 не дал labels; provider null/unreported и ошибка не доказывают отсутствие billing. Его неизвестная стоимость остаётся в `missing_cost_jobs` с префиксом `prior_attempt/` и в accounting, отдельно от результата повтора.",
              f"- Reported judge usage: `{json.dumps(api['usage_totals'])}`; provider metadata: `{json.dumps(api['providers'])}`.",
              f"- Key checks: {evidence['original_key_checks']} исходных / {evidence['new_key_checks']} новых; исходный guard margin=${plan['operational_key_safety']['margin_usd']}. Стоимость — учёт, не критерий выбора.",
              f"- После завершения read-only `/key`: remaining=${key_after.get('limit_remaining')}, daily usage=${key_after.get('usage_daily')}; это snapshot учёта, не доказательство окончательного billing failed попытки.",
              f"- Сумма latency всех {resources['http_latency_records']} HTTP receipts: {resources['http_latency_sum_s']:.2f} s, включая HTTP429; пауза между сегментами в эту сумму не входит.",
              "- Smoke evidence скопировано из исходного run: это initial transport/schema/identity smoke, не новый вызов и не проверка качества.",
              "- Отдельный continuation smoke — первый разрешённый retry batch, все 32 labels приняты после проверки HTTP/schema/IDs/model/provider; это не дополнительная калибровка и не аудит качества.",
              "- Single methods — 1 extraction/вход; heuristic ensemble — 3; LLM ensemble — 3 extraction + 1 merger/вход. Source usage, attempts и latency retained records — в metrics.json. Старые retries не полностью реконструированы; отсутствующая цена не равна нулю.",
              "- Сумма latency не равна wall-clock при параллельной обработке. Union/vote2 используют общие ресурсы: строки методов нельзя суммировать как независимые расходы.",
              "", "## История исполнения и ограничения", "",
              "Исходные delivery plan, requests, halt и receipts не изменялись. Новый continuation plan и ledger фиксируют разрешённое пользователем продолжение после HTTP429: только один точный повтор label-less failed job и 30 never-attempted jobs, без quality-conditioned retries, relabeling и научных изменений.",
              "API delivery заменил файловые агентные сессии до main labels; научные денежные caps и provider price filter были удалены, техническая защита ключа сохранена. Содержание rubric, samples, модель и output limits не менялись.",
              "Результаты ограничены tuning subset, historical route/retry differences (включая AutoOntik), непроверенным silver judge, общими labels и зависимостью фактов внутри абзаца. Target precision≥0.95 не сертифицирована даже при высокой точечной оценке. Старые matcher-зависимые P/R/F1 не используются как доказательство улучшения.",
              "", "## Артефакты", "",
              f"- `runs/{RUN.name}/metrics.json`, `binary_precision.json`, `merger_audit.json`, `api_final_status.json`, `score_provenance.json`.",
              f"- `runs/{RUN.name}/api_judging/`: continuation plan, история исходного состояния и новый append-only ledger; исходный `data/experiments/e08/api_judging/` сохранён.",
              "- `data/judging/e08/{binary,merger_audit}/judgments.jsonl`: объединённые labels, исходные строки сохранены. Selection/rubric/prediction locks и оба scorer не менялись.",
              "- `reports/E08_RESULTS.md` — неизменённый исторический неполный отчёт; данный отчёт имеет отдельный путь. Provenance связывает continuation plan/history, новый ledger, исходные receipts, labels и scores по SHA-256.",
              f"- `runs/{RUN.name}/original_label_snapshots/`: исходные 579/134 labels восстановлены только из прежних receipts; их hashes совпали с provenance частичного scoring. Main judgments после continuation — накопительный полный файл, не снимок старой остановки.",
              "", "## Граница решения", "",
              "При неполных labels E08 не завершён, победитель не выбирается. Полнота silver labels сама по себе не доказывает превосходства, истинный recall или пригодность корпуса. Близкие точечные оценки не образуют надёжный рейтинг. Ограниченное проектное решение оформляется отдельно после чтения результатов; автоматического запуска collection-1000 нет."]
    create_text(REPORT, "\n".join(lines) + "\n")
    print(json.dumps({"status": result["status"], "report": str(REPORT.relative_to(ROOT)),
                      "metrics": str((RUN / "metrics.json").relative_to(ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
