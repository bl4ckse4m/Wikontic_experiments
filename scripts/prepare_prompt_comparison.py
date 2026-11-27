import argparse
import json
from collections import Counter
from pathlib import Path

from wikontic_exp.extraction import parse_triplets


DATA = Path("data/answers")
RECOVERY = Path("runs/prompt_recovery")
OUT = Path("runs/prompt_comparison/answers.jsonl")
COVERAGE_TIMEOUT_ID = "12366a4bfda0e7b3"

FILES = {
    "baseline_fixed": "gptoss_tuning_baseline_fixed_single.jsonl",
    "coverage": "gptoss_tuning_coverage_single.jsonl",
    "fewshot": "gptoss_tuning_fewshot_single.jsonl",
    "autoontik": "openrouter_pilot_autoontik_single.jsonl",
}


def rows(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def accepted_coverage_retry():
    previous = RECOVERY / f"coverage_{COVERAGE_TIMEOUT_ID}"
    retry = previous.with_name(previous.name + "_retry")
    old_request = json.loads((previous / "request.json").read_bytes())
    new_request = json.loads((retry / "request.json").read_bytes())
    if any(old_request[k] != new_request[k] for k in ("target", "body", "route")):
        raise ValueError("Повтор не совпадает с исходным запросом coverage")

    path = retry / "result.json"
    result = json.loads(path.read_bytes())
    raw = json.loads((retry / "response.txt").read_bytes())
    choice = raw["choices"][0]
    if (result["status"] != "ok"
            or (result["method"], result["input_id"]) != ("coverage", COVERAGE_TIMEOUT_ID)
            or result["model"] != raw["model"] or raw["model"] != old_request["body"]["model"]
            or result["finish_reason"] != "stop" or choice["finish_reason"] != "stop"):
        raise ValueError("Повтор coverage не прошёл проверку")
    triplets = parse_triplets(choice["message"]["content"], parse_mode="legacy")
    if triplets != result["triplets"]:
        raise ValueError("Список coverage не совпадает с сырым ответом")
    return triplets, str(path)


def main():
    parser = argparse.ArgumentParser(description="Собрать выдачи четырёх промптов без API")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--include-coverage-retry", action="store_true")
    args = parser.parse_args()

    input_ids = {r["input_id"] for r in rows(Path("data/tuning/inputs.jsonl"))}
    if len(input_ids) != 1863:
        raise ValueError("Ожидалось 1 863 уникальных абзаца tuning")

    recovered = {}
    for target in rows(RECOVERY / "targets.jsonl"):
        method, iid = target["method"], target["input_id"]
        path = RECOVERY / f"{method}_{iid}" / "result.json"
        result = json.loads(path.read_bytes())
        if result["status"] == "ok":
            recovered[method, iid] = (result["triplets"], str(path))
    if args.include_coverage_retry:
        recovered["coverage", COVERAGE_TIMEOUT_ID] = accepted_coverage_retry()

    auto_path = DATA / "openrouter_autoontik_recovery_12000.jsonl"
    auto, = rows(auto_path)
    if auto["status"] != "ok":
        raise ValueError("Дозапуск AutoOntik не принят")
    recovered["autoontik", auto["input_id"]] = (
        parse_triplets(auto["raw_text"], parse_mode="legacy"), str(auto_path)
    )

    output = []
    for method, filename in FILES.items():
        path = DATA / filename
        original = rows(path)
        if len(original) != len(input_ids) or {r["input_id"] for r in original} != input_ids:
            raise ValueError(f"ID не совпадают с tuning: {path}")

        for row in original:
            iid = row["input_id"]
            key = (method, iid)
            if row["status"] == "ok":
                triplets = parse_triplets(row["raw_text"], parse_mode="legacy")
                status, source = "ok", str(path)
            elif row["status"] == "truncated" and key in recovered:
                triplets, source = recovered[key]
                status = "ok"
            elif row["status"] == "truncated" and key == ("coverage", COVERAGE_TIMEOUT_ID):
                triplets, status = None, "missing"
                source = str(RECOVERY / f"{method}_{iid}" / "result.json")
            else:
                raise ValueError(f"Неожиданный пропуск: {method} / {iid}")

            output.append({
                "input_id": iid, "method": method, "status": status,
                "triplets": triplets, "source": source,
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        for row in output:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(args.out, dict(Counter(r["status"] for r in output)))


if __name__ == "__main__":
    main()
