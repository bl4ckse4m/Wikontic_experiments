import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import httpx

from wikontic_exp.api_client import _read_credentials
from wikontic_exp.extraction import parse_triplets, render_messages

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/prompt_recovery"
EXPECTED = {"baseline_fixed": 2, "coverage": 4, "fewshot": 6}
READ_TIMEOUT_S = 600


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def prepare_jobs(skip_read_timeouts):
    sources = {r["input_id"]: r for r in rows(ROOT / "data/tuning/inputs.jsonl")}
    targets = rows(RUN / "targets.jsonl")
    counts = Counter(r["method"] for r in targets)
    ids = {(r["method"], r["input_id"]) for r in targets}
    if counts != EXPECTED or len(ids) != len(targets):
        raise SystemExit("Состав целей дозапуска изменился")

    jobs = []
    skipped_timeouts = set()
    for target in targets:
        method, iid = target["method"], target["input_id"]
        prompt = (ROOT / target["prompt"]).read_bytes()
        if hashlib.sha256(prompt).hexdigest() != target["prompt_sha256"]:
            raise SystemExit(f"Промпт изменился: {target['prompt']}")
        if target["model"] != "Openai/Gpt-oss-120b" or target["parameters"] != {
            "temperature": 0.2, "max_tokens": 12000,
        }:
            raise SystemExit(f"Изменились модель или параметры для {method} {iid}")

        source = sources[iid]
        body = {
            "model": target["model"],
            "messages": render_messages(prompt.decode("utf-8"), source["title"],
                                        source["paragraph_text"]),
            **target["parameters"],
        }
        request = {"target": target, "route": "institute", "body": body}
        out = RUN / f"{method}_{iid}"
        if out.exists():
            request_path, result_path = out / "request.json", out / "result.json"
            if not request_path.is_file() or not result_path.is_file():
                raise SystemExit(f"Незавершённая попытка: {out}; повтора нет")
            saved = json.loads(request_path.read_bytes())
            if any(saved.get(k) != v for k, v in request.items()):
                raise SystemExit(f"Сохранённый запрос не совпадает: {out}")
            previous = json.loads(result_path.read_bytes())
            response_path = out / "response.txt"
            if (previous["status"] in {"ok", "truncated"}
                    and response_path.is_file()
                    and previous["model"] == target["model"]
                    and (previous["method"], previous["input_id"]) == (method, iid)):
                print(f"{method} {iid}: уже обработано ({previous['status']})", flush=True)
                continue
            if (out.name in skip_read_timeouts and previous["status"] == "error"
                    and previous.get("reason", "").startswith("ReadTimeout:")
                    and not response_path.exists()):
                skipped_timeouts.add(out.name)
                print(f"{method} {iid}: ReadTimeout оставлен без повтора", flush=True)
                continue
            raise SystemExit(f"Ошибочная попытка: {out}; повтора нет")
        jobs.append((out, request))

    if skip_read_timeouts != skipped_timeouts:
        raise SystemExit("--skip-read-timeout должен указывать на уже записанный ReadTimeout без ответа")
    return jobs


def run_job(out, request, base_url, key):
    target = request["target"]
    method, iid = target["method"], target["input_id"]
    result_path = out / "result.json"
    out.mkdir()
    save(out / "request.json", {
        **request, "read_timeout_s": READ_TIMEOUT_S,
        "endpoint_sha256": hashlib.sha256(base_url.rstrip("/").encode()).hexdigest(),
    })
    result = {"input_id": iid, "method": method}

    def fail(reason, **details):
        save(result_path, {**result, "status": "error", "reason": reason, **details})
        raise SystemExit(f"{method} {iid}: {reason}; см. {result_path}")

    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=request["body"], timeout=httpx.Timeout(READ_TIMEOUT_S, connect=20),
        )
    except httpx.RequestError as exc:
        fail(type(exc).__name__ + ": результат запроса неизвестен, повтора нет")

    raw = response.content.replace(key.encode(), b"[REDACTED]")
    (out / "response.txt").write_bytes(raw)
    if response.status_code != 200:
        fail(f"HTTP {response.status_code}")

    try:
        payload = json.loads(raw)
        choice = payload["choices"][0]
        result.update(model=payload["model"], finish_reason=choice["finish_reason"])
    except (ValueError, KeyError, IndexError, TypeError):
        fail("Неожиданный формат ответа API")

    if result["model"] != target["model"]:
        fail("Ответила другая модель")
    if result["finish_reason"] == "length":
        save(result_path, {**result, "status": "truncated"})
        print(f"{method} {iid}: снова обрезано, повтора нет", flush=True)
        return
    if result["finish_reason"] != "stop":
        fail("Неожиданная причина остановки")

    try:
        text = choice["message"]["content"]
        if not isinstance(text, str):
            raise ValueError("Нет текста")
        triplets = parse_triplets(text, parse_mode="legacy")
    except (ValueError, KeyError, TypeError):
        fail("Не удалось разобрать триплеты")
    save(result_path, {**result, "status": "ok", "triplets": triplets})
    print(f"{method} {iid}: ok, триплетов {len(triplets)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Дозапуск только ещё не отправленных запросов")
    parser.add_argument("--execute", action="store_true", help="отправить запросы, а не только показать план")
    parser.add_argument("--skip-read-timeout", action="append", default=[], metavar="METHOD_INPUT_ID",
                        help="явно оставить старый ReadTimeout без ответа и без повтора")
    args = parser.parse_args()

    jobs = prepare_jobs(set(args.skip_read_timeout))
    print(f"Неотправленные запросы: {len(jobs)}", flush=True)
    if not args.execute or not jobs:
        for out, _ in jobs:
            print(out.name)
        return

    base_url, key = _read_credentials("~/.config/wikontic/creds")
    endpoint_sha = hashlib.sha256(base_url.rstrip("/").encode()).hexdigest()
    for path in RUN.glob("*/request.json"):
        saved_sha = json.loads(path.read_bytes()).get("endpoint_sha256")
        if saved_sha is not None and saved_sha != endpoint_sha:
            raise SystemExit("URL отличается от зафиксированного для дозапуска")
    for out, request in jobs:
        run_job(out, request, base_url, key)


if __name__ == "__main__":
    main()
