"""Последовательный запуск извлечения на сервере института."""

import argparse
import fcntl
import hashlib
import json
import math
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from wikontic_exp import api_client, extraction
from wikontic_exp.api_client import DEFAULT_CREDS_FILE, _read_credentials, chat_response
from wikontic_exp.extraction import parse_triplets, render_messages

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data/tuning/inputs.jsonl"
MODELS = {
    "gpt-oss": "Openai/Gpt-oss-120b",
    "qwen": "Qwen/Qwen3-235B-A22B-Instruct-2507",
}
PROMPTS = {
    "baseline_fixed": "prompts/baseline_fixed/propmt_1_types_qualifiers_fixed.txt",
    "coverage": "prompts/candidates/coverage_v1.txt",
    "fewshot": "prompts/candidates/fewshot_v1.txt",
    "autoontik": "prompts/autoontik/prompt_1_with_types_and_qualifiers.txt",
}
TERMINAL = {"ok", "invalid_output", "truncated"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(value):
    return json.dumps(value, ensure_ascii=False) + "\n"


def save(path, text):
    """Публиковать результат только после записи целого файла."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def sample_ids(path):
    return json.loads(path.read_bytes())["input_ids"]


def build_plan(args):
    data = INPUTS.read_bytes()
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    sources = {row["input_id"]: row for row in rows}
    if len(sources) != len(rows):
        raise SystemExit("Повторный input_id в tuning")

    if args.sample:
        ids = sample_ids(args.sample)
    elif args.all:
        first = sample_ids(args.sample_first) if args.sample_first else []
        first_set = set(first)
        ids = [*first, *(iid for iid in sources if iid not in first_set)]
    else:
        ids = args.input_id
    if (not ids or len(set(ids)) != len(ids)
            or any(iid not in sources or not re.fullmatch(r"[0-9a-f]{16}", iid) for iid in ids)):
        raise SystemExit("Нужен непустой список разных input_id из tuning")

    prompts = {}
    for method, path in PROMPTS.items():
        if args.prompt and method not in args.prompt:
            continue
        raw = (ROOT / path).read_bytes()
        prompts[method] = {"path": path, "sha256": sha(raw), "text": raw.decode("utf-8")}

    config = {
        "model": args.model_id if args.model_id is not None else MODELS[args.model],
        "route": "institute",
        "sources_sha256": sha(data),
        "input_ids": ids,
        "prompts": prompts,
        "temperature": args.temperature,
        "max_tokens": [args.max_tokens, args.retry_max_tokens],
        "read_timeout_s": args.timeout,
        "connect_timeout_s": 20,
        "retry_on": "length",
        "parse_mode": "legacy",
        "versions": {"python": platform.python_version(), "httpx": httpx.__version__},
        "code_sha256": {
            Path(path).name: sha(Path(path).read_bytes())
            for path in (__file__, api_client.__file__, extraction.__file__)
        },
    }
    return config, sources


def attempt_paths(job, number):
    folder = job / f"attempt_{number}"
    return folder / "request.json", folder / "result.json"


def last_result(job):
    """Получить последнее состояние без повторной отправки неопределённой попытки."""
    request, result = attempt_paths(job, 1)
    if not request.parent.exists():
        if job.exists():
            raise SystemExit(f"Неполная запись попытки: {job}; автоматического повтора нет")
        return None
    for i in (1, 2):
        request, result = attempt_paths(job, i)
        if i == 2 and not request.parent.exists():
            break
        if not request.is_file() or not result.is_file():
            raise SystemExit(f"Неполная запись попытки: {request.parent}; автоматического повтора нет")
        record = json.loads(result.read_bytes())
        status = record["status"]
        if (record["attempt"] != i or status not in TERMINAL | {"halted", "retry_length"}
                or (i == 2 and status == "retry_length")):
            raise SystemExit(f"Несогласованное состояние попытки: {result}")
        if status != "retry_length":
            if i == 1 and attempt_paths(job, 2)[0].parent.exists():
                raise SystemExit(f"Неожиданная вторая попытка: {job}")
            return record
    return record


def interpret(response, model, key, *, can_retry):
    raw = response.text.replace(key, "[REDACTED]")
    record = {"http_status": response.status_code, "response_body": raw}
    if response.status_code != 200:
        return dict(record, status="halted", error=f"HTTP {response.status_code}")
    try:
        payload = json.loads(raw)
        choice, = payload["choices"]
        result_model, finish = payload["model"], choice["finish_reason"]
    except (ValueError, KeyError, TypeError):
        return dict(record, status="halted", error="Неверный формат ответа API")
    record.update(returned_model=result_model, finish_reason=finish, usage=payload.get("usage"))
    if result_model != model:
        return dict(record, status="halted", error="Ответила другая модель")
    message = choice.get("message")
    text = message.get("content") if isinstance(message, dict) else None
    record["raw_text"] = text if isinstance(text, str) else None
    if finish == "length":
        return dict(record, status="retry_length" if can_retry else "truncated")
    if finish != "stop":
        return dict(record, status="invalid_output", error="Неожиданная причина завершения")
    try:
        if not isinstance(text, str):
            raise ValueError("Нет текста")
        triplets = parse_triplets(text, parse_mode="legacy")
    except ValueError as exc:
        return dict(record, status="invalid_output", error=str(exc))
    return dict(record, status="ok", triplets=triplets)


def run_job(job, source, method, config, base_url, key, previous):
    """Один запрос и, только при length, один запрос с большим лимитом."""
    model = config["model"]
    prompt = config["prompts"][method]
    messages = render_messages(prompt["text"], source["title"], source["paragraph_text"])
    start = 0 if previous is None else 1
    for i in range(start, 2):
        request, result = attempt_paths(job, i + 1)
        request.parent.mkdir(parents=True, exist_ok=False)
        params = {"temperature": config["temperature"], "max_tokens": config["max_tokens"][i]}
        started = datetime.now(timezone.utc).isoformat()
        save(request, dump({"started_utc": started, "model": model, "messages": messages, **params}))

        clock = time.monotonic()
        try:
            response = chat_response(
                model, messages, base_url=base_url, api_key=key,
                timeout=httpx.Timeout(config["read_timeout_s"], connect=config["connect_timeout_s"]),
                **params,
            )
        except (httpx.RequestError, httpx.InvalidURL) as exc:
            record = {"status": "halted", "error": type(exc).__name__}
        else:
            record = interpret(response, model, key, can_retry=i == 0)
        record.update(latency_s=round(time.monotonic() - clock, 3), parameters=params,
                      attempt=i + 1, input_id=source["input_id"], method=method)
        save(result, dump(record))

        if record["status"] != "retry_length":
            return record
        print(f"\n{source['input_id']} {method}: length; повтор с max_tokens={config['max_tokens'][1]}",
              file=sys.stderr, flush=True)


def progress(done, total, *, final=False):
    tty = sys.stderr.isatty()
    if tty or final or done % 100 == 0:
        filled = 24 * done // total
        bar = "#" * filled + "-" * (24 - filled)
        prefix = "\r" if tty else ""
        print(f"{prefix}[{bar}] {done}/{total} ({done / total:.1%})",
              file=sys.stderr, end="\n" if final or not tty else "", flush=True)


def export(out, config, finished):
    lines = []
    for (iid, method), record in finished.items():
        if record["status"] not in TERMINAL:
            continue
        result = attempt_paths(out / "jobs" / method / iid, record["attempt"])[1]
        prompt = config["prompts"][method]
        lines.append(dump({
            "input_id": iid, "method": method, "model": config["model"],
            "prompt": prompt["path"], "prompt_sha256": prompt["sha256"],
            "parameters": record["parameters"], "status": record["status"],
            "finish_reason": record["finish_reason"], "raw_text": record["raw_text"],
            "triplets": record.get("triplets"), "source": result.relative_to(out).as_posix(),
        }))
    path = out / "answers.jsonl"
    text = "".join(lines)
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        save(path, text)


def execute(args, config, sources):
    base_url, key = _read_credentials(args.creds_file)
    config["endpoint_sha256"] = sha(base_url.rstrip("/").encode("utf-8"))
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Другой процесс уже использует {out}") from None
        manifest = out / "config.json"
        if manifest.exists():
            if json.loads(manifest.read_bytes()) != config:
                raise SystemExit("Настройки запуска изменились; используйте новый --out-dir")
        else:
            if any(p.name != ".lock" for p in out.iterdir()):
                raise SystemExit("Папка не пуста и не содержит config.json")
            save(manifest, dump(config))

        jobs = {(iid, method): out / "jobs" / method / iid
                for iid in config["input_ids"] for method in config["prompts"]}
        finished = {}
        pending = []
        for (iid, method), job in jobs.items():
            record = last_result(job)
            if record is not None and record["status"] == "halted":
                raise SystemExit(f"Запуск остановлен на {method} / {iid}; см. {job}")
            if record is not None and record["status"] in TERMINAL:
                finished[iid, method] = record
            else:
                pending.append((iid, method, job, record))

        try:
            progress(len(finished), len(jobs), final=not pending)
            for iid, method, job, previous in pending[:args.limit]:
                record = run_job(job, sources[iid], method, config, base_url, key, previous)
                if record["status"] == "halted":
                    raise SystemExit(f"\n{method} / {iid}: {record['error']}; см. {job}")
                finished[iid, method] = record
                progress(len(finished), len(jobs))
        finally:
            export(out, config, finished)
            if pending:
                progress(len(finished), len(jobs), final=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--model", choices=MODELS)
    identity.add_argument("--model-id", help="точный ID новой модели на сервере института")
    parser.add_argument("--prompt", choices=PROMPTS, action="append")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="весь tuning")
    selection.add_argument("--sample", type=Path, help="JSON с input_ids")
    selection.add_argument("--input-id", action="append", help="явный input_id; можно повторить")
    parser.add_argument("--sample-first", type=Path, help="при --all начать с input_ids из этого JSON")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--creds-file", type=Path, default=DEFAULT_CREDS_FILE)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--retry-max-tokens", type=int, default=12000)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--limit", type=int, help="сколько новых пар обработать за этот запуск команды")
    parser.add_argument("--execute", action="store_true", help="отправить запросы; иначе показать план")
    args = parser.parse_args()
    if args.model_id is not None and (not args.model_id or any(c.isspace() for c in args.model_id)):
        parser.error("--model-id должен содержать непустой ID без пробелов")
    if args.sample_first and not args.all:
        parser.error("--sample-first требует --all")
    if not 0 < args.max_tokens < args.retry_max_tokens:
        parser.error("требуется 0 < --max-tokens < --retry-max-tokens")
    if (not math.isfinite(args.temperature) or args.temperature < 0
            or not math.isfinite(args.timeout) or args.timeout <= 0
            or (args.limit is not None and args.limit < 1)):
        parser.error("недопустимы temperature, timeout или limit")

    config, sources = build_plan(args)
    print(f"{config['model']}: {len(config['input_ids'])} абзацев × {len(config['prompts'])} промпта"
          f" = {len(config['input_ids']) * len(config['prompts'])} пар до повторов", flush=True)
    if args.execute:
        execute(args, config, sources)


if __name__ == "__main__":
    main()
