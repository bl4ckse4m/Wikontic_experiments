"""Один вызов на задание; сырые ответы и ошибки сохраняются до продолжения."""

import fcntl
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from platform import python_version

import httpx

from .api_client import chat_response
from .experiment_io import canonical, digest


def save(path, text):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def call(body, base_url, api_key, timeout, parse):
    started = time.monotonic()
    row = {"started_utc": datetime.now(timezone.utc).isoformat()}
    try:
        response = chat_response(
            body["model"], body["messages"], base_url=base_url, api_key=api_key,
            timeout=httpx.Timeout(timeout, connect=20),
            **{k: v for k, v in body.items() if k not in {"model", "messages"}},
        )
    except (httpx.RequestError, httpx.InvalidURL) as exc:
        return dict(row, status="halted", error=type(exc).__name__,
                    elapsed_s=round(time.monotonic() - started, 3))

    # Keep the response body but never preserve echoed authentication material.
    raw = response.text.replace(api_key, "[REDACTED]").replace(base_url, "[REDACTED]")
    row.update(http_status=response.status_code, response_body=raw,
               elapsed_s=round(time.monotonic() - started, 3))
    if response.status_code != 200:
        return dict(row, status="halted", error=f"HTTP {response.status_code}")
    try:
        envelope = json.loads(raw)
        choice, = envelope["choices"]
        finish = choice["finish_reason"]
        model = envelope["model"]
        row.update(response_model=model, finish_reason=finish, usage=envelope.get("usage"),
                   response_id=envelope.get("id"), provider=envelope.get("provider"))
        if model != body["model"]:
            return dict(row, status="halted", error="Ответила другая модель")
        if finish != "stop":
            return dict(row, status="invalid_output", error="Ответ не завершён с stop")
        text = choice["message"]["content"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Нет текста ответа")
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        return dict(row, status="invalid_output", error=f"Формат API: {type(exc).__name__}")
    row["raw_text"] = text
    try:
        return dict(row, **parse(text))
    except (TypeError, ValueError) as exc:
        return dict(row, status="invalid_output", error=str(exc))


def progress(done, total, *, final=False):
    tty = sys.stderr.isatty()
    if tty or final or done % 100 == 0:
        print(("\r" if tty else "") + f"Обработано: {done}/{total} ({done / total:.1%})",
              file=sys.stderr, end="\n" if final or not tty else "", flush=True)


def run_recorded(out, tasks, *, stage, base_url, api_key, timeout, limit, parse, version):
    """Записать задания в журнал; `out` — сводка, а не журнал попыток."""
    out = Path(out)
    journal = out.with_suffix(out.suffix + ".jobs")
    manifest_path = journal / "config.json"
    if out.exists() and not manifest_path.is_file():
        raise SystemExit(f"{out} уже существует без журнала; укажите новый --out")
    if not base_url or not api_key:
        raise ValueError("Требуются адрес и ключ одного сервиса")
    journal.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": stage, "version": version, "count": len(tasks), "plan_sha256": digest(tasks),
        "endpoint_sha256": digest(base_url.rstrip("/")), "read_timeout_s": timeout, "connect_timeout_s": 20,
        "versions": {"python": python_version(), "httpx": httpx.__version__},
    }
    with (journal / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Другой процесс уже использует {journal}") from None
        if manifest_path.exists():
            if json.loads(manifest_path.read_bytes()) != manifest:
                raise SystemExit("План запуска изменился; укажите новый --out")
        else:
            if any(p.name != ".lock" for p in journal.iterdir()):
                raise SystemExit(f"Журнал без манифеста: {journal}")
            save(manifest_path, canonical(manifest) + "\n")

        saved, pending = {}, []
        for i, task in enumerate(tasks, start=1):
            folder = journal / f"{i:06}"
            if not folder.exists():
                pending.append((i, folder, task))
                continue
            request, result = folder / "request.json", folder / "result.json"
            if not request.is_file() or not result.is_file():
                raise SystemExit(f"Незавершённая попытка: {folder}; повтора нет")
            if json.loads(request.read_bytes()) != task:
                raise SystemExit(f"Записанный запрос не совпадает: {folder}")
            row = json.loads(result.read_bytes())
            if any(row.get(k) != v for k, v in task["metadata"].items() if k != "status"):
                raise SystemExit(f"Ответ не совпадает с заданием: {folder}")
            if row["status"] == "halted":
                raise SystemExit(f"Запуск остановлен: {folder}; повтора нет")
            saved[i] = row

        try:
            progress(len(saved), len(tasks), final=not pending)
            for i, folder, task in pending[:limit]:
                folder.mkdir()
                save(folder / "request.json", canonical(task) + "\n")
                body, meta = task["body"], task["metadata"]
                if body is None:
                    result = task["without_call"]
                else:
                    result = call(body, base_url, api_key, timeout, lambda text: parse(text, meta))
                row = {**meta, **result}
                save(folder / "result.json", canonical(row) + "\n")
                saved[i] = row
                if row["status"] == "halted":
                    raise SystemExit(f"\n{meta['input_id']} / {meta['method']}: {row['error']}; см. {folder}")
                progress(len(saved), len(tasks))
        finally:
            # Full API responses remain in the individual journal entries.
            save(out, "".join(
                canonical({**{k: v for k, v in row.items() if k != "response_body"},
                           "result_file": (journal / f"{i:06}" / "result.json").relative_to(out.parent).as_posix()})
                + "\n"
                for i, row in sorted(saved.items())
            ))
            if pending:
                progress(len(saved), len(tasks), final=True)
