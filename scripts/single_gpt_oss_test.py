import hashlib
import json
from pathlib import Path

import httpx

from wikontic_exp.api_client import _read_credentials
from wikontic_exp.extraction import parse_triplets, render_messages

iid = "201b128a1ad31b24"
method = "baseline_fixed"
root = Path("runs/prompt_recovery")

def rows(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

target = next(
    r for r in rows(root / "targets.jsonl")
    if (r["method"], r["input_id"]) == (method, iid)
)
source = next(
    r for r in rows(Path("data/tuning/inputs.jsonl")) if r["input_id"] == iid
)
prompt_path = Path(target["prompt"])
if hashlib.sha256(prompt_path.read_bytes()).hexdigest() != target["prompt_sha256"]:
    raise SystemExit("Промпт изменился: запрос не отправлен")

body = {
    "model": target["model"],
    "messages": render_messages(
        prompt_path.read_text(encoding="utf-8"),
        source["title"],
        source["paragraph_text"],
    ),
    **target["parameters"],
}
base_url, key = _read_credentials("~/.config/wikontic/creds")

out = root / f"{method}_{iid}"
if out.exists():
    raise SystemExit(f"Попытка уже записана в {out}; запрос не повторён")
out.mkdir()

def save(name, value):
    (out / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

def fail(reason, **details):
    save("result.json", {"status": "error", "reason": reason, **details})
    raise SystemExit(f"{reason}. См. {out}/result.json")

save("request.json", {"target": target, "route": "institute", "body": body})
try:
    response = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json=body,
        timeout=180,
    )
except httpx.RequestError as exc:
    fail(type(exc).__name__ + ": результат запроса неизвестен, повтора нет")

raw = response.content.replace(key.encode(), b"[REDACTED]")
(out / "response.txt").write_bytes(raw)
if response.status_code != 200:
    fail("Ошибка HTTP", http_status=response.status_code)

try:
    payload = json.loads(raw)
    returned_model = payload["model"]
    choice = payload["choices"][0]
    finish_reason = choice["finish_reason"]
    text = choice["message"]["content"]
except (ValueError, KeyError, IndexError, TypeError):
    fail("Неожиданный формат ответа API")

if returned_model != target["model"] or finish_reason != "stop":
    fail("Другая модель или ответ не завершён",
         model=returned_model, finish_reason=finish_reason)
if not isinstance(text, str):
    fail("Ответ не содержит текста")
try:
    triplets = parse_triplets(text, parse_mode="legacy")
except ValueError:
    fail("Не удалось разобрать триплеты")

save("result.json", {
    "status": "ok", "input_id": iid, "method": method,
    "model": returned_model, "finish_reason": finish_reason, "triplets": triplets,
})
print(f"OK: {returned_model}, finish_reason={finish_reason}, триплетов: {len(triplets)}")
print(out / "result.json")