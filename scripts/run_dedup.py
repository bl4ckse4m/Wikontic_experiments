"""Разбить объединённый пул на группы моделью института и сохранить представителей."""

import argparse
import math
from pathlib import Path

from wikontic_exp import api_client, dedup, experiment_io, extraction, recorded_calls
from wikontic_exp.api_client import DEFAULT_CREDS_FILE, _read_credentials
from wikontic_exp.dedup import merge_triplets, render_dedup_messages
from wikontic_exp.experiment_io import ROOT, digest, load_answers, load_sources, planned_lists, select_ids
from wikontic_exp.recorded_calls import run_recorded


def institute_id(model):
    # В архиве AutoOntik этот же gpt-oss запрошен через OpenRouter.
    return "Openai/Gpt-oss-120b" if model == "openai/gpt-oss-120b" else model


def select_source(path, method, ids):
    table = {key: r for key, r in load_answers(path).items() if key[1] == method}
    if not table:
        raise ValueError(f"Нет метода {method} в {path}")
    return planned_lists(table, ids)


def build_tasks(left, right, sources, *, model, method, prompt, temperature, max_tokens):
    tasks = []
    for a, b in zip(left, right, strict=True):
        iid = a["input_id"]
        if iid != b["input_id"]:
            raise ValueError("Наборы абзацев для объединения различаются")
        meta = {"input_id": iid, "method": method, "model": model,
                "inputs": [{k: r[k] for k in ("model", "method", "status", "source")} for r in (a, b)]}
        if a["status"] != "ok" or b["status"] != "ok":
            tasks.append({"metadata": meta, "body": None,
                          "without_call": {"status": "no_extraction", "triplets": None}})
            continue

        pool = a["triplets"] + b["triplets"]
        meta.update(pool_sha256=digest(pool), pool_size=len(pool))
        if not pool:
            tasks.append({"metadata": meta, "body": None,
                          "without_call": {"status": "ok", "triplets": [], "groups": []}})
            continue
        source = sources[iid]
        tasks.append({"metadata": meta, "body": {
            "model": model, "temperature": temperature, "max_tokens": max_tokens,
            "messages": render_dedup_messages(source["title"], source["paragraph_text"], pool, prompt),
        }})
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--left-method", required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--right-method", required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--model-id", required=True, help="ID одной из двух моделей института")
    parser.add_argument("--method", required=True, help="название результата объединения")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=ROOT / "prompts/merger/partition.txt")
    parser.add_argument("--creds-file", type=Path, default=DEFAULT_CREDS_FILE)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.out.resolve() in {
        p.resolve() for p in (args.left, args.right, args.sample, args.prompt, experiment_io.INPUTS)
    }:
        parser.error("--out должен отличаться от входных файлов")
    if (not math.isfinite(args.temperature) or args.temperature < 0
            or args.max_tokens < 1 or not math.isfinite(args.timeout) or args.timeout <= 0
            or (args.limit is not None and args.limit < 1)):
        parser.error("Недопустимые параметры генерации")

    sources = load_sources()
    try:
        ids = select_ids(sources, sample=args.sample)
        left = select_source(args.left, args.left_method, ids)
        right = select_source(args.right, args.right_method, ids)
        allowed = {institute_id(left[0]["model"]), institute_id(right[0]["model"])}
        if len(allowed) != 2 or args.model_id not in allowed:
            parser.error("Дедупликатор должен быть одной из двух разных моделей извлечения")
        tasks = build_tasks(left, right, sources, model=args.model_id, method=args.method,
                            prompt=args.prompt.read_text(encoding="utf-8"), temperature=args.temperature,
                            max_tokens=args.max_tokens)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"{len(tasks)} абзацев; {sum(t['body'] is not None for t in tasks)} запросов "
          f"к {args.model_id} до учёта сохранённых результатов", flush=True)
    if not args.execute:
        return

    pools = {a["input_id"]: a["triplets"] + b["triplets"] for a, b in zip(left, right, strict=True)
             if a["status"] == b["status"] == "ok"}

    def parse(raw, meta):
        groups, triplets = merge_triplets(raw, pools[meta["input_id"]])
        return {"status": "ok", "groups": groups, "triplets": triplets}

    url, key = _read_credentials(args.creds_file)
    version = digest([Path(p).read_text(encoding="utf-8") for p in (
        __file__, api_client.__file__, extraction.__file__, dedup.__file__,
        experiment_io.__file__, recorded_calls.__file__,
    )])
    run_recorded(args.out, tasks, stage="dedup", base_url=url, api_key=key,
                 timeout=args.timeout, limit=args.limit, parse=parse, version=version)


if __name__ == "__main__":
    main()
