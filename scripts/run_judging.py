"""Одна проверка подтверждённости на один полный список триплетов."""

import argparse
import math
import re
from pathlib import Path

from wikontic_exp import api_client, experiment_io, extraction, judging, recorded_calls
from wikontic_exp.api_client import DEFAULT_CREDS_FILE
from wikontic_exp.experiment_io import (
    ROOT, comparison_lists, digest, load_answers, load_sources, planned_lists, select_ids,
)
from wikontic_exp.judging import parse_judge_labels, render_judge_messages
from wikontic_exp.recorded_calls import run_recorded

JUDGE_MODEL = "anthropic/claude-sonnet-5.5"
JUDGE_URL = "https://openrouter.ai/api/v1"
ANSWER_FILES = {"gptoss": "gptoss_paragraph.jsonl", "qwen": "qwen_paragraph.jsonl"}


def read_openrouter_key(path):
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"\s*openrouter_key\s*[:=]\s*(.*?)\s*", line, re.I)
        if match and (key := match[1].strip("\"'")):
            return key
    raise SystemExit("В файле настроек не найден непустой openrouter_key")


def build_tasks(answers, sources, judge_model):
    tasks = []
    for answer in answers:
        source = sources[answer["input_id"]]
        triplets = answer["triplets"]
        meta = {**answer, "extraction_status": answer["status"], "judge_model": judge_model}
        if answer["status"] != "ok":
            task = {"body": None, "without_call": {"status": "no_extraction", "labels": None}}
        elif not triplets:
            task = {"body": None, "without_call": {"status": "empty", "labels": []}}
        else:
            task = {"body": {"model": judge_model, "messages": render_judge_messages(
                source["title"], source["paragraph_text"], triplets,
            )}}
        tasks.append({"metadata": meta, **task})
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--input-id", action="append")
    selection.add_argument("--sample", type=Path, help="JSON со списком input_ids")
    selection.add_argument("--all", action="store_true", help="все абзацы tuning")
    parser.add_argument("--out", type=Path, required=True)
    files = parser.add_mutually_exclusive_group()
    files.add_argument("--answers", type=Path, help="подготовленные выдачи JSONL")
    files.add_argument("--entry", nargs=3, action="append", metavar=("NAME", "METHOD", "FILE"),
                       help="добавить отдельно именованную выдачу")
    parser.add_argument("--source-model", help="ID модели, если его нет в старом файле выдач")
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--creds-file", type=Path, default=DEFAULT_CREDS_FILE)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--limit", type=int, help="число новых заданий за одну команду")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="показать план (по умолчанию)")
    mode.add_argument("--execute", action="store_true", help="отправить запросы Claude")
    args = parser.parse_args()
    if (args.all or args.sample) and args.answers is None and not args.entry:
        parser.error("--all и --sample требуют --answers или --entry")
    paths = [args.answers] if args.answers else [Path(e[2]) for e in (args.entry or [])]
    if not paths:
        paths = [ROOT / "data/answers" / name for name in ANSWER_FILES.values()]
    paths.extend(p for p in (args.sample, experiment_io.INPUTS) if p is not None)
    if args.out.resolve() in {p.resolve() for p in paths}:
        parser.error("--out должен отличаться от входных файлов")
    if (not math.isfinite(args.timeout) or args.timeout <= 0
            or (args.limit is not None and args.limit < 1)
            or not args.judge_model.startswith("anthropic/claude-")):
        parser.error("Нужны допустимые timeout, limit и точный ID Claude")

    sources = load_sources()
    try:
        ids = select_ids(sources, sample=args.sample, ids=args.input_id)
        if args.answers or args.entry:
            lists = comparison_lists(ids, path=args.answers, entries=args.entry,
                                     source_model=args.source_model)
        else:
            table = {}
            for method, filename in ANSWER_FILES.items():
                table.update(load_answers(ROOT / "data/answers" / filename, default_method=method))
            lists = planned_lists(table, ids)
        tasks = build_tasks(lists, sources, args.judge_model)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"{len(ids)} абзацев, {len(tasks)} выдач; "
          f"{sum(t['body'] is not None for t in tasks)} запросов до учёта сохранённых результатов",
          flush=True)
    if not args.execute:
        return
    key = read_openrouter_key(args.creds_file)
    version = digest([Path(p).read_text(encoding="utf-8") for p in (
        __file__, api_client.__file__, extraction.__file__, judging.__file__,
        experiment_io.__file__, recorded_calls.__file__,
    )])
    run_recorded(
        args.out, tasks, stage="grounding", base_url=JUDGE_URL, api_key=key,
        timeout=args.timeout, limit=args.limit, version=version,
        parse=lambda text, meta: {"status": "ok", "labels": parse_judge_labels(text, len(meta["triplets"]))},
    )


if __name__ == "__main__":
    main()
