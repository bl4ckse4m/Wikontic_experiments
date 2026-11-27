"""Офлайн-таблица Precision и размера выдачи на общих абзацах."""

import argparse
import json
from pathlib import Path

from wikontic_exp.experiment_io import comparison_lists, load_sources, read_jsonl, select_ids
from wikontic_exp.scoring import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    files = parser.add_mutually_exclusive_group(required=True)
    files.add_argument("--answers", type=Path)
    files.add_argument("--entry", nargs=3, action="append", metavar=("NAME", "METHOD", "FILE"))
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-model")
    parser.add_argument("--baseline", default="baseline_fixed")
    args = parser.parse_args()
    try:
        ids = select_ids(load_sources(), sample=args.sample)
        answers = comparison_lists(ids, path=args.answers, entries=args.entry, source_model=args.source_model)
        report = summarize(answers, read_jsonl(args.judgments), baseline=args.baseline)
    except ValueError as exc:
        parser.error(str(exc))
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Общие оценённые абзацы: {len(report['common_input_ids'])}/{len(ids)}"
          + ("" if report["complete"] else " (неполное сравнение)"))
    for method, row in report["methods"].items():
        precision = row["precision_common"]
        print(f"{method}: {row['supported_common']}/{row['candidates_common']}; "
              f"Precision = {f'{precision:.2%}' if precision is not None else '—'}")


if __name__ == "__main__":
    main()
