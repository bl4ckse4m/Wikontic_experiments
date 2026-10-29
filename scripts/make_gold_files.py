"""Generate fillable gold-annotation files from a selection file and tuning inputs.

Usage:
  python scripts/make_gold_files.py --inputs data/tuning/inputs.jsonl \
    --selection data/gold/gold_selection.json --out-dir data/gold/pending \
    [--pilot-inputs data/tuning/pilot12_inputs.jsonl --pilot-n 8]

Each output file data/gold/pending/<input_id>.gold.json contains the source text and
an empty "triplets" list to fill by hand (protocol in data/gold/gold_template.json).
Existing files are never overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def make_file(rec, out_dir: Path) -> str:
    out = out_dir / f"{rec['input_id']}.gold.json"
    if out.exists():
        return f"skip (exists): {out.name}"
    payload = {
        "input_id": rec["input_id"],
        "title": rec["title"],
        "paragraph_text": rec["paragraph_text"],
        "refs": rec.get("refs", []),
        "status": "unannotated",
        "triplets": [],
        "annotation_protocol": "data/gold/gold_template.json",
        "note": "fill triplets per protocol; then set status=annotated",
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"created: {out.name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pilot-inputs", default=None)
    ap.add_argument("--pilot-n", type=int, default=8)
    args = ap.parse_args()

    inputs = {r["input_id"]: r for r in load_jsonl(args.inputs)}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []

    if args.pilot_inputs:
        for rec in load_jsonl(args.pilot_inputs)[: args.pilot_n]:
            made.append(make_file(rec, out_dir))

    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    for item in selection.get("items", []):
        rec = inputs.get(item["input_id"])
        if rec is None:
            made.append(f"missing input {item['input_id']}")
            continue
        made.append(make_file(rec, out_dir))

    print(json.dumps({"n_files": len(list(out_dir.glob('*.gold.json'))), "actions": made},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
