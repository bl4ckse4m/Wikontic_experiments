import csv
import json
from pathlib import Path


def rows(path):
    with Path(path).open(encoding="utf-8") as f:
        yield from map(json.loads, f)


lengths = {
    row["input_id"]: len(row["paragraph_text"])
    for row in rows("data/tuning/inputs.jsonl")
}

out = Path("runs/prompt_comparison/sizes.csv")
with out.open("x", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["input_id", "method", "paragraph_chars", "triplet_count"])
    for row in rows("runs/prompt_comparison/answers_complete.jsonl"):
        iid = row["input_id"]
        writer.writerow([iid, row["method"], lengths[iid], len(row["triplets"])])
print(out)