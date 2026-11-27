import json
import random
from pathlib import Path


def read_ids(path):
    with Path(path).open(encoding="utf-8") as f:
        return {json.loads(line)["input_id"] for line in f if line.strip()}


run = Path("runs/prompt_comparison")
tuning = read_ids("data/tuning/inputs.jsonl")
e08 = set(json.loads(Path("data/experiments/e08/selection.json").read_bytes())["input_ids"])
already_judged = read_ids(run / "judgments_full.jsonl")

pool = sorted(tuning - e08 - already_judged)
sample = sorted(random.Random(42).sample(pool, 50))

out = run / "sample50.json"
with out.open("x", encoding="utf-8") as f:
    json.dump({"seed": 42, "eligible_count": len(pool), "input_ids": sample},
              f, indent=2)
print(f"{out}: {len(sample)} из {len(pool)} абзацев")