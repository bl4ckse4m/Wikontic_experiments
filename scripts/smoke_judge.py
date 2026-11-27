import json
import re
from pathlib import Path

from wikontic_exp.judging import judge_triplets

creds_file = Path.home() / ".config/wikontic/creds"
for line in creds_file.read_text(encoding="utf-8").splitlines():
    match = re.fullmatch(r"\s*openrouter_key\s*[:=]\s*(.*?)\s*", line, re.I)
    if match:
        judge_key = match[1].strip("\"'")
        break
else:
    raise SystemExit(f"В {creds_file} не найден openrouter_key")

triplets = [
    {
        "subject": "Анна",
        "relation": "основала",
        "object": "музей",
        "qualifiers": [{"relation": "год основания", "object": "2010"}],
    },
    {
        "subject": "Анна",
        "relation": "основала",
        "object": "музей",
        "qualifiers": [{"relation": "год основания", "object": "2011"}],
    },
]

labels = judge_triplets(
    "История музея",
    "Анна основала музей в 2010 году.",
    triplets,
    judge_model="anthropic/claude-sonnet-5.5",
    base_url="https://openrouter.ai/api/v1",
    api_key=judge_key,
)

print("Метки:", labels)
if labels != [1, 0]:
    raise SystemExit("Ожидалось [1, 0]: первый год верный, второй — нет")