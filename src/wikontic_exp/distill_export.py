"""Export of the distillation corpus: source text -> teacher target facts.

Student target = facts only. Provenance (teacher model/prompts/aggregation,
source ids, quality status) travels in separate metadata fields and must never
be concatenated into the generated student target. ``quality_status`` is
"unreviewed" until independent review; unreviewed is not gold.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def export_corpus(
    inputs_path: Path,
    merged_path: Path,
    out_path: Path,
    teacher_meta: dict[str, Any],
    only_quality_status: str = "unreviewed",
) -> dict[str, Any]:
    inputs = {}
    for line in inputs_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            inputs[rec["input_id"]] = rec
    merged = json.loads(merged_path.read_text(encoding="utf-8"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for iid, groups in sorted(merged["results"].items()):
            if iid not in inputs:
                continue
            facts = [g["fact"] for g in groups]
            student_target = {"triplets": facts}
            rec = {
                "source_text": inputs[iid]["paragraph_text"],
                "title": inputs[iid]["title"],
                "target": student_target,
                "meta": {
                    "input_id": iid,
                    "source_ids": inputs[iid].get("refs", []),
                    "teacher": teacher_meta,
                    "merger_algorithm": merged.get("algorithm"),
                    "merger_algorithm_version": merged.get("algorithm_version"),
                    "merger_cache_key": merged.get("merger_cache_key"),
                    "quality_status": only_quality_status,
                    "exported_utc": datetime.now(timezone.utc).isoformat(),
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    summary = {"n_records": n, "out_path": str(out_path), "quality_status": only_quality_status,
               "note": "quality_status=unreviewed: not gold until independent review"}
    (out_path.parent / (out_path.stem + "_export_summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
