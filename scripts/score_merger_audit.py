"""Score a blind silver equivalence audit; never treat heuristic as truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wikontic_exp.binary_judging import wilson
from wikontic_exp.llm_dedup import write_json
from wikontic_exp.runner import load_inputs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--judgments", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    private = json.loads(Path(args.manifest).read_text())
    expected = {r["audit_id"]: r for r in private["pairs"]}
    labels = {}
    for row in load_inputs(args.judgments):
        aid = row.get("audit_id")
        if aid not in expected or aid in labels:
            raise ValueError("Unknown or duplicate audit_id")
        if (type(row.get("equivalent")) is not int or row["equivalent"] not in (0, 1)
                or row.get("status") != "silver" or not row.get("annotator") or not row.get("rationale")):
            raise ValueError("Expected binary equivalent, silver status, annotator and rationale")
        labels[aid] = row["equivalent"]
    report = {"quality_status": "LLM-assisted silver; preliminary", "strata": {},
              "note": "Conditional pair error rates, not fact precision/recall. Wilson intervals assume independent judgments; systematic LLM and paragraph dependence are not captured."}
    for same in (True, False):
        rows = [r for r in private["pairs"] if r["llm_same"] == same]
        missing = sum(r["audit_id"] not in labels for r in rows)
        result = {"sampled": len(rows), "missing": missing,
                  "population_pairs": private["population_pairs"].get(str(same), 0)}
        if rows and not missing:
            # Wrong merges among proposed merges; missed duplicates among nonmerges.
            errors = sum(labels[r["audit_id"]] != int(same) for r in rows)
            result.update(error_rate=errors / len(rows), errors=errors,
                          wilson95=wilson(errors, len(rows)))
        report["strata"]["proposed_merge" if same else "proposed_separate"] = result
    write_json(Path(args.out), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
