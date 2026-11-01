"""Offline E08 stage gates. Does not load credentials or make model requests.

freeze: close calibration and bind the existing rubric, judge and source hashes.
prepare-judging: after the fixed merger run, export main candidates and blind pairs.
verify: check the immutable rubric and (if present) prediction locks.

Keep this adapter separate: existing E08 source files are already selection-locked.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from wikontic_exp import binary_judging, e08_preparation, llm_dedup
from wikontic_exp.aggregation import load_candidates
from wikontic_exp.hashing import file_sha256, hash_obj, sha256_hex

BASE = "data/experiments/e08"
CAL = "data/judging/e08/calibration"
MAIN = "data/judging/e08"
MERGER = "runs/e08_llm_dedup_xprompt_v2_50"
JUDGE = "openrouter/anthropic/claude-sonnet-5.5"
RUNS = e08_preparation.RUNS
METHODS = {
    "baseline_fixed": RUNS["baseline_fixed"], "coverage": RUNS["coverage"],
    "fewshot": RUNS["fewshot"], "autoontik": RUNS["autoontik"],
    "heuristic_union": "runs/ensemble_xprompt/merged_union.json",
    "heuristic_vote2": "runs/ensemble_xprompt/merged_vote2.json",
    "llm_union": f"{MERGER}/merged_union.json",
    "llm_vote2": f"{MERGER}/merged_vote2.json",
}


def path(root: Path, rel: str) -> Path:
    return e08_preparation.local_path(root.resolve(), rel)


def read(root: Path, rel: str):
    return json.loads(path(root, rel).read_bytes())


def rows(root: Path, rel: str) -> list[dict]:
    return [json.loads(line) for line in path(root, rel).read_bytes().splitlines() if line.strip()]


def hashes(root: Path, names) -> dict[str, str]:
    return {rel: file_sha256(str(path(root, rel))) for rel in sorted(set(names))}


def verify_hashes(root: Path, mapping: dict) -> None:
    for rel, expected in mapping.items():
        if file_sha256(str(path(root, rel))) != expected:
            raise ValueError(f"Stage lock mismatch: {rel}")


def create_files(root: Path, files: dict[str, bytes]) -> None:
    for rel in files:
        if path(root, rel).exists():
            raise ValueError(f"Existing work is not overwritten: {rel}")
    for rel, data in files.items():
        dest = path(root, rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as stream:
            stream.write(data)


def checked_labels(candidates: list[dict], labels: list[dict], *, pair: bool,
                   family: str, complete: bool) -> dict[str, dict]:
    key = "audit_id" if pair else "annotation_id"
    expected = {r[key] for r in candidates}
    if len(expected) != len(candidates):
        raise ValueError("Duplicate source annotation IDs")
    result = {}
    for row in labels:
        if not isinstance(row, dict):
            raise ValueError("Each judgment must be a JSON object")
        aid = row.get(key)
        if aid not in expected or aid in result:
            raise ValueError("Unknown/duplicate judgment ID")
        name = row.get("annotator")
        if not isinstance(name, str):
            raise ValueError("Invalid judge model metadata")
        name = name.lower().replace(" ", "-").removeprefix("openrouter/").removeprefix("anthropic/")
        if (name != f"claude-{family}-5.5" or row.get("status") != "silver"
                or not isinstance(row.get("rationale"), str) or not row["rationale"].strip()):
            raise ValueError("Invalid judgment metadata or wrong judge model")
        for label in (("equivalent",) if pair else ("core_label", "full_label")):
            if type(row.get(label)) is not int or row[label] not in (0, 1):
                raise ValueError("Invalid binary judgment label")
        if not pair and (not isinstance(row.get("evidence"), str) or row["full_label"] > row["core_label"]):
            raise ValueError("Invalid full/core judgment assertion")
        result[aid] = row
    if complete and set(result) != expected:
        raise ValueError("Incomplete calibration judgments")
    return result


def judgments(root: Path, candidate_rel: str, judgment_rel: str, *, pair: bool,
              family: str) -> dict[str, dict]:
    key = "audit_id" if pair else "annotation_id"
    candidates = rows(root, candidate_rel)
    expected = {r[key] for r in candidates}
    if len(expected) != 24 or len(candidates) != 24:
        raise ValueError("Expected 24 distinct calibration items")
    return checked_labels(candidates, rows(root, judgment_rel), pair=pair, family=family, complete=True)


def main_task(root: Path, kind: str) -> bytes:
    text = path(root, f"{CAL}/{kind}/TASK.md").read_text(encoding="utf-8")
    marker = "# Binary assessment" if kind == "binary" else "# Blind equivalence audit"
    # Remove only the calibration wrapper; keep the previously evaluated rubric.
    text = text[text.index(marker):].replace(f"{CAL}/{kind}/", f"{MAIN}/{kind}/")
    execution = (
        "# E08 main evaluation\n\n"
        f"Use {JUDGE} with the low reasoning variant in a fresh OpenCode session. "
        "If that model/variant is unavailable, stop; do not substitute another model.\n"
        "Judge each supplied ID once. Do not revise existing judgments. "
        "You may read this task's current output file only to resume IDs that are still missing. "
        "This is the only exception to the read-only input list below. "
        "Do not read calibration judgments, private manifests, reports, or source code.\n\n"
    )
    return (execution + text).encode("utf-8")


def freeze(root: Path) -> dict:
    e08_preparation.verify(root)
    if path(root, f"{BASE}/rubric_lock.json").exists():
        verify(root)
        return {"status": "already frozen; no files updated", "model_calls": 0}
    decision = read(root, f"{BASE}/calibration_decision.json")
    if (decision["decision"] != "close_calibration_with_known_limitations"
            or decision["main_judge"] != {"model": JUDGE, "reasoning": "low",
                                         "interface": "OpenCode fresh sessions", "status": "silver"}
            or decision["additional_calibration_runs"] != 0):
        raise ValueError("Execution decision differs from the accepted finite plan")
    reference_files = [f"{BASE}/lock.json", f"{BASE}/calibration_decision.json", "scripts/e08_main.py"]
    if path(root, "tests/test_e08_main.py").exists():
        reference_files.append("tests/test_e08_main.py")
    summary = {"status": "closed with known limitations; not validated ground truth",
               "known_limitations": decision["known_limitations"], "agreement": {}}
    tasks = {}
    for kind in ("binary", "merger_audit"):
        pair = kind != "binary"
        filename = "pairs.jsonl" if pair else "candidates.jsonl"
        candidate = f"{CAL}/{kind}/{filename}"
        task = f"{CAL}/{kind}/TASK.md"
        old = f"{CAL}/{kind}/judgments.jsonl"
        new = f"{CAL}/{kind}/judgments_sonnet.jsonl"
        haiku = judgments(root, candidate, old, pair=pair, family="haiku")
        sonnet = judgments(root, candidate, new, pair=pair, family="sonnet")
        for label in (("equivalent",) if pair else ("core_label", "full_label")):
            differing = sorted(aid for aid in haiku if haiku[aid][label] != sonnet[aid][label])
            summary["agreement"][label] = {"agree": 24 - len(differing), "n": 24,
                                          "disagreement_ids": differing,
                                          "note": "Agreement, not accuracy"}
        reference_files.extend([candidate, task, old, new])
        tasks[kind] = {"path": f"{MAIN}/{kind}/TASK.md",
                       "sha256": sha256_hex(main_task(root, kind))}
    summary_bytes = e08_preparation.json_bytes(summary)
    lock = {"experiment": "E08", "phase": "rubric_frozen_with_known_limitations",
            "judge": decision["main_judge"], "source_files_sha256": hashes(root, reference_files),
            "summary_sha256": sha256_hex(summary_bytes), "main_tasks": tasks,
            "note": "The low setting is the required main configuration, not verified historical API metadata."}
    create_files(root, {f"{BASE}/calibration_summary.json": summary_bytes,
                        f"{BASE}/rubric_lock.json": e08_preparation.json_bytes(lock)})
    return {"status": "calibration closed; rubric frozen", "model_calls": 0,
            "additional_calibration_runs": 0, "judge": decision["main_judge"]}


def verify(root: Path) -> dict:
    e08_preparation.verify(root)
    rubric = read(root, f"{BASE}/rubric_lock.json")
    verify_hashes(root, rubric["source_files_sha256"])
    verify_hashes(root, {f"{BASE}/calibration_summary.json": rubric["summary_sha256"]})
    for kind, spec in rubric["main_tasks"].items():
        if sha256_hex(main_task(root, kind)) != spec["sha256"]:
            raise ValueError("Main task differs from the frozen rubric")
    prediction_path = f"{BASE}/prediction_lock.json"
    label_status = {}
    if path(root, prediction_path).exists():
        prediction = read(root, prediction_path)
        verify_hashes(root, prediction["files_sha256"])
        current = sorted(p.relative_to(root).as_posix()
                         for p in path(root, f"{MERGER}/responses").glob("*.json"))
        if current != prediction["merger_response_files"]:
            raise ValueError("Main merger response set changed after prediction lock")
        for kind, filename in (("binary", "candidates.jsonl"), ("merger_audit", "pairs.jsonl")):
            candidates = rows(root, f"{MAIN}/{kind}/{filename}")
            output = f"{MAIN}/{kind}/judgments.jsonl"
            labels = rows(root, output) if path(root, output).exists() else []
            checked_labels(candidates, labels, pair=kind != "binary", family="sonnet", complete=False)
            label_status[kind] = {"expected": len(candidates), "present": len(labels),
                                  "missing": len(candidates) - len(labels)}
    return {"status": "verified", "model_calls": 0,
            "prediction_lock_present": path(root, prediction_path).exists(), "judgments": label_status}


def merger_data(root: Path) -> tuple[dict, dict, dict]:
    ids = read(root, f"{BASE}/selection.json")["input_ids"]
    selected = set(ids)
    union, vote = (read(root, METHODS[name]) for name in ("llm_union", "llm_vote2"))
    expected_params = {"temperature": 0, "max_tokens": 6000}
    for data, algorithm in ((union, "llm_equivalence"), (vote, "llm_equivalence_vote2")):
        if (data["algorithm"] != algorithm or data["algorithm_version"] != llm_dedup.VERSION
                or sorted(data["selected_input_ids"]) != ids
                or len(data["selected_input_ids"]) != 50
                or data["params"]["gen_params"] != expected_params
                or data["params"]["model"] != "openai/gpt-oss-120b"
                or data["params"]["prompt_hash"] != e08_preparation.V2_PROMPT_HASH
                or data["params"]["route"]["provider"] != "openrouter"
                or data["params"]["seed"] != 42):
            raise ValueError("Main merger configuration/frame differs from E08")
    if (vote["params"] != union["params"] or vote["failures"] != union["failures"]
            or vote["results"] != {iid: [f for f in fs if f["n_support"] >= 2]
                                  for iid, fs in union["results"].items()}):
        raise ValueError("LLM vote2 is not the fixed union subset")
    successes, failures = set(union["results"]), set(union["failures"])
    if successes & failures or (successes | failures) - selected:
        raise ValueError("Invalid merger result/failure membership")
    inputs = {r["input_id"]: r for r in rows(root, f"{BASE}/inputs.jsonl")}
    sources = read(root, f"{BASE}/preflight.json")["source_tasks"]
    originals = defaultdict(list)
    for name in ("coverage", "baseline_fixed", "fewshot"):
        for item in sources[name]:
            rec = read(root, f"{RUNS[name]}/responses/{item['cache_key']}.json")
            originals[item["input_id"]].extend(load_candidates([rec]))
    records = {}
    for file in sorted(path(root, f"{MERGER}/responses").glob("*.json")):
        rec = json.loads(file.read_bytes())
        iid = rec["input_id"]
        if iid not in selected or iid in records or rec["cache_key"] != file.stem:
            raise ValueError("Unexpected/ambiguous main merger response")
        expected = {"title": inputs[iid]["title"], "paragraph_text": inputs[iid]["paragraph_text"],
                    "candidates": [{"id": n, "fact": c.fact} for n, c in enumerate(originals[iid])]}
        if rec["payload"] != expected or rec["route"] != union["params"]["route"]:
            raise ValueError("Main merger payload/provenance differs from the original candidates")
        if iid in successes:
            if rec["status"] != "ok":
                raise ValueError("Failed response used as successful merger")
            groups = llm_dedup.validate_partition({"groups": rec["groups"]}, len(originals[iid]))
            materialized = llm_dedup.materialize(groups, originals[iid])
            for f in materialized:
                f["dedup_cache_key"] = rec["cache_key"]
            if materialized != union["results"][iid]:
                raise ValueError("Exported facts are not the validated original representatives")
        elif iid in failures:
            if rec["status"] != "failed" or rec.get("error") != union["failures"][iid]:
                raise ValueError("Failure export differs from the stored response")
        else:
            raise ValueError("Response not reflected in merger export; finish cache resume before judging")
        records[iid] = rec
    if successes - set(records):
        raise ValueError("Successful merger export has no source response")
    for iid in failures - set(records):
        if originals[iid] or union["failures"][iid] != "No usable extraction candidates; no merger request made":
            raise ValueError("Missing failed merger response")
    return union, vote, records


def pair_files(union: dict, records: dict) -> tuple[list[dict], dict]:
    pools = {True: [], False: []}
    for iid in sorted(union["results"]):
        rec = records[iid]
        labels = {cid: n for n, g in enumerate(rec["groups"]) for cid in g["members"]}
        cands = rec["payload"]["candidates"]
        for i in range(len(cands)):
            for j in range(i):
                pools[labels[i] == labels[j]].append((iid, j, i))
    rng = random.Random(42)
    public, private = [], []
    for same in (True, False):
        for iid, a, b in rng.sample(pools[same], min(100, len(pools[same]))):
            rec = records[iid]
            payload = rec["payload"]
            # Opaque IDs, unlike sequential IDs that reveal the two sampling blocks.
            aid = hash_obj({"e08_pair": iid, "pair": [a, b], "cache_key": rec["cache_key"]})
            public.append({"audit_id": aid, "title": payload["title"],
                           "paragraph_text": payload["paragraph_text"],
                           "a": payload["candidates"][a]["fact"], "b": payload["candidates"][b]["fact"]})
            private.append({"audit_id": aid, "llm_same": same, "input_id": iid,
                            "source_key": rec["cache_key"], "pair": [a, b]})
    rng.shuffle(public)
    return public, {"seed": 42, "population_pairs": {str(k): len(v) for k, v in pools.items()},
                    "pairs": private}


def prepare_judging(root: Path) -> dict:
    verify(root)
    if path(root, f"{BASE}/prediction_lock.json").exists():
        return {"status": "already prepared and verified; no files updated", "model_calls": 0}
    for kind in ("binary", "merger_audit"):
        folder = path(root, f"{MAIN}/{kind}")
        if folder.exists() and any(folder.iterdir()):
            raise ValueError(f"Existing annotation work is not overwritten: {folder}")
    union, vote, records = merger_data(root)
    ids = read(root, f"{BASE}/selection.json")["input_ids"]
    selected = set(ids)
    source_tasks = read(root, f"{BASE}/preflight.json")["source_tasks"]
    availability, active = {}, []
    for name, rel in METHODS.items():
        predictions = [r for r in binary_judging.prediction_rows(path(root, rel)) if r["input_id"] in selected]
        counts = Counter(r["input_id"] for r in predictions)
        if name in RUNS:
            states = {r["input_id"]: ("ok" if r["status"] == "ok" else "failed") for r in source_tasks[name]}
        elif name.startswith("heuristic"):
            states = {iid: ("ok" if any(next(r for r in source_tasks[m] if r["input_id"] == iid)["status"] == "ok"
                                       for m in ("coverage", "baseline_fixed", "fewshot")) else "failed") for iid in ids}
        else:
            states = {iid: ("ok" if iid in union["results"] else "failed" if iid in union["failures"] else "pending")
                      for iid in ids}
        availability[name] = {"n_evaluation_inputs": 50, "status_counts": dict(Counter(states.values())),
                              "n_successful_empty": sum(s == "ok" and not counts[iid] for iid, s in states.items()),
                              "n_inputs_with_predictions": len(counts), "population": len(predictions),
                              "input_states": states,
                              "note": "Empty population means undefined precision, not zero precision"}
        if predictions:
            active.append(f"{name}={path(root, rel)}")
    public_pairs, private_pairs = pair_files(union, records)
    binary_dir = path(root, f"{MAIN}/binary")
    if active:
        binary_summary = binary_judging.prepare(argparse.Namespace(
            methods=active, inputs=str(path(root, f"{BASE}/inputs.jsonl")),
            selection=str(path(root, f"{BASE}/selection.json")),
            out_dir=str(binary_dir), n=150, seed=42))
    else:
        create_files(root, {f"{MAIN}/binary/candidates.jsonl": b"",
                            f"{MAIN}/binary/sampling_private.json": e08_preparation.json_bytes({
                                "version": "binary-silver-1.0", "seed": 42, "methods": {}})})
        binary_summary = {"unique_facts_to_judge": 0}
    rubric = read(root, f"{BASE}/rubric_lock.json")
    files = {f"{MAIN}/merger_audit/pairs.jsonl": e08_preparation.jsonl_bytes(public_pairs),
             f"{MAIN}/merger_audit/audit_private.json": e08_preparation.json_bytes(private_pairs),
             f"{BASE}/availability.json": e08_preparation.json_bytes(availability)}
    for kind, spec in rubric["main_tasks"].items():
        content = main_task(root, kind)
        if sha256_hex(content) != spec["sha256"]:
            raise ValueError("Main task hash differs from rubric lock")
        files[spec["path"]] = content
    summary = {"unique_facts_to_judge": binary_summary["unique_facts_to_judge"],
               "pairs_to_judge": len(public_pairs),
               "zero_population_methods": [m for m, a in availability.items() if not a["population"]],
               "judge": rubric["judge"], "model_calls": 0,
               "note": "Zero-population methods stay in availability; precision is undefined. No rejudging/resampling."}
    files[f"{BASE}/judging_summary.json"] = e08_preparation.json_bytes(summary)
    create_files(root, files)
    input_files = [f"{BASE}/rubric_lock.json", *METHODS.values(),
                   *[f"{MERGER}/responses/{r['cache_key']}.json" for r in records.values()]]
    # Extraction run directories are already pinned by the selection lock.
    input_files = [r for r in input_files if path(root, r).is_file()]
    generated = [*files, f"{MAIN}/binary/candidates.jsonl", f"{MAIN}/binary/sampling_private.json"]
    verify(root)
    prediction_lock = {"experiment": "E08", "phase": "main_predictions_and_samples_frozen",
                       "files_sha256": hashes(root, input_files + generated),
                       "merger_response_files": sorted(p.relative_to(root).as_posix()
                                                       for p in path(root, f"{MERGER}/responses").glob("*.json"))}
    create_files(root, {f"{BASE}/prediction_lock.json": e08_preparation.json_bytes(prediction_lock)})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "verify", "prepare-judging"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = {"freeze": freeze, "verify": verify, "prepare-judging": prepare_judging}[args.command](args.root.resolve())
    except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
        parser.exit(1, f"E08 stopped without launching models: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
