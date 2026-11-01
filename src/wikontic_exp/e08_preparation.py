"""Offline E08 selection lock and blind calibration export. No API/credential imports.

python -m wikontic_exp.e08_preparation prepare
python -m wikontic_exp.e08_preparation verify
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .hashing import file_sha256, hash_obj, input_id, sha256_hex

SEED = 42
N_INPUTS = 50
EXPERIMENT_DIR = "data/experiments/e08"
CALIBRATION_DIR = "data/judging/e08/calibration"
V2_RUN = "runs/openrouter_llm_dedup_xprompt_v2_300"
V1_RUN = "runs/openrouter_llm_dedup_xprompt_300"
V2_PROMPT = "prompts/merger/llm_dedup_v2.txt"
V2_PROMPT_HASH = "34d0b0d54eaa5b2df3afac6bd1b141c9b05ea11813e15f2d0b2325fcd4d900b1"
RUNS = {
    "coverage": "runs/gptoss_tuning_coverage_single",
    "baseline_fixed": "runs/gptoss_tuning_baseline_fixed_single",
    "fewshot": "runs/gptoss_tuning_fewshot_single",
    "autoontik": "runs/openrouter_pilot_autoontik_single",
}
GLOBS = (
    "data/gold/pending/*.gold.json",
    "runs/gptoss_pilot_*/index.jsonl",
    "runs/gptoss_pilot2_*/index.jsonl",
) + tuple(f"{run}/responses/*.json" for run in (*RUNS.values(), "runs/gptoss_ens3_300"))

# Curated references to ORIGINAL candidates, not labels or repaired facts.
# These IDs were inspected during E07 and are excluded from E08's main sample.
EU = "046d77d5cc8f1734"
ARMY = "23d6798d0d114080"
CHESTER = "16942cb24a0e3092"
UNICODE = "2a1d1293da56a7ea"
PLANET = "0e4ee65735284e92"
PHINEAS = "299042019bdd723e"
CALIBRATION_TITLES = {
    EU: "European Union law", ARMY: "Army of Northern Virginia", CHESTER: "Chester Crandell",
    UNICODE: "Unicode", PLANET: "HD 11964 c", PHINEAS: "List of Phineas and Ferb characters",
}
FACT_CASES = (
    (EU, 0), (EU, 3), (EU, 8), (EU, 15), (EU, 24), (EU, 27), (EU, 28),
    (PLANET, 1), (PLANET, 6), (PLANET, 11), (PLANET, 15),
    (PLANET, 17), (PLANET, 24), (PLANET, 29),
    (CHESTER, 2), (CHESTER, 6), (CHESTER, 10), (CHESTER, 13),
    (UNICODE, 0), (UNICODE, 1), (UNICODE, 33), (UNICODE, 36),
    (ARMY, 10), (PHINEAS, 2),
)
PAIR_CASES = (
    (EU, 8, 27), (EU, 3, 13), (EU, 5, 24), (EU, 1, 11), (EU, 9, 28),
    (ARMY, 0, 10), (ARMY, 0, 24), (ARMY, 3, 14), (ARMY, 5, 18),
    (ARMY, 6, 33), (ARMY, 21, 35),
    (CHESTER, 2, 10), (CHESTER, 6, 13), (CHESTER, 13, 20), (CHESTER, 0, 7),
    (UNICODE, 5, 36), (UNICODE, 6, 37), (UNICODE, 4, 20), (UNICODE, 1, 33),
    (PLANET, 13, 26), (PLANET, 4, 27), (PLANET, 11, 24), (PLANET, 6, 17),
    (PHINEAS, 6, 19),
)


def local_path(root: Path, relative: str) -> Path:
    """Do not follow a manifest path or symlink outside this project."""
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected project-relative path: {relative}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path leaves project: {relative}")
    return resolved


class Inventory:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.files: dict[str, str] = {}

    def read(self, relative: str, *, track: bool = True) -> bytes:
        data = local_path(self.root, relative).read_bytes()
        if track:
            digest = sha256_hex(data)
            if relative in self.files and self.files[relative] != digest:
                raise ValueError(f"Source changed during preparation: {relative}")
            self.files[relative] = digest
        return data

    def json(self, relative: str, *, track: bool = True):
        return json.loads(self.read(relative, track=track))

    def jsonl(self, relative: str) -> list[dict]:
        return [json.loads(line) for line in self.read(relative).splitlines() if line.strip()]

    def discovery(self) -> dict[str, list[str]]:
        return {pattern: sorted(p.relative_to(self.root).as_posix()
                                for p in self.root.glob(pattern)) for pattern in GLOBS}


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode("utf-8")


def input_map(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        iid = row["input_id"]
        if iid in result:
            raise ValueError(f"Duplicate input ID: {iid}")
        if iid != input_id(row["title"], row["paragraph_text"]):
            raise ValueError(f"Input ID does not match text: {iid}")
        result[iid] = row
    return result


def collect_exclusions(inv: Inventory, discovered: dict, extra: str | None) -> dict[str, list[str]]:
    reasons = defaultdict(set)

    def add(ids, reason):
        for iid in ids:
            if not isinstance(iid, str) or not iid:
                raise ValueError("Invalid exclusion ID")
            reasons[iid].add(reason)

    for run in (V1_RUN, V2_RUN):
        rel = f"{run}/selection.json"
        ids = inv.json(rel)["input_ids"]
        if len(ids) != 20 or len(set(ids)) != 20:
            raise ValueError(f"Expected preserved 20-input development selection: {rel}")
        add(ids, rel)
    add(inv.json("reports/tuning_comparison.json")["gold_kinds"], "historical_silver")
    add((r["input_id"] for r in inv.json("data/gold/gold_selection.json")["items"]),
        "gold_selection")
    add((r["input_id"] for r in inv.jsonl("data/tuning/pilot12_inputs.jsonl")), "pilot12")
    for rel in discovered["data/gold/pending/*.gold.json"]:
        # Only filenames identify exposure. Never read old annotation contents.
        add([Path(rel).name.removesuffix(".gold.json")], "historical_annotation_filename")
    for pattern in GLOBS[1:3]:
        for rel in discovered[pattern]:
            add((r["input_id"] for r in inv.jsonl(rel)), rel)
    add(["12366a4bfda0e7b3"], "manually_inspected_autoontik_recovery")
    if extra:
        for row in inv.json(extra):
            if not isinstance(row.get("reason"), str) or not row["reason"].strip():
                raise ValueError("Extra exclusions require a nonempty reason")
            add([row["input_id"]], "additional_exposure: " + row["reason"])
    return {iid: sorted(rs) for iid, rs in sorted(reasons.items())}


def select_ids(frame: dict, exclusions: dict) -> list[str]:
    eligible = sorted(set(frame) - set(exclusions))
    if len(eligible) < N_INPUTS:
        raise ValueError(f"Only {len(eligible)} eligible inputs; need {N_INPUTS}")
    random.Random(SEED).shuffle(eligible)
    return sorted(eligible[:N_INPUTS])


def validate_run(inv: Inventory, run: str, wanted: set[str], inputs: dict,
                 n_generations: int = 1) -> dict[tuple[str, int], dict]:
    config = inv.json(f"{run}/run_config.json")
    method = config["method"]
    if config.get("mock") is not False or method["n_generations"] != n_generations:
        raise ValueError(f"Unexpected mock/generation configuration: {run}")
    prompt = inv.read(method["prompt_path"]).decode("utf-8")
    if sha256_hex(prompt) != method["prompt_sha256"]:
        raise ValueError(f"Historical prompt bytes changed: {run}")
    records = {}
    terminal = {"ok", "truncated", "invalid_json", "schema_error", "api_error", "api_permanent_error"}
    for path in sorted(local_path(inv.root, run).joinpath("responses").glob("*.json")):
        rel = path.relative_to(inv.root).as_posix()
        # Read metadata to find IDs, but lock only records used by E08/calibration.
        raw = inv.read(rel, track=False)
        record = json.loads(raw)
        task = record["task"]
        iid = task["input_id"]
        if iid not in wanted:
            continue
        inv.files[rel] = sha256_hex(raw)
        key = (iid, task["gen_index"])
        if key in records:
            raise ValueError(f"Ambiguous duplicate source task: {run}, {key}")
        if record.get("mock") is not False or record.get("status") not in terminal:
            raise ValueError(f"Not a completed real extraction task: {rel}")
        if type(task["gen_index"]) is not int or task["gen_index"] not in range(n_generations):
            raise ValueError(f"Unexpected generation index: {rel}")
        if task["input_hash"] != hash_obj({k: inputs[iid][k] for k in ("title", "paragraph_text")}):
            raise ValueError(f"Source paragraph mismatch: {rel}")
        for field, expected in (("model", config["access_model"]),
                                ("prompt_hash", method["prompt_sha256"]),
                                ("schema_id", method["schema_id"]),
                                ("gen_params", method["gen_params"])):
            if task[field] != expected:
                raise ValueError(f"Source {field} mismatch: {rel}")
        if task.get("route") != config.get("access_route") and "route" in task:
            raise ValueError(f"Source route mismatch: {rel}")
        if record["cache_key"] != path.stem:
            raise ValueError(f"Source cache filename mismatch: {rel}")
        if record["status"] == "ok":
            facts = (record.get("parsed") or {}).get("triplets")
            if not isinstance(facts, list):
                raise ValueError(f"Successful response has no parsed triplets: {rel}")
            for fact in facts:
                if not isinstance(fact, dict) or not all(isinstance(fact.get(k), str)
                                                       for k in ("subject", "relation", "object")):
                    raise ValueError(f"Invalid parsed fact: {rel}")
        records[key] = record
    expected = {(iid, g) for iid in wanted for g in range(n_generations)}
    if set(records) != expected:
        raise ValueError(f"Missing source tasks: {run}: {sorted(expected - set(records))}")
    return records


def original_facts(records: dict, iid: str) -> list[dict]:
    record = records[(iid, 0)]
    return record["parsed"]["triplets"] if record["status"] == "ok" else []


def validate_heuristics(inv: Inventory, selected: list[str], sources: dict) -> None:
    union = inv.json("runs/ensemble_xprompt/merged_union.json")
    vote = inv.json("runs/ensemble_xprompt/merged_vote2.json")
    for data, algorithm, threshold in ((union, "union_dedup", 1), (vote, "vote", 2)):
        if (data["algorithm"] != algorithm or data["algorithm_version"] != "agg-1.0"
                or data["params"] != {"vote_threshold": threshold,
                                     "similarity_threshold": 0.9, "per_input": True}):
            raise ValueError("Historical heuristic configuration mismatch")
    for iid in selected:
        originals = {sources[m][(iid, 0)]["cache_key"]: sources[m][(iid, 0)]
                     for m in ("coverage", "baseline_fixed", "fewshot")}
        expected = Counter({key: len(r["parsed"]["triplets"]) for key, r in originals.items()
                            if r["status"] == "ok" and r["parsed"]["triplets"]})
        observed = Counter()
        facts = union["results"].get(iid, [])
        for fact in facts:
            support = set()
            for source in fact["sources"]:
                key = source["cache_key"]
                if key not in originals or originals[key]["status"] != "ok":
                    raise ValueError(f"Heuristic uses a different extraction source: {iid}")
                task = originals[key]["task"]
                if any(source[k] != task[k] for k in ("input_id", "model", "prompt_hash", "gen_index")):
                    raise ValueError(f"Heuristic provenance mismatch: {iid}")
                observed[key] += 1
                support.add((source["model"], source["prompt_hash"], source["gen_index"]))
            if fact["n_support"] != len(support) or not support:
                raise ValueError(f"Heuristic support mismatch: {iid}")
        if observed != expected:
            raise ValueError(f"Heuristic does not cover the original candidate occurrences: {iid}")
        if vote["results"].get(iid, []) != [f for f in facts if f["n_support"] >= 2]:
            raise ValueError(f"Heuristic vote2 is not the fixed union subset: {iid}")


def calibration_files(inv: Inventory, inputs: dict, exclusions: dict,
                      selected: list[str], sources: dict) -> dict[str, bytes]:
    development = {c[0] for c in FACT_CASES} | {c[0] for c in PAIR_CASES}
    if len(FACT_CASES) != 24 or len(PAIR_CASES) != 24:
        raise ValueError("Calibration must contain 24 facts and 24 pairs")
    if development - set(exclusions) or development & set(selected):
        raise ValueError("Calibration must use excluded development inputs only")
    if any(inputs[iid]["title"] != CALIBRATION_TITLES[iid] for iid in development):
        raise ValueError("Curated calibration input/title mismatch")
    merged = inv.json(f"{V2_RUN}/merged_union.json")
    if merged["params"]["prompt_hash"] != V2_PROMPT_HASH:
        raise ValueError("Development merger prompt identity changed")
    payloads, refs = {}, {}
    for iid in sorted(development):
        rows = merged["results"].get(iid)
        if not rows:
            raise ValueError(f"Missing development merger result: {iid}")
        keys = {r["dedup_cache_key"] for r in rows}
        if len(keys) != 1:
            raise ValueError(f"Ambiguous development merger result: {iid}")
        key = keys.pop()
        rel = f"{V2_RUN}/responses/{key}.json"
        record = inv.json(rel)
        payload = record["payload"]
        expected = [fact for m in ("coverage", "baseline_fixed", "fewshot")
                    for fact in original_facts(sources[m], iid)]
        if (record["input_id"] != iid or record["cache_key"] != key or record["status"] != "ok"
                or payload["title"] != inputs[iid]["title"]
                or payload["paragraph_text"] != inputs[iid]["paragraph_text"]
                or payload["candidates"] != [{"id": n, "fact": f} for n, f in enumerate(expected)]):
            raise ValueError(f"Development payload differs from original candidates: {iid}")
        payloads[iid] = payload
        refs[iid] = rel

    def context(iid):
        return {k: payloads[iid][k] for k in ("title", "paragraph_text")}

    def fact(iid, cid):
        return payloads[iid]["candidates"][cid]["fact"]

    public_facts, public_pairs, private_facts, private_pairs = [], [], [], []
    for iid, cid in FACT_CASES:
        f = fact(iid, cid)
        aid = hash_obj({"phase": "e08_calibration_fact", "input_id": iid, "fact": f})
        public_facts.append({"annotation_id": aid, **context(iid), "fact": f})
        private_facts.append({"annotation_id": aid, "input_id": iid, "candidate_id": cid,
                              "source_path": refs[iid]})
    for iid, a, b in PAIR_CASES:
        aid = hash_obj({"phase": "e08_calibration_pair", "input_id": iid, "pair": [a, b]})
        public_pairs.append({"audit_id": aid, **context(iid), "a": fact(iid, a), "b": fact(iid, b)})
        private_pairs.append({"audit_id": aid, "input_id": iid, "pair": [a, b],
                              "source_path": refs[iid]})
    for rows, key in ((public_facts, "annotation_id"), (public_pairs, "audit_id")):
        if len({row[key] for row in rows}) != 24:
            raise ValueError("Duplicate calibration items")
        random.Random(SEED).shuffle(rows)

    binary_task = inv.read("data/judging/ANNOTATION_TASK.md").decode("utf-8")
    binary_task = binary_task.replace("data/judging/round1/", f"{CALIBRATION_DIR}/binary/")
    binary_task = binary_task.replace(
        "Unlike AutoOntik's isolated-sentence groundedness protocol, title-based\nsubject identification",
        "Title-based subject identification")
    pair_task = inv.read("data/judging/MERGER_AUDIT_TASK.md").decode("utf-8")
    pair_task = pair_task.replace("data/judging/merger_audit/", f"{CALIBRATION_DIR}/merger_audit/")
    heading = ("# E08 rubric calibration — not the main evaluation\n\n"
               "Use a fresh annotation session. These are development examples. "
               "Do not inspect any private manifest, report, source code, or previous judgments.\n\n")
    tasks = {}
    for name, text in (("binary", binary_task), ("merger_audit", pair_task)):
        text = text.replace("/home/alexey/PycharmProjects/Wikontic_experiments", str(inv.root))
        tasks[f"{CALIBRATION_DIR}/{name}/TASK.md"] = (heading + text).encode("utf-8")
    return {
        **tasks,
        f"{CALIBRATION_DIR}/binary/candidates.jsonl": jsonl_bytes(public_facts),
        f"{CALIBRATION_DIR}/merger_audit/pairs.jsonl": jsonl_bytes(public_pairs),
        f"{CALIBRATION_DIR}/calibration_private.json": json_bytes({
            "phase": "calibration_only", "seed": SEED,
            "facts": private_facts, "pairs": private_pairs,
            "note": "Curated coverage of failure modes, not a representative quality sample. No expected labels."}),
    }


def prepare(root: Path, extra_exclusions: str | None = None) -> dict:
    root = root.resolve()
    for folder in (EXPERIMENT_DIR, CALIBRATION_DIR):
        path = local_path(root, folder)
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise ValueError(f"Existing work is not overwritten: {folder}; use verify")
    inv = Inventory(root)
    frame = input_map(inv.jsonl("data/tuning/subset300_inputs.jsonl"))
    inputs = input_map(inv.jsonl("data/tuning/inputs.jsonl"))
    if len(frame) != 300:
        raise ValueError("Expected the preserved C2 frame of 300 inputs")
    for iid, row in frame.items():
        if iid not in inputs or any(row[k] != inputs[iid][k] for k in ("title", "paragraph_text")):
            raise ValueError(f"C2/tuning input mismatch: {iid}")
    discovered = inv.discovery()
    exclusions = collect_exclusions(inv, discovered, extra_exclusions)
    selected = select_ids(frame, exclusions)
    development = {c[0] for c in FACT_CASES} | {c[0] for c in PAIR_CASES}
    if development - set(inputs):
        raise ValueError("Calibration source paragraph missing")
    if hash_obj(inv.read(V2_PROMPT).decode("utf-8")) != V2_PROMPT_HASH:
        raise ValueError("Frozen merger v2 prompt changed")
    # Selection is complete BEFORE looking at statuses/candidate content.
    sources = {name: validate_run(inv, run, set(selected) | development, inputs)
               for name, run in RUNS.items() if name != "autoontik"}
    sources["autoontik"] = validate_run(inv, RUNS["autoontik"], set(selected), inputs)
    validate_run(inv, "runs/gptoss_ens3_300", set(frame), inputs, n_generations=3)
    validate_heuristics(inv, selected, sources)
    files = calibration_files(inv, inputs, exclusions, selected, sources)
    selected_inputs = []
    for iid in selected:
        row = inputs[iid]
        selected_inputs.append({"input_id": iid, "title": row["title"],
                                "paragraph_text": row["paragraph_text"],
                                "refs": [{k: r[k] for k in ("sample_id", "paragraph_idx") if k in r}
                                         for r in row.get("refs", [])]})
    source_tasks = {
        name: [{"input_id": iid, "cache_key": records[(iid, 0)]["cache_key"],
                "status": records[(iid, 0)]["status"],
                "n_facts": len(original_facts(records, iid))} for iid in selected]
        for name, records in sources.items()
    }
    selection = {"experiment": "E08", "seed": SEED, "input_ids": selected,
                 "frame_size": len(frame), "eligible_size": len(set(frame) - set(exclusions)),
                 "algorithm": "sorted eligible IDs; Random(42).shuffle; first 50; sort"}
    files.update({
        f"{EXPERIMENT_DIR}/selection.json": json_bytes(selection),
        f"{EXPERIMENT_DIR}/inputs.jsonl": jsonl_bytes(selected_inputs),
        f"{EXPERIMENT_DIR}/exclusions.json": json_bytes({"reasons_by_input_id": exclusions}),
        f"{EXPERIMENT_DIR}/preflight.json": json_bytes({
            "source_tasks": source_tasks, "selected": len(selected),
            "calibration_facts": 24, "calibration_pairs": 24,
            "tests": "not run by this command; run pytest separately",
            "next": "human-reviewed rubric calibration; no model requests made"}),
    })
    # Only explicit scientific/code inputs; no .env, access configs, credentials or QA sidecar.
    code_paths = sorted(p.relative_to(root).as_posix()
                        for folder in ("src/wikontic_exp", "tests")
                        for p in (root / folder).glob("*.py"))
    for rel in ["reports/E08_PROTOCOL.md", "schemas/facts_v1.json", "pyproject.toml",
                "scripts/prepare_merger_audit.py", "scripts/score_merger_audit.py",
                *code_paths]:
        inv.read(rel)
    lock = {
        "experiment": "E08", "phase": "selection_and_initial_calibration",
        "python": sys.version, "implementation": platform.python_implementation(),
        "source_files_sha256": dict(sorted(inv.files.items())),
        "discovered_files": discovered,
        "generated_files_sha256": {p: sha256_hex(data) for p, data in sorted(files.items())},
        "merger": {"model": "openai/gpt-oss-120b", "provider": "openrouter",
                   "prompt_hash": V2_PROMPT_HASH, "temperature": 0, "max_tokens": 6000},
        "rubric": "initial tasks only; human calibration and final judge identity pending",
        "source_tasks": source_tasks,
    }
    # Catch changed sources/discovery before writing anything. Generated files are immutable.
    if inv.discovery() != discovered:
        raise ValueError("Exposure files changed during preparation")
    for rel, digest in inv.files.items():
        if file_sha256(str(local_path(root, rel))) != digest:
            raise ValueError(f"Source changed during preparation: {rel}")
    files[f"{EXPERIMENT_DIR}/lock.json"] = json_bytes(lock)
    for rel in files:
        if local_path(root, rel).exists():
            raise ValueError(f"Existing work is not overwritten: {rel}")
    for rel, data in files.items():
        path = local_path(root, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    return {"selected": len(selected), "calibration_facts": 24, "calibration_pairs": 24,
            "lock": f"{EXPERIMENT_DIR}/lock.json", "model_calls": 0}


def verify(root: Path) -> dict:
    root = root.resolve()
    lock = json.loads(local_path(root, f"{EXPERIMENT_DIR}/lock.json").read_bytes())
    if lock.get("experiment") != "E08" or lock.get("phase") != "selection_and_initial_calibration":
        raise ValueError("Not an E08 preparation lock")
    if Inventory(root).discovery() != lock["discovered_files"]:
        raise ValueError("Exposure discovery changed; do not regenerate selection silently")
    for group in ("source_files_sha256", "generated_files_sha256"):
        for rel, digest in lock[group].items():
            if file_sha256(str(local_path(root, rel))) != digest:
                raise ValueError(f"Lock mismatch: {rel}")
    return {"status": "verified", "phase": lock["phase"],
            "note": "No judgments/model identity verified; no files updated", "model_calls": 0}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "verify"))
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--extra-exclusions", help="Project-relative JSON list of {input_id, reason}; before lock only")
    args = p.parse_args()
    if args.command == "verify" and args.extra_exclusions:
        p.error("Extra exclusions cannot be added after lock")
    try:
        result = prepare(args.root, args.extra_exclusions) if args.command == "prepare" else verify(args.root)
    except (ValueError, KeyError, OSError, TypeError, IndexError) as exc:
        p.exit(1, f"E08 preparation stopped: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
