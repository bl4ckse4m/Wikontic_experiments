"""Offline continuation finalization under the runner lock; no API or credentials.

Adapted from the original run, using its unchanged binary/pair scorers.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from uuid import uuid4

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
BASE_RUN = ROOT / "runs/e08_main_judging_20261101T202207_663551Z"
BASE_API = ROOT / "data/experiments/e08/api_judging"
CONTINUATION = RUN / "api_judging"


def create(path, obj):
    data = (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if path.exists():
        if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(data).digest():
            raise ValueError(f"Existing final artifact differs: {path.name}")
        return
    with path.open("xb") as stream:
        stream.write(data)


def attach_api():
    spec = importlib.util.spec_from_file_location("e08_finalize_resume", ROOT / "scripts/e08_resume_api.py")
    resume = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resume)
    return resume.attach(ROOT, CONTINUATION.relative_to(ROOT).as_posix())


def provenance_inputs(api):
    inputs = {f"{api.stage.MAIN}/{kind}/judgments.jsonl" for kind in api.KINDS}
    inputs.update({f"{api.stage.MAIN}/binary/sampling_private.json",
                   f"{api.stage.MAIN}/merger_audit/audit_private.json",
                   "scripts/e08_main.py", "scripts/e08_judge_api.py", "scripts/e08_resume_api.py",
                   "scripts/score_merger_audit.py", "src/wikontic_exp/binary_judging.py"})
    # The original plan/requests/halt/receipts remain history, never rewritten.
    # Include the continuation history snapshot and every new sanitized ledger record.
    for folder in (BASE_API, CONTINUATION):
        inputs.update(str(p.relative_to(ROOT)) for p in folder.rglob("*")
                      if p.is_file() and p.suffix in (".json", ".jsonl"))
    for folder in (BASE_API, CONTINUATION):
        plan = json.loads((folder / "plan.json").read_bytes())
        inputs.update(plan.get("source_files_sha256", {}))
    inputs.update(str((RUN / name).relative_to(ROOT)) for name in
                   ("initial_manifest.json", "smoke_check.json", "resume_smoke_check.json", "build_report.py", "finalize_results.py"))
    inputs.add(str((BASE_RUN / "smoke_check.json").relative_to(ROOT)))
    inputs.update(str(p.relative_to(ROOT)) for p in (RUN / "original_label_snapshots").glob("*") if p.is_file())
    inputs.update(str((RUN / name).relative_to(ROOT)) for name in ("key_status_after.json", "http_resources.json")
                  if (RUN / name).exists())
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sorted(inputs)}


def main():
    api = attach_api()
    with api.exclusive_run(ROOT):
        plan, jobs = api.verify_plan(ROOT)
        state = api.scan(ROOT, jobs)
        status = api.summary(plan, state, ROOT)
        if state["must_stop"]:
            raise ValueError("Uncertain attempt or transport/identity failure; inspect continuation status before finalizing")
        if state["states"].get("pending") and not status["halted"]:
            raise ValueError("Execution not finished or halted; finalization forbidden")
        # Only reconstruct labels from saved receipts; preserve every existing row.
        api.sync_outputs(ROOT, state)
        hashes = provenance_inputs(api)
        snapshot_path = RUN / "finalization_inputs.json"
        if snapshot_path.exists():
            snapshot = json.loads(snapshot_path.read_bytes())
            if snapshot["files_sha256"] != hashes or snapshot["api_status"] != status:
                raise ValueError("Finalization inputs changed; do not reuse partial results")
        else:
            snapshot = {"created_utc": datetime.now(timezone.utc).isoformat(),
                        "files_sha256": hashes, "api_status": status}
            create(snapshot_path, snapshot)
        commands = [
            [sys.executable, "-m", "wikontic_exp.binary_judging", "score", "--manifest",
             f"{api.stage.MAIN}/binary/sampling_private.json", "--judgments",
             f"{api.stage.MAIN}/binary/judgments.jsonl", "--out", str(RUN / "binary_precision.json")],
            [sys.executable, "scripts/score_merger_audit.py", "--manifest",
             f"{api.stage.MAIN}/merger_audit/audit_private.json", "--judgments",
             f"{api.stage.MAIN}/merger_audit/judgments.jsonl", "--out", str(RUN / "merger_audit.json")]]
        with tempfile.TemporaryDirectory(prefix="e08_resumed_score_", dir=RUN) as temp:
            for n, cmd in enumerate(commands):
                actual = [*cmd[:-1], str(Path(temp) / Path(cmd[-1]).name)]
                with (RUN / "commands.log").open("a", encoding="utf-8") as stream:
                    stream.write("Offline score scratch then exclusive-create/hash-compare: " + " ".join(actual) + "\n")
                with (RUN / f"scorer_{n}_{uuid4().hex}.log").open("x") as stream:
                    subprocess.run(actual, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
                create(Path(cmd[-1]), json.loads(Path(actual[-1]).read_bytes()))
        if provenance_inputs(api) != hashes:
            raise ValueError("Scoring inputs or ledger changed during finalization")
        create(RUN / "api_final_status.json", status)
        for name in ("binary_precision.json", "merger_audit.json", "api_final_status.json"):
            hashes[str((RUN / name).relative_to(ROOT))] = hashlib.sha256((RUN / name).read_bytes()).hexdigest()
        create(RUN / "score_provenance.json", {"created_utc": snapshot["created_utc"],
               "files_sha256": hashes, "commands": commands,
               "policy": "Unchanged scorers; merged retained labels; original failed attempt and unknown cost retained; no resampling or relabeling"})
        subprocess.run([sys.executable, str(RUN / "build_report.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
