"""Offline finalization under the API runner lock; no credentials or inference."""
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
spec = importlib.util.spec_from_file_location("e08_finalize_api", ROOT / "scripts/e08_judge_api.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


def create(path, obj):
    if path.exists():
        if json.loads(path.read_bytes()) != obj:
            raise ValueError(f"Existing final artifact differs: {path.name}")
        return
    with path.open("x", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def main():
    with api.exclusive_run(ROOT):
        plan, jobs = api.verify_plan(ROOT)
        state = api.scan(ROOT, jobs)
        if state["must_stop"]:
            api.halt(ROOT, "Finalization detected uncertain attempt or transport/identity failure; no POST permitted")
        status = api.summary(plan, state, ROOT)
        if state["states"].get("pending") and not status["halted"]:
            raise ValueError("Execution not finished or halted; finalization forbidden")
        # Offline export may reconstruct missing lines from receipts, never relabel.
        api.sync_outputs(ROOT, state)
        inputs = [f"{api.stage.MAIN}/{kind}/judgments.jsonl" for kind in api.KINDS]
        inputs += [f"{api.stage.MAIN}/binary/sampling_private.json",
                   f"{api.stage.MAIN}/merger_audit/audit_private.json"]
        inputs += ["scripts/e08_main.py", "scripts/e08_judge_api.py", "scripts/score_merger_audit.py",
                   "src/wikontic_exp/binary_judging.py", f"{api.API_DIR}/plan.json", f"{api.API_DIR}/requests.jsonl"]
        inputs += [str(p.relative_to(ROOT)) for p in (ROOT / api.API_DIR).rglob("*.json")
                   if str(p.relative_to(ROOT)) not in inputs]
        inputs += [str((RUN / n).relative_to(ROOT)) for n in
                   ("smoke_check.json", "build_report.py", "finalize_results.py")]
        hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in inputs}
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
        with tempfile.TemporaryDirectory(prefix="e08_score_", dir=RUN) as temp:
            for n, cmd in enumerate(commands):
                actual = [*cmd[:-1], str(Path(temp) / Path(cmd[-1]).name)]
                with (RUN / "commands.log").open("a") as f:
                    f.write("Offline score scratch then exclusive-create/compare: " + " ".join(actual) + "\n")
                with (RUN / f"scorer_{n}_{uuid4().hex}.log").open("x") as f:
                    subprocess.run(actual, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
                create(Path(cmd[-1]), json.loads(Path(actual[-1]).read_bytes()))
        for p, digest in hashes.items():
            if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != digest:
                raise ValueError("Scoring inputs changed during finalization")
        create(RUN / "api_final_status.json", status)
        for name in ("binary_precision.json", "merger_audit.json", "api_final_status.json"):
            hashes[str((RUN / name).relative_to(ROOT))] = hashlib.sha256((RUN / name).read_bytes()).hexdigest()
        create(RUN / "score_provenance.json", {"created_utc": snapshot["created_utc"],
               "files_sha256": hashes, "commands": commands,
               "policy": "Existing scorers, current retained labels, no resampling or relabeling"})
        subprocess.run([sys.executable, str(RUN / "build_report.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
