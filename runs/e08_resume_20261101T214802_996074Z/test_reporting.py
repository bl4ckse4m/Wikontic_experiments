"""Synthetic continuation report tests only; never score or access the API."""
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

RUN = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location("e08_resumed_test_" + name, RUN / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = load("build_report")
finalize = load("finalize_results")


class ReportingTests(unittest.TestCase):
    def test_hash_idempotent_and_exclusive_outputs(self):
        for module in (report, finalize):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory(dir=RUN) as folder:
                path = Path(folder) / "synthetic.json"
                module.create(path, {"score": 0.5})
                before = path.read_bytes()
                module.create(path, {"score": 0.5})
                with self.assertRaisesRegex(ValueError, "differs"):
                    module.create(path, {"score": 1})
                self.assertEqual(path.read_bytes(), before)
                path.write_bytes(b'{"score": 0.5}')
                with self.assertRaisesRegex(ValueError, "differs"):
                    module.create(path, {"score": 0.5})

    def test_nested_flat_and_unknown_resources(self):
        result = report.usage([
            {"attempts": 2, "latency_s": 1.5, "usage": {"prompt_tokens": 3, "completion_tokens": 4}},
            {"response": {"attempts": 1, "latency_s": 2, "raw_response": {"usage": {"cost": 0.1}}}}, {}])
        self.assertEqual(result["recorded_attempts"], 3)
        self.assertEqual(result["recorded_latency_s"], 3.5)
        self.assertEqual(result["prompt_tokens"], 3)
        self.assertEqual(result["cost_records"], 1)
        self.assertIsNone(report.usage([{}])["reported_cost_usd"])

    def test_counts_withhold_incomplete_scores_and_reject_stale_status(self):
        private = {"methods": {"synthetic": {"population": 10, "samples": [{"annotation_id": "a"}]}}}
        binary = {"methods": {"synthetic": {"population": 10, "sampled": 1, "missing": 1}}}
        pairs = {"strata": {name: {"sampled": 1, "missing": 1} for name in ("proposed_merge", "proposed_separate")}}
        manifest = {"pairs": [{"audit_id": "p", "llm_same": True}, {"audit_id": "q", "llm_same": False}]}
        api = {"labels": {"binary": 0, "merger_audit": 0}}
        verification = {"judgments": {k: {"present": 0} for k in api["labels"]}}
        with patch.object(report.stage, "rows", return_value=[]), patch.object(report, "read", return_value=manifest):
            report.verify_score_counts(binary, pairs, api, private, verification)
            binary["methods"]["synthetic"]["full_label"] = {"precision": 1}
            with self.assertRaisesRegex(ValueError, "withholding mismatch"):
                report.verify_score_counts(binary, pairs, api, private, verification)
            api["labels"]["binary"] = 1
            with self.assertRaisesRegex(ValueError, "API status differs"):
                report.verify_score_counts(binary, pairs, api, private, verification)

    def test_contrasts_are_descriptive_and_withhold_missing(self):
        methods = {name: {"quality": {"full_label": {"precision": value}}} for name, value in
                   (("autoontik", .8), ("baseline_fixed", .9), ("llm_union", .7), ("heuristic_union", .6))}
        contrasts = report.primary_contrasts(methods)
        self.assertAlmostEqual(contrasts[0]["full_label_precision_difference"], -.1)
        self.assertAlmostEqual(contrasts[1]["full_label_precision_difference"], .1)
        self.assertIsNone(contrasts[0]["core_label_precision_difference"])
        self.assertIn("no superiority", contrasts[0]["interpretation"])

    def evidence_fixture(self, folder):
        base, continuation = Path(folder) / "base", Path(folder) / "continuation"
        for parent, count in ((base, 70), (continuation, 31)):
            for child in ("records", "key_checks"):
                (parent / child).mkdir(parents=True)
            for i in range(count):
                (parent / "records" / f"{i}.attempt.json").write_text("{}")
                (parent / "key_checks" / f"{i}.json").write_text("{}")
        api = {"continuation": {"original_http_attempts": 70, "new_http_attempts": 31,
                               "retry_job_id": "retry", "previous_failure_cost_unknown": True},
               "missing_cost_jobs": ["prior_attempt/retry"], "jobs": {"ok": 100}}
        return base, continuation, api

    def test_100_latest_successes_retain_101_attempts_and_unknown_cost(self):
        with tempfile.TemporaryDirectory(dir=RUN) as folder:
            base, continuation, api = self.evidence_fixture(folder)
            with patch.object(report, "BASE_API", base), patch.object(report, "CONTINUATION", continuation):
                result = report.execution_evidence(api)
                self.assertEqual(result["total_http_attempts"], 101)
                self.assertTrue(result["previous_failure_cost_unknown"])
                api["missing_cost_jobs"] = []
                with self.assertRaisesRegex(ValueError, "unknown-cost"):
                    report.execution_evidence(api)

    def test_latest_success_cannot_hide_failed_http_history(self):
        with tempfile.TemporaryDirectory(dir=RUN) as folder:
            base, continuation, api = self.evidence_fixture(folder)
            (continuation / "records/30.attempt.json").unlink()
            api["continuation"]["new_http_attempts"] = 30
            with patch.object(report, "BASE_API", base), patch.object(report, "CONTINUATION", continuation):
                with self.assertRaisesRegex(ValueError, "101 retained"):
                    report.execution_evidence(api)

    def test_score_provenance_requires_continuation_history(self):
        with tempfile.TemporaryDirectory(dir=RUN) as folder:
            root = Path(folder)
            (root / "score_provenance.json").write_text(json.dumps({"files_sha256": {}}))
            with patch.object(report, "ROOT", root), patch.object(report, "RUN", root), \
                    patch.object(report, "BASE_API", root / "base"), \
                    patch.object(report, "BASE_RUN", root / "original"), \
                    patch.object(report, "CONTINUATION", root / "continuation"):
                with self.assertRaisesRegex(ValueError, "continuation history"):
                    report.verify_score_provenance()

    def test_score_provenance_detects_stale_inputs(self):
        with tempfile.TemporaryDirectory(dir=RUN) as folder:
            root = Path(folder)
            (root / "labels.jsonl").write_text("changed")
            (root / "score_provenance.json").write_text(json.dumps({"files_sha256": {
                "labels.jsonl": hashlib.sha256(b"original").hexdigest()}}))
            with patch.object(report, "ROOT", root), patch.object(report, "RUN", root):
                with self.assertRaisesRegex(ValueError, "Stale score"):
                    report.verify_score_provenance()

    def test_attach_uses_relative_continuation_identifier(self):
        resume = SimpleNamespace(attach=Mock(return_value="synthetic adapter"))
        spec = SimpleNamespace(loader=SimpleNamespace(exec_module=Mock()))
        with patch.object(finalize.importlib.util, "spec_from_file_location", return_value=spec), \
                patch.object(finalize.importlib.util, "module_from_spec", return_value=resume):
            self.assertEqual(finalize.attach_api(), "synthetic adapter")
        resume.attach.assert_called_once_with(finalize.ROOT, finalize.CONTINUATION.relative_to(finalize.ROOT).as_posix())

    def test_finalization_refuses_pending_or_uncertain_without_scoring(self):
        for state in ({"must_stop": False, "states": {"pending": 1}},
                      {"must_stop": True, "states": {"uncertain": 1}}):
            api = SimpleNamespace(exclusive_run=Mock(return_value=nullcontext()),
                                  verify_plan=Mock(return_value=({}, [])), scan=Mock(return_value=state),
                                  summary=Mock(return_value={"halted": False}), sync_outputs=Mock())
            with patch.object(finalize, "attach_api", return_value=api), \
                    patch.object(finalize.subprocess, "run") as subprocess_run:
                with self.assertRaises(ValueError):
                    finalize.main()
                subprocess_run.assert_not_called()
                api.sync_outputs.assert_not_called()

    def test_provenance_hashes_both_ledgers_and_continuation_sources(self):
        with tempfile.TemporaryDirectory(dir=RUN) as folder:
            root = Path(folder)
            run, base_api, continuation = root / "run", root / "base", root / "run/api_judging"
            original = root / "original"
            paths = ["main/binary/judgments.jsonl", "main/merger_audit/judgments.jsonl",
                     "main/binary/sampling_private.json", "main/merger_audit/audit_private.json",
                     "scripts/e08_main.py", "scripts/e08_judge_api.py", "scripts/e08_resume_api.py",
                     "scripts/score_merger_audit.py", "src/wikontic_exp/binary_judging.py",
                      "run/initial_manifest.json", "run/smoke_check.json", "run/resume_smoke_check.json", "run/build_report.py",
                     "run/finalize_results.py", "original/smoke_check.json", "amendment.md",
                     "base/halt.json", "base/requests.jsonl", "base/records/old.response.json",
                     "run/api_judging/records/new.attempt.json", "run/api_judging/requests.jsonl"]
            for rel in paths:
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("synthetic")
            (base_api / "plan.json").write_text("{}")
            (continuation / "plan.json").write_text(json.dumps({"source_files_sha256": {"amendment.md": "synthetic"}}))
            api = SimpleNamespace(stage=SimpleNamespace(MAIN="main"), KINDS=("binary", "merger_audit"))
            with patch.object(finalize, "ROOT", root), patch.object(finalize, "RUN", run), \
                    patch.object(finalize, "BASE_API", base_api), \
                    patch.object(finalize, "CONTINUATION", continuation), patch.object(finalize, "BASE_RUN", original):
                hashes = finalize.provenance_inputs(api)
                self.assertTrue(set(paths) <= hashes.keys())
                (continuation / "records/new.attempt.json").write_text("changed")
                self.assertNotEqual(hashes, finalize.provenance_inputs(api))


if __name__ == "__main__":
    with patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")), \
            patch.object(socket, "getaddrinfo", side_effect=AssertionError("DNS forbidden")), \
            patch.object(socket.socket, "sendto", side_effect=AssertionError("Network forbidden")):
        unittest.main()
