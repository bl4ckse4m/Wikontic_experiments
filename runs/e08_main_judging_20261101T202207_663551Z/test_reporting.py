"""Synthetic-only regression checks for report assembly; no live API or labels."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("e08_reporting_test", Path(__file__).with_name("build_report.py"))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class ReportingTests(unittest.TestCase):
    def test_partial_results_are_reused_only_if_identical(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "partial.json"
            report.create(path, {"score": 0.5})
            before = path.read_bytes()
            report.create(path, {"score": 0.5})
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "differs"):
                report.create(path, {"score": 1})
            self.assertEqual(path.read_bytes(), before)

    def test_nested_and_flat_resources(self):
        result = report.usage([
            {"attempts": 2, "latency_s": 1.5, "usage": {"prompt_tokens": 3, "completion_tokens": 4}},
            {"response": {"attempts": 1, "latency_s": 2, "raw_response": {"usage": {"cost": 0.1}}}},
            {}])
        self.assertEqual(result["recorded_attempts"], 3)
        self.assertEqual(result["attempts_records"], 2)
        self.assertEqual(result["recorded_latency_s"], 3.5)
        self.assertEqual(result["latency_records"], 2)
        self.assertEqual(result["prompt_tokens"], 3)
        self.assertEqual(result["cost_records"], 1)
        self.assertIsNone(report.usage([{}])["reported_cost_usd"])

    def test_score_provenance_detects_stale_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = [f"{report.stage.MAIN}/{kind}/judgments.jsonl" for kind in ("binary", "merger_audit")]
            paths += ["binary_precision.json", "merger_audit.json", "api_final_status.json"]
            for rel in paths:
                p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("synthetic")
            (root / "score_provenance.json").write_text(json.dumps({"files_sha256": {
                rel: hashlib.sha256((root / rel).read_bytes()).hexdigest() for rel in paths}}))
            with patch.object(report, "ROOT", root), patch.object(report, "RUN", root):
                report.verify_score_provenance()
                (root / paths[0]).write_text("changed")
                with self.assertRaisesRegex(ValueError, "Stale score"):
                    report.verify_score_provenance()

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


if __name__ == "__main__":
    unittest.main()
