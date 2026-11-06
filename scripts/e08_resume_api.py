"""Explicit single continuation of E08, preserving the frozen runner and ledger.

Only run calls the API. One specific label-less admission429 may be attempted
once again; original successes and any new uncertain attempt are never replayed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import time

BASE_API = "data/experiments/e08/api_judging"
REGISTRY = "data/experiments/e08/api_resume_authorization.json"
AMENDMENT = "research/experiments/E08_resume_20261102.md"
DEFAULT_CONTINUATION = "runs/e08_resume_20261101T214802_996074Z/api_judging"


def load_api():
    spec = importlib.util.spec_from_file_location("e08_frozen_resume", Path(__file__).with_name("e08_judge_api.py"))
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    return api


def admission_retry(api, root, job):
    """Narrow allowlist, never apply to timeouts, provider errors or model answers."""
    attempt, response = api.record_paths(job)
    receipt = api.stage.read(root, response)
    raw = receipt.get("response")
    error = raw.get("error") if isinstance(raw, dict) else None
    meta = error.get("metadata") if isinstance(error, dict) else None
    if (receipt.get("status_code") != 429 or receipt.get("error")
            or not isinstance(raw, dict) or set(raw) != {"error"}
            or not isinstance(error, dict) or error.get("code") != 429
            or "could not verify available credits" not in error.get("message", "")
            or not isinstance(meta, dict) or meta.get("limit_source") != "openrouter_admission_control"
            or meta.get("provider_name") is not None):
        raise ValueError("Only the observed label-less admission-control429 is retryable")
    after = int(meta.get("headers", {}).get("Retry-After", "10"))
    if after < 0 or after > 3600:
        raise ValueError("Unrecognized Retry-After")
    started = api.stage.read(root, attempt)["started_unix"]
    not_before = started + receipt.get("latency_s", 0) + after
    if time.time() < not_before:
        raise ValueError("Original Retry-After has not elapsed; no request permitted")
    return receipt


def prepare(root: Path, continuation: str, key_status_file: str):
    api = load_api()
    if api.stage.path(root, f"{continuation}/plan.json").exists():
        attached = attach(root, continuation)
        plan, jobs = attached.verify_plan(root)
        return {"status": "continuation already prepared", "model_calls": 0, "jobs": len(jobs)}
    base_plan, jobs = api.verify_plan(root)
    base_state = api.scan(root, jobs)
    if base_state["states"].get("uncertain") or base_state["states"].get("failed") != 1:
        raise ValueError("Expected exactly one known failure, no uncertain attempts")
    failed = [j for j in jobs if j["job_id"] in base_state["errors"]]
    if len(failed) != 1 or not api.stage.path(root, f"{BASE_API}/halt.json").exists():
        raise ValueError("Missing original terminal failure/halt")
    retry = failed[0]
    admission_retry(api, root, retry)
    eligible = [j["job_id"] for j in jobs if j == retry or not api.stage.path(root, api.record_paths(j)[0]).exists()]
    if eligible[0] != retry["job_id"]:
        raise ValueError("Retry must precede all unattempted jobs in the fixed order")
    snapshot = api.key_accounting(api.stage.read(root, key_status_file))
    remaining = api.remaining_credit(snapshot)
    safety = base_plan["operational_key_safety"]
    if not safety or remaining <= api.money(safety["margin_usd"]):
        raise ValueError("No safe operational key credit")
    if api.stage.path(root, REGISTRY).exists():
        raise ValueError("An E08 continuation is already authorized; no additional retry segment permitted")
    authorization = {"continuation": continuation, "retry_job_id": retry["job_id"],
                     "amendment": AMENDMENT, "authority": "Owner explicitly instructed continuation after stop report",
                     "max_additional_http_attempts": len(eligible), "max_retries_of_original_429": 1}
    api.durable_record(root, REGISTRY, authorization)
    history = [p.relative_to(root).as_posix() for p in api.stage.path(root, BASE_API).rglob("*.json")]
    history.append(f"{BASE_API}/requests.jsonl")
    sources = ["scripts/e08_resume_api.py", "tests/test_e08_resume_api.py", AMENDMENT, REGISTRY, key_status_file]
    requests = api.stage.e08_preparation.jsonl_bytes(jobs)
    prior = api.stage.read(root, api.record_paths(retry)[1])
    usage = prior.get("response", {}).get("usage", {})
    cost = usage.get("cost") if isinstance(usage, dict) else None
    historical_debit = api.money(cost) if cost is not None else api.money(retry["reservation_usd"])
    selected = [j for j in jobs if j["job_id"] in eligible]
    chars = sum(len(api.canonical_json(j["request"])) for j in selected)
    max_output = sum(j["request"]["max_tokens"] for j in selected)
    scenario_output = sum(min(j["request"]["max_tokens"], len(j["ids"]) * (140 if j["kind"] == "binary" else 110) + 512) for j in selected)
    plan = {"version": "e08-explicit-continuation-1", **authorization,
            "base_api_dir": BASE_API, "base_workload": base_plan["workload"],
            "source_files_sha256": api.stage.hashes(root, sources),
            "history_files_sha256": api.stage.hashes(root, history),
            "requests_sha256": api.sha256_hex(requests), "eligible_job_ids": eligible,
            "original_http_attempts": sum(v for k,v in base_state["states"].items() if k != "pending"),
            "historical_failure_debit_usd": str(historical_debit), "historical_failure_cost_unknown": cost is None,
            "initial_operational_safety": deepcopy(safety), "fresh_key_accounting": snapshot,
            "forecast": {"new_requests": len(eligible), "input_chars": chars, "max_output_tokens": max_output,
                         "scenario_usd_est": str((Decimal(chars)/4*Decimal('0.000002') + scenario_output*Decimal('0.00001')).quantize(Decimal('0.01'))),
                         "max_output_input_chars_div3_usd_est": str((Decimal(chars)/3*Decimal('0.000002') + max_output*Decimal('0.00001')).quantize(Decimal('0.01'))),
                         "not_a_billing_guarantee": True},
            "policy": "One explicitly authorized retry of original admission429; remaining original pending jobs once. No new-attempt retries, fallback or successful-ID rejudging. Original unknown failed cost retained in cumulative accounting; original allowance never reset."}
    api.stage.create_files(root, {f"{continuation}/plan.json": api.stage.e08_preparation.json_bytes(plan),
                                 f"{continuation}/requests.jsonl": requests})
    attach(root, continuation).verify_plan(root)
    return {"status": "explicit continuation frozen", "model_calls": 0, **authorization, "forecast": plan["forecast"]}


def attach(root: Path, continuation: str):
    """Isolated adapter over frozen functions; original module/file is unchanged."""
    api = load_api()
    original_verify, original_scan, original_summary = api.verify_plan, api.scan, api.summary
    original_paths = api.record_paths
    cp = api.stage.read(root, f"{continuation}/plan.json")
    eligible = set(cp["eligible_job_ids"])

    def verify(where):
        previous = api.API_DIR
        previous_paths = api.record_paths
        api.API_DIR = BASE_API
        api.record_paths = original_paths
        try:
            base_plan, jobs = original_verify(where)
            base_state = original_scan(where, jobs)
            retry_jobs = [j for j in jobs if j["job_id"] == cp["retry_job_id"]]
            if len(retry_jobs) != 1 or list(base_state["errors"]) != [cp["retry_job_id"]]:
                raise ValueError("Original failure identity or history changed")
            receipt = admission_retry(api, where, retry_jobs[0])
            expected = [j["job_id"] for j in jobs if j == retry_jobs[0]
                        or not api.stage.path(where, original_paths(j)[0]).exists()]
            if expected != cp["eligible_job_ids"]:
                raise ValueError("Continuation allowlist includes successful or unknown jobs")
            if (not cp["historical_failure_cost_unknown"]
                    or api.money(cp["historical_failure_debit_usd"]) != api.money(retry_jobs[0]["reservation_usd"])):
                raise ValueError("Historical unknown-cost reservation must be retained unchanged")
        finally:
            api.API_DIR = previous
            api.record_paths = previous_paths
        api.stage.verify_hashes(where, cp["source_files_sha256"])
        api.stage.verify_hashes(where, cp["history_files_sha256"])
        api.stage.verify_hashes(where, {f"{continuation}/requests.jsonl": cp["requests_sha256"]})
        if api.stage.rows(where, f"{continuation}/requests.jsonl") != jobs:
            raise ValueError("Continuation requests differ from the original frozen requests")
        if cp["initial_operational_safety"] != base_plan["operational_key_safety"]:
            raise ValueError("Continuation cannot expand the original key allowance")
        if (len(eligible) != len(cp["eligible_job_ids"]) or cp["retry_job_id"] not in eligible
                or len(eligible) != cp["max_additional_http_attempts"]):
            raise ValueError("Invalid continuation allowlist")
        authorization = api.stage.read(where, REGISTRY)
        if any(cp[k] != value for k, value in authorization.items()) or cp["continuation"] != continuation:
            raise ValueError("Continuation differs from its one-time authorization")
        return base_plan, jobs

    def paths(job):
        if job["job_id"] in eligible:
            return tuple(f"{continuation}/records/{job['job_id']}{suffix}" for suffix in (".attempt.json", ".response.json"))
        # original_paths uses api.API_DIR at call time; construct explicit base paths.
        return tuple(f"{BASE_API}/records/{job['job_id']}{suffix}" for suffix in (".attempt.json", ".response.json"))

    def scan(where, jobs):
        state = original_scan(where, jobs)
        # A continuation is intentionally stricter: no further POST after any
        # newly failed response, including HTTP200 schema/IDs/truncation failures.
        if state["states"].get("failed"):
            state["must_stop"] = True
        debit = api.money(cp["historical_failure_debit_usd"])
        state["accounted_usd"] += debit
        if cp["historical_failure_cost_unknown"]:
            state["missing_cost_jobs"].append("prior_attempt/" + cp["retry_job_id"])
        else:
            state["reported_cost_usd"] += debit
        state["providers"]["unreported_historical_admission_error"] = 1
        return state

    def summary(plan, state, where):
        result = original_summary(plan, state, where)
        new_attempts = list(api.stage.path(where, f"{continuation}/records").glob("*.attempt.json"))
        result["continuation"] = {"original_http_attempts": cp["original_http_attempts"],
            "new_http_attempts": len(new_attempts), "retry_job_id": cp["retry_job_id"],
            "previous_failure_cost_unknown": cp["historical_failure_cost_unknown"],
            "total_http_attempts": cp["original_http_attempts"] + len(new_attempts)}
        result["note"] = "Jobs reflect latest attempt per original batch; historical admission429 retained separately. Exactly one explicit transport retry, no quality-conditioned repeats."
        return result

    api.API_DIR = continuation
    api.record_paths, api.scan, api.summary, api.verify_plan = paths, scan, summary, verify
    verify(root)
    return api


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "status", "export", "run"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--continuation", default=DEFAULT_CONTINUATION)
    parser.add_argument("--key-status-file")
    parser.add_argument("--creds-file")
    parser.add_argument("--max-new-requests", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.command == "prepare":
            if not args.key_status_file:
                raise ValueError("prepare requires sanitized --key-status-file")
            result = prepare(root, args.continuation, args.key_status_file)
        else:
            api = attach(root, args.continuation)
            if args.command == "run":
                with api.stage.path(root, f"{BASE_API}/runner.lock").open("a") as stream:
                    api.fcntl.flock(stream, api.fcntl.LOCK_EX | api.fcntl.LOCK_NB)
                    result = api.run(root, args.creds_file, args.max_new_requests)
            else:
                plan, jobs = api.verify_plan(root)
                with api.exclusive_run(root):
                    state = api.scan(root, jobs)
                    if args.command == "export":
                        api.sync_outputs(root, state)
                    result = api.summary(plan, state, root)
    except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
        parser.exit(1, f"E08 continuation stopped: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
