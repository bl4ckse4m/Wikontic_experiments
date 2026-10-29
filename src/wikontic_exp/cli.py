"""CLI for the Wikontic extraction-method experiments."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from . import aggregation, data_prep, distill_export, evaluation, qualifiers
from .api_client import AccessConfig, OpenAICompatClient
from .runner import MethodConfig, Runner, aggregate_usage, load_inputs


def _p(path: str) -> Path:
    return Path(path)


def cmd_prepare_data(args: argparse.Namespace) -> int:
    res = data_prep.prepare(
        data_dir=_p(args.data_dir),
        split_file=args.split_file,
        seed=args.seed,
        tuning_n=args.tuning_n,
        collection_n=args.collection_n,
    )
    gold = data_prep.select_gold_sample(
        inputs_path=_p(args.data_dir) / "tuning" / "inputs.jsonl",
        out_path=_p(args.data_dir) / "gold" / "gold_selection.json",
        n=args.gold_n,
    )
    print(json.dumps({
        "tuning": {k: res["tuning"][k] for k in ("n_qa_records", "n_unique_paragraph_inputs",
                                                 "qa_id_overlap_with_collection", "paragraph_overlap_with_collection")},
        "collection": {k: res["collection"][k] for k in ("n_qa_records", "n_unique_paragraph_inputs",
                                                        "qa_id_overlap_with_tuning", "paragraph_overlap_with_tuning",
                                                        "collection_n_obtained",
                                                        "rejected_for_tuning_paragraph_overlap",
                                                        "paragraph_slots_collection")},
        "gold_selection_n": gold["n_selected"],
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    cfg = AccessConfig.load(args.access, creds_file=args.creds_file, openrouter=args.openrouter)
    missing = cfg.missing()
    if missing:
        print(json.dumps({"status": "missing_access_settings", "missing": missing}, indent=2))
        return 2

    async def _run() -> dict:
        async with OpenAICompatClient(cfg) as c:
            return await c.probe()

    print(json.dumps(asyncio.run(_run()), ensure_ascii=False, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    access = (AccessConfig("mock", "mock", args.model or "mock") if args.mock
              else AccessConfig.load(args.access, creds_file=args.creds_file, openrouter=args.openrouter))
    model = access.model or args.model
    method = MethodConfig.load(args.method, model_default=model)
    if args.openrouter and not args.mock:
        method.model = access.model
    if args.model:
        method.model = args.model
        access.model = args.model
    if not args.mock:
        missing = access.missing()
        if missing:
            print(json.dumps({"status": "missing_access_settings", "missing": missing}, indent=2))
            return 2
        if not method.model:
            print("no model configured", file=sys.stderr)
            return 2
    inputs = load_inputs(args.inputs, limit=args.limit)
    runner = Runner(method=method, run_dir=_p(args.run_dir), access=access, mock=args.mock)
    runner.write_run_config(args.inputs, {"limit": args.limit, "resume": args.resume,
                                          "retry_failures": args.retry_failures})
    stats = asyncio.run(runner.run(inputs, resume=args.resume, retry_failures=args.retry_failures))
    stats["usage_totals"] = aggregate_usage(_p(args.run_dir))
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0 if stats.get("failed", 0) == 0 else 1


def _collect_records(run_dirs: list[str]) -> list[dict]:
    records: list[dict] = []
    for rd in run_dirs:
        records.extend(aggregation.load_run_records(_p(rd)))
    return records


def cmd_aggregate(args: argparse.Namespace) -> int:
    records = _collect_records(args.run_dirs)
    cands = aggregation.load_candidates(records)
    agg = aggregation.aggregate(
        cands,
        algorithm=args.algorithm,
        vote_threshold=args.vote_threshold,
        similarity_threshold=args.similarity_threshold,
    )
    key = aggregation.save_aggregate(agg, records, _p(args.out))
    summary = {
        "merger_cache_key": key,
        "n_candidate_facts": len(cands),
        "n_inputs": len(agg["results"]),
        "n_merged_facts": sum(len(v) for v in agg["results"].values()),
        "n_decisions": len(agg["decisions"]),
        "algorithm": agg["algorithm"],
        "params": agg["params"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    records = _collect_records(args.run_dirs)
    texts = {}
    if args.inputs:
        for rec in load_inputs(args.inputs):
            texts[rec["input_id"]] = rec
    out: dict = {"proxy": evaluation.proxy_report(records, texts)}
    if args.gold:
        gold = evaluation.load_gold(_p(args.gold))
        preds = []
        for r in records:
            if r.get("status") == "ok" and r.get("parsed"):
                preds.extend(r["parsed"]["triplets"])
        out["gold"] = evaluation.gold_eval(preds, gold)
        out["gold_file"] = args.gold
        out["gold_status"] = "independent annotation file; if empty/partial -> proxy only"
    out["compactness"] = evaluation.compactness(
        [f for r in records if r.get("status") == "ok" and r.get("parsed") for f in r["parsed"]["triplets"]]
    )
    out["usage"] = {rd: aggregate_usage(_p(rd)) for rd in args.run_dirs}
    _p(args.out).parent.mkdir(parents=True, exist_ok=True)
    _p(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_qual_ablation(args: argparse.Namespace) -> int:
    res = qualifiers.build_ablation_inputs(_p(args.frozen), _p(args.out_dir))
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def cmd_export_corpus(args: argparse.Namespace) -> int:
    teacher_meta = json.loads(_p(args.teacher_meta).read_text(encoding="utf-8"))
    res = distill_export.export_corpus(
        inputs_path=_p(args.inputs),
        merged_path=_p(args.merged),
        out_path=_p(args.out),
        teacher_meta=teacher_meta,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def cmd_usage(args: argparse.Namespace) -> int:
    out = {rd: aggregate_usage(_p(rd)) for rd in args.run_dirs}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wikontic-exp", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare-data", help="sample tuning/collection, write manifests and inputs")
    sp.add_argument("--data-dir", default="data")
    sp.add_argument("--split-file", default=data_prep.DEFAULT_FILE)
    sp.add_argument("--seed", type=int, default=data_prep.SEED)
    sp.add_argument("--tuning-n", type=int, default=data_prep.TUNING_N)
    sp.add_argument("--collection-n", type=int, default=data_prep.COLLECTION_N)
    sp.add_argument("--gold-n", type=int, default=30)
    sp.set_defaults(func=cmd_prepare_data)

    sp = sub.add_parser("probe", help="probe the API endpoint capabilities")
    sp.add_argument("--access", default="configs/access.local.yaml")
    sp.add_argument("--creds-file", default=None, help="external KEY=VALUE file; default ~/.config/wikontic/creds")
    sp.add_argument("--openrouter", action="store_true", help="explicitly use separate OpenRouter credentials (no automatic fallback)")
    sp.set_defaults(func=cmd_probe)

    sp = sub.add_parser("run", help="run one extraction method over inputs")
    sp.add_argument("--method", required=True, help="method config JSON")
    sp.add_argument("--inputs", required=True, help="inputs jsonl (input_id,title,paragraph_text)")
    sp.add_argument("--run-dir", required=True)
    sp.add_argument("--access", default="configs/access.local.yaml")
    sp.add_argument("--creds-file", default=None)
    sp.add_argument("--openrouter", action="store_true", help="use OpenRouter key, endpoint and model instead of primary API")
    sp.add_argument("--model", default=None)
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--mock", action="store_true", help="dry-run with deterministic mock client")
    sp.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    sp.add_argument("--retry-failures", action="store_true")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("aggregate", help="merge independent answers")
    sp.add_argument("--run-dirs", nargs="+", required=True)
    sp.add_argument("--algorithm", choices=["union_dedup", "vote"], default="union_dedup")
    sp.add_argument("--vote-threshold", type=int, default=1)
    sp.add_argument("--similarity-threshold", type=float, default=0.9)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_aggregate)

    sp = sub.add_parser("evaluate", help="proxy + optional gold evaluation")
    sp.add_argument("--run-dirs", nargs="+", required=True)
    sp.add_argument("--inputs", default=None)
    sp.add_argument("--gold", default=None)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("qual-ablation", help="build Q0/Q1/Q2 from one frozen fact set")
    sp.add_argument("--frozen", required=True)
    sp.add_argument("--out-dir", required=True)
    sp.set_defaults(func=cmd_qual_ablation)

    sp = sub.add_parser("export-corpus", help="export distillation corpus")
    sp.add_argument("--inputs", required=True)
    sp.add_argument("--merged", required=True)
    sp.add_argument("--teacher-meta", required=True)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_export_corpus)

    sp = sub.add_parser("usage", help="cost/usage totals of run dirs")
    sp.add_argument("--run-dirs", nargs="+", required=True)
    sp.set_defaults(func=cmd_usage)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
