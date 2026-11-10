"""Evaluate the original Base and ACE conditions on a fresh FiNER holdout."""

import argparse
import ast
import json
import os
import random
import statistics
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "evidence"
RUNS = ROOT / "runs"
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(RUNS / "tokenizer"))

from compact_data import _verify_tags
from finer_api import (
    BASE_URL, MISS_PRICE, NANO_PER_CNY, OUTPUT_PRICE, BudgetedDeepSeekClient,
    ace_timed_llm_call,
)
from finer_data import _sha256
from rescue_data import _source_rows
from run_finer import MODEL, UPSTREAM, append_json, evaluation_row, write_json
from run_rescue import INITIAL_CAP, INPUT_CAP, load_playbooks

BATCHES = 84
RANK_PREFIX = "ACE-FINER-RETEST-V1|"
ORDER_SEED = 20261110
ARMS = ("base", "ace_upstream")
PRIOR_LEDGER = ROOT / "evidence" / "deepseek_ledger_compact.jsonl"


def selection(*, batch_count=BATCHES, rank_prefix=RANK_PREFIX, extra_prior_runs=(), holdout_split="test"):
    if holdout_split not in ("test", "val"):
        raise ValueError("Holdout must come from the released test or validation file")
    names = ("finer_manifest.json", "rescue_manifest.json", "compact_manifest.json")
    manifests = {name: (ROOT / name).read_bytes() for name in names}
    finer, rescue, compact = map(json.loads, manifests.values())
    sources = finer["sources"]
    if any(m["sources"] != sources or m["upstream_commit"] != UPSTREAM
           for m in (finer, rescue, compact)):
        raise ValueError("Prior manifests do not match the pinned sources")
    sources_rows = {split: list(_source_rows(split, data)) for split, data in sources.items()}
    ledger = PRIOR_LEDGER.read_bytes()
    previous_runs = {}
    previous_rows = set()
    ledgers = [ledger]
    for directory in extra_prior_runs:
        payload = (directory / "run_manifest.json").read_bytes()
        run = json.loads(payload)
        holdout = run.get("holdout", run)
        run_sources = holdout.get("sources", run.get("source_files"))
        if run["upstream_commit"] != UPSTREAM or run_sources != sources:
            raise ValueError("Additional prior run does not match the pinned sources")
        if "selection" in holdout:
            previous_rows.update(record["row_id"] for record in holdout["selection"])
        else:
            previous_rows.update(run["test_row_ids"])
        usage = (directory / "usage.jsonl").read_bytes()
        ledgers.append(usage)
        previous_runs[directory.name] = {
            "manifest_sha256": _sha256(payload), "ledger_sha256": _sha256(usage),
        }
    events = [json.loads(line) for payload in ledgers for line in payload.splitlines()]
    reserves = {e["attempt_id"]: e for e in events if e["event"] == "reserve"}
    settles = {e["attempt_id"]: e for e in events if e["event"] == "settle"}
    prior_nano = sum(
        settles[key]["cost_peak_nano_cny"] if key in settles else event["reserved_nano_cny"]
        for key, event in reserves.items()
    )
    if any(e["limit_violation"] for e in settles.values()):
        raise ValueError("Prior usage exceeded a token bound")

    reserved_ids = {e["row_id"] for e in reserves.values() if "row_id" in e}
    reserved_ids.update(r["row_id"] for phase in finer["selections"].values()
                        for split in phase.values() for r in split)
    reserved_ids.update(r["row_id"] for r in rescue["records"])
    reserved_ids.update(r["row_id"] for split in compact["selections"].values() for r in split)
    reserved_ids.update(previous_rows)
    known_ids = {r["row_id"] for split in sources_rows.values() for r, _ in split}
    if any(r.startswith("finer:") and r not in known_ids for r in reserved_ids):
        raise ValueError("Unknown prior FiNER sample")
    seen = {sentence for split, rows in sources_rows.items() for r, _ in rows
            if split != holdout_split or r["row_id"] in reserved_ids
            for sentence in r["sentence_sha256"]}
    for record, _ in sources_rows[holdout_split]:
        record["selection_rank_sha256"] = _sha256(rank_prefix + record["original_context_sha256"])
    ranked = sorted(sources_rows[holdout_split], key=lambda pair: (
        pair[0]["selection_rank_sha256"],
        pair[0]["physical_line_1based"],
    ))
    eligible = []
    for record, row in ranked:
        hashes = set(record["sentence_sha256"])
        if row["row_id"] in reserved_ids or seen.intersection(hashes):
            continue
        seen.update(hashes)
        eligible.append((record, row))
    if not eligible or (batch_count is not None and len(eligible) < batch_count):
        raise ValueError("Too few fresh test batches")
    records, rows = zip(*eligible[:batch_count])
    for row in rows:
        _verify_tags(row)
    return rows, {
        "upstream_commit": UPSTREAM, "sources": sources,
        "prior_manifest_sha256": {name: _sha256(data) for name, data in manifests.items()},
        "prior_ledger_sha256": _sha256(ledger), "prior_committed_nano_cny": prior_nano,
        "prior_attempts": len(reserves), "prior_model_fingerprints": sorted({
            e["system_fingerprint"] for e in settles.values() if e.get("system_fingerprint")
        }),
        "rank_prefix": rank_prefix, "eligible_batches": len(eligible),
        **({"extra_prior_runs": previous_runs} if previous_runs else {}),
        "selection": records,
    }


def differences_interval(deltas):
    rng = random.Random(ORDER_SEED)
    samples = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(10000))
    return [samples[250], samples[9749]]


def summarize(pairs):
    def stats(arm, rows):
        counts = Counter(p[arm]["error_type"] if "error_type" in p[arm]
                         else p[arm]["status"] for p in rows)
        return {"correct": sum(p[arm]["correct_labels"] for p in rows), "status": dict(counts)}

    def metrics(rows):
        total = 4 * len(rows)
        counts = {arm: stats(arm, rows) for arm in ARMS}
        return {
            "denominator": total, "arms": counts,
            "delta_pp": 100 * (counts["ace_upstream"]["correct"] - counts["base"]["correct"]) / total,
        }

    grouped = {}
    for pair in pairs:
        diff = pair["ace_upstream"]["correct_labels"] - pair["base"]["correct_labels"]
        grouped.setdefault(pair["row_id"], []).append(diff)
    if any(len(deltas) != 3 for deltas in grouped.values()):
        raise ValueError("Every batch must have three paired evaluations")
    completed = [p for p in pairs if all(p[arm]["status"] == "scored"
                 and p[arm]["predicted_positions"] == 4 and p[arm]["missing_positions"] == 0
                 for arm in ARMS)]
    transitions = Counter()
    for pair in pairs:
        gold = pair["base"]["target"].lower().split(",")
        predictions = [(pair[arm]["prediction"] or "").lower().split(",") + [""] * 4 for arm in ARMS]
        for expected, base, ace in zip(gold, *predictions):
            transitions[f"{int(base.strip() == expected.strip())}{int(ace.strip() == expected.strip())}"] += 1
    pooled = metrics(pairs)
    if (transitions["11"] + transitions["10"] != pooled["arms"]["base"]["correct"]
            or transitions["11"] + transitions["01"] != pooled["arms"]["ace_upstream"]["correct"]):
        raise ValueError("Independent position counts disagree with the upstream scorer")
    return {
        "pooled": pooled, "both_complete": metrics(completed) if completed else None,
        "trials": {seed: metrics([p for p in pairs if p["seed"] == seed])
                   for seed in sorted({p["seed"] for p in pairs})},
        "base_ace_correct_transitions": dict(sorted(transitions.items())),
        "paired_batch_bootstrap_95pct_delta_pp": differences_interval(
            [25 * statistics.mean(deltas) for deltas in grouped.values()]
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--confirm-100", action="store_true", help="Test the frozen 100-batch checkpoints")
    args = parser.parse_args()
    default_output = RUNS / ("confirm_100" if args.confirm_100 else "fresh_retest")
    args.out = (args.out or default_output).resolve()
    if args.run and args.out != default_output:
        parser.error("Live inference writes to the fixed runs directory for this comparison.")
    if args.run and not os.environ.get("DEEPSEEK_API_KEY"):
        parser.error("Set DEEPSEEK_API_KEY in the local environment before running live inference.")

    from ace import ACE
    from ace.core import generator as generator_module
    from eval.finance import data_processor

    template = run_path(str(ROOT / "ace" / "prompts" / "generator.py"))["GENERATOR_PROMPT"]
    input_cap, timeout = INPUT_CAP, 30
    if args.confirm_100:
        scale = WORK / "training_scale"
        scale_plan = json.loads((scale / "run_manifest.json").read_bytes())
        if (scale_plan["generator_prompt_sha256"] != _sha256(template)
                or scale_plan["model"] != MODEL or scale_plan["base_url"] != BASE_URL
                or scale_plan["output_cap"] != INITIAL_CAP):
            raise ValueError("The evaluation settings differ from the 100-batch run")
        input_cap, timeout = scale_plan["administrative_input_bound"], 45
        rows, manifest = selection(
            batch_count=None, rank_prefix="ACE-FINER-CONFIRM-100-V1|",
            extra_prior_runs=tuple(WORK / name for name in (
                "fresh_retest", "evidence_first", "training_dose", "training_scale",
            )),
        )
        expected = json.loads((scale / "checkpoint_hashes.json").read_bytes())
        books = {}
        for seed_text, hashes in sorted(expected.items()):
            seed = int(seed_text)
            checkpoint = json.loads((scale / f"trial_{seed}" / "checkpoint_100.json").read_bytes())
            if (checkpoint["seed"] != seed or checkpoint["completed_total"] != 100
                    or _sha256(checkpoint["playbook"]) != hashes["100"]):
                raise ValueError(f"Unverified 100-batch checkpoint for seed {seed}")
            books[seed] = checkpoint["playbook"]
        manifest["ace_upstream_training_batches"] = 100
    else:
        rows, manifest = selection()
        books = load_playbooks(ROOT / "evidence" / "formal")
    base_book = ACE._initialize_empty_playbook(None)
    for seed in books:
        base_summary = json.loads((ROOT / "evidence" / "formal" / f"trial_{seed}"
                                   / "base" / "summary.json").read_bytes())
        if base_summary["test"]["playbook_sha256"] != _sha256(base_book):
            raise ValueError("The Base playbook does not match the original comparison")
    bounds = [
        544 + len(template.format(book, "(empty)", row["question"], row["context"]).encode("utf-8"))
        for row in rows for ace_book in books.values() for book in (base_book, ace_book)
    ]
    budget_nano = 50 * NANO_PER_CNY - manifest["prior_committed_nano_cny"]
    worst_nano = 2 * sum(bound * MISS_PRICE + INITIAL_CAP * OUTPUT_PRICE for bound in bounds)
    if (max(bounds) > input_cap or budget_nano <= 0
            or (not args.confirm_100 and worst_nano > budget_nano)):
        raise ValueError("Worst-case test cost exceeds the available budget")
    manifest.update({
        "model": MODEL, "base_url": BASE_URL, "temperature": 0, "thinking": "disabled",
        "json_mode": True, "max_output_tokens": INITIAL_CAP, "max_input_tokens": input_cap,
        "length_retry": False, "maximum_transport_attempts": 2,
        "playbook_sha256": {seed: _sha256(book) for seed, book in books.items()},
        "base_playbook_sha256": _sha256(base_book), "generator_prompt_sha256": _sha256(template),
        "prior_committed_cny": manifest["prior_committed_nano_cny"] / NANO_PER_CNY,
        "available_cny": budget_nano / NANO_PER_CNY,
        "max_new_reservation_cny": worst_nano / NANO_PER_CNY, "planned_completions": len(bounds),
        "max_new_attempts": 2 * len(bounds), "bootstrap_seed": ORDER_SEED,
        "selection_rule": "SHA-256 rank; exclude previous rows and their normalized sentences, "
                          "and sentences anywhere in the released train and validation files.",
    })
    print(json.dumps({key: manifest[key] for key in (
        "eligible_batches", "planned_completions", "prior_committed_cny", "available_cny",
    )}), flush=True)
    if not args.run:
        return
    args.out.mkdir(parents=True, exist_ok=False)
    run_id = uuid4().hex
    write_json(args.out / "run_manifest.json", {"run_id": run_id, **manifest})
    client = BudgetedDeepSeekClient(
        args.out / "usage.jsonl", budget_cny=Decimal(budget_nano) / NANO_PER_CNY,
        max_attempts=2 * len(bounds), max_input_tokens=input_cap,
        max_output_tokens=INITIAL_CAP, timeout_s=timeout,
    )
    generator = generator_module.Generator(client, "deepseek", MODEL, max_tokens=INITIAL_CAP)
    processor = data_processor.DataProcessor("finer")
    jobs = [(i, row, seed) for i, row in enumerate(rows) for seed in books]

    def paired(job):
        i, row, seed = job
        order = ARMS if (i + seed // 2) % 2 == 0 else ARMS[::-1]
        pair = {"row_id": row["row_id"], "seed": seed, "arm_order": order}
        for arm in order:
            book = base_book if arm == "base" else books[seed]
            system = SimpleNamespace(generator=generator, playbook=book)
            phase = "confirm_100" if args.confirm_100 else "retest"
            pair[arm] = evaluation_row(system, processor, phase, row, seed, arm, run_id)
        return pair

    try:
        with (patch.object(generator_module, "timed_llm_call", ace_timed_llm_call),
              patch.object(data_processor, "eval", ast.literal_eval, create=True),
              (args.out / "paired_rows.jsonl").open("w", encoding="utf-8") as progress,
              ThreadPoolExecutor(max_workers=12) as pool):
            first = paired(jobs[0])
            append_json(progress, first)
            if any(first[arm].get("error_type") not in (None, "TruncatedCompletion") for arm in ARMS):
                raise RuntimeError("Initial API pair failed. See the recorded provider status.")
            pairs = [first]
            for pair in pool.map(paired, jobs[1:]):
                pairs.append(pair)
                append_json(progress, pair)
                if len(pairs) % 24 == 0:
                    print(f"Completed {len(pairs)}/{len(jobs)} paired batches", flush=True)
        usage = client.summary()
    finally:
        client.close()
    summary = summarize(pairs)
    summary["usage"] = usage
    summary["status"] = "complete" if all(
        set(values["status"]) <= {"scored", "TruncatedCompletion", "invalid_answer"}
        for values in summary["pooled"]["arms"].values()
    ) else "incomplete"
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
