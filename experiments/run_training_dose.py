"""Continue the original FiNER playbooks from 12 to 24 training batches."""

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, redirect_stdout
from decimal import Decimal
from importlib import import_module
import json
from pathlib import Path
import random
from runpy import run_path
import statistics
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from retest_finer import (
    BASE_URL, INITIAL_CAP, INPUT_CAP, MODEL, NANO_PER_CNY, PRIOR_LEDGER, ROOT, UPSTREAM, WORK,
    BudgetedDeepSeekClient, _sha256, append_json, differences_interval,
    evaluation_row, load_playbooks, write_json,
)
from finer_api import BudgetExceeded, LedgerError, ProviderRequestError, ace_timed_llm_call, request_scope
from finer_data import load_splits
from rescue_data import _source_rows
from run_finer import new_ace

OLD, NEW = "ace_12", "ace_24"
ARMS = ("base", OLD, NEW)
HOLDOUT = WORK / "fresh_retest"
FORMAL = ROOT / "evidence" / "formal"
TRAIN_COUNT = 12


def read_rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def prior_budget(extra_ledgers=()):
    paths = (PRIOR_LEDGER, HOLDOUT / "usage.jsonl", WORK / "evidence_first" / "usage.jsonl", *extra_ledgers)
    spent = 0
    fingerprints = set()
    ledgers = {}
    for path in paths:
        payload = path.read_bytes()
        events = [json.loads(line) for line in payload.splitlines()]
        reserves = {e["attempt_id"]: e for e in events if e["event"] == "reserve"}
        settles = {e["attempt_id"]: e for e in events if e["event"] == "settle"}
        if any(e["limit_violation"] for e in settles.values()):
            raise ValueError("Earlier usage exceeded its reserved bound")
        spent += sum(settles[key]["cost_peak_nano_cny"] if key in settles
                     else reserve["reserved_nano_cny"] for key, reserve in reserves.items())
        fingerprints.update(e["system_fingerprint"] for e in settles.values() if e.get("system_fingerprint"))
        ledgers[path.parent.name + "/" + path.name] = _sha256(payload)
    return spent, sorted(fingerprints), ledgers


def frozen_inputs():
    old_books = load_playbooks(FORMAL)
    frozen = json.loads((HOLDOUT / "run_manifest.json").read_bytes())
    formal = json.loads((FORMAL / "run_manifest.json").read_bytes())
    if frozen["playbook_sha256"] != {str(seed): _sha256(book) for seed, book in old_books.items()}:
        raise ValueError("Playbook hashes do not match the original 84-batch test")
    template = run_path(str(ROOT / "ace" / "prompts" / "generator.py"))["GENERATOR_PROMPT"]
    if _sha256(template) != frozen["generator_prompt_sha256"]:
        raise ValueError("Generator prompt differs from the frozen comparison")
    raw_test = {r["row_id"]: (r, row) for r, row in _source_rows("test", frozen["sources"]["test"])}
    rows = []
    for record in frozen["selection"]:
        source, row = raw_test[record["row_id"]]
        if source["jsonl_line_sha256"] != record["jsonl_line_sha256"]:
            raise ValueError("The holdout source changed")
        rows.append(row)
    if len(rows) != 84 or len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("Expected the same 84 distinct test batches")
    archived = {(p["seed"], p["row_id"]): p for p in read_rows(HOLDOUT / "paired_rows.jsonl")}
    if set(archived) != {(seed, r["row_id"]) for seed in old_books for r in rows}:
        raise ValueError("Archived direct-answer reference is incomplete")
    if any(archived[seed, r["row_id"]]["base"]["target"] != r["target"]
           for seed in old_books for r in rows):
        raise ValueError("Archived answers use different targets")

    standard = load_splits(pilot=False, scope="standard")["train"]
    prefix, extra = standard[:TRAIN_COUNT], standard[TRAIN_COUNT:2 * TRAIN_COUNT]
    checkpoints, orders = {}, {}
    for seed, book in old_books.items():
        archive = FORMAL / f"trial_{seed}" / "ace_upstream"
        checkpoint = json.loads((archive / "checkpoint.json").read_bytes())
        training = read_rows(archive / "training_rows.jsonl")
        expected_ids = formal["train_order"][str(seed)]
        if (checkpoint["trial"] != seed or checkpoint["arm"] != "ace_upstream"
                or checkpoint["last_attempted_step"] != TRAIN_COUNT
                or checkpoint["completed_training_batches"] != TRAIN_COUNT
                or checkpoint["playbook_sha256"] != _sha256(book) or checkpoint["playbook"] != book
                or expected_ids != [record["row_id"] for record in training]
                or set(expected_ids) != {r["row_id"] for r in prefix}
                or any(record["status"] != "complete" for record in training)):
            raise ValueError(f"Trial {seed} lacks a reproducible step-12 checkpoint")
        checkpoints[seed] = checkpoint
        orders[seed] = list(extra)
        random.Random(seed).shuffle(orders[seed])
    return rows, archived, checkpoints, orders, prefix, frozen, formal


def extend(seed, checkpoint, rows, processor, path, run_id):
    path.mkdir()
    system = new_ace(INITIAL_CAP)
    system.playbook = checkpoint["playbook"]
    system.next_global_id = checkpoint["next_global_id"]
    config = system._extract_config_params({
        "task_name": "finer", "max_num_rounds": 3, "curator_frequency": 1,
        "json_mode": True, "no_ground_truth": False,
    })
    completed = 0
    with (path / "training_rows.jsonl").open("w", encoding="utf-8") as progress:
        for step, row in enumerate(rows, TRAIN_COUNT + 1):
            before, next_id = system.playbook, system.next_global_id
            result = {"step": step, "row_id": row["row_id"]}
            try:
                with request_scope(run_id=run_id, trial=seed, arm=NEW, phase="train", row_id=row["row_id"]):
                    pre, post, _ = system._train_single_sample(
                        task_dict={k: row[k] for k in ("row_id", "context", "question", "target")},
                        data_processor=processor, step_id=f"train_e_1_s_{step}", epoch=1, step=step,
                        usage_log_path=str(path / "bullet_usage_log.jsonl"), log_dir=str(path),
                        config_params=config, total_samples=2 * TRAIN_COUNT,
                    )
                completed += 1
                result.update(status="complete", pre_prediction=pre, post_prediction=post)
            except (BudgetExceeded, LedgerError):
                raise
            except Exception as error:
                if isinstance(error, ProviderRequestError) and error.status_code in (400, 401, 402, 403, 404):
                    raise
                system.playbook, system.next_global_id = before, next_id
                result.update(status="error", error_type=type(error).__name__)
            result["playbook_sha256"] = _sha256(system.playbook)
            append_json(progress, result)
            write_json(path / "checkpoint.json", {
                "seed": seed, "last_step": step, "completed_total": TRAIN_COUNT + completed,
                "next_global_id": system.next_global_id, "playbook": system.playbook,
                "playbook_sha256": result["playbook_sha256"],
            })
    (path / "playbook_24.txt").write_bytes(system.playbook.encode("utf-8"))
    return {"playbook": system.playbook, "completed_additional": completed,
            "sha256": _sha256(system.playbook)}


def summarize(pairs):
    totals = Counter()
    per_batch = defaultdict(list)
    for pair in pairs:
        gold = pair["base"]["target"].lower().split(",")
        masks = {}
        for arm in ARMS:
            r = pair[arm]
            pred = (r["prediction"] or "").lower().split(",") + [""] * 4
            masks[arm] = [a.strip() == b.strip() for a, b in zip(pred, gold)]
            if len(gold) != 4 or sum(masks[arm]) != r["correct_labels"]:
                raise ValueError("Independent label scoring disagrees with recorded scores")
        for b, a, c in zip(*(masks[arm] for arm in ARMS)):
            totals[b, a, c] += 1
            per_batch[pair["row_id"]].append(c - a)
    n = 4 * len(pairs)
    metrics = {}
    for i, arm in enumerate(ARMS):
        metrics[arm] = {
            "correct": sum(k for flags, k in totals.items() if flags[i]),
            "status": dict(Counter(p[arm].get("error_type", p[arm]["status"]) for p in pairs)),
        }
        if i:
            metrics[arm].update(
                base_right_to_wrong=sum(k for flags, k in totals.items() if flags[0] and not flags[i]),
                base_wrong_to_right=sum(k for flags, k in totals.items() if not flags[0] and flags[i]),
            )
    return {
        "positions": n, "arms": metrics,
        "accuracy_difference_pp": 100 * (metrics[NEW]["correct"] - metrics[OLD]["correct"]) / n,
        "batch_delta": [100 * statistics.mean(values) for values in per_batch.values()],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    from eval.finance import data_processor
    rows, archived, checkpoints, orders, prefix, frozen, formal = frozen_inputs()
    spent, fingerprints, ledgers = prior_budget()
    prior_budget_nano = 50 * NANO_PER_CNY - spent
    max_attempts = 2 * (9 * TRAIN_COUNT * len(orders) + 2 * len(archived))
    run_id = uuid4().hex
    manifest = {
        "run_id": run_id, "upstream_commit": UPSTREAM, "model": MODEL, "base_url": BASE_URL,
        "temperature": 0, "thinking": "disabled", "json_mode": True, "output_cap": INITIAL_CAP,
        "input_cap": INPUT_CAP, "reflection_round_cap": 3, "initial_training_batches": TRAIN_COUNT,
        "additional_training_batches": TRAIN_COUNT, "curator_progress_total_initial": TRAIN_COUNT,
        "curator_progress_total_continuation": 2 * TRAIN_COUNT,
        "original_train_order": formal["train_order"],
        "extension_order": {seed: [r["row_id"] for r in batches] for seed, batches in orders.items()},
        "initial_checkpoint_sha256": {seed: c["playbook_sha256"] for seed, c in checkpoints.items()},
        "test_row_ids": [r["row_id"] for r in rows], "source_files": frozen["sources"],
        "archived_test_manifest_sha256": _sha256((HOLDOUT / "run_manifest.json").read_bytes()),
        "archived_reference_sha256": _sha256((HOLDOUT / "paired_rows.jsonl").read_bytes()),
        "prompt_sha256": frozen["generator_prompt_sha256"],
        "baseline_source": "Archived no-playbook predictions on the same 84 batches",
        "checkpoint_policy": "Compare fixed step-12 and step-24 endpoints; no test-based selection.",
        "prior_ledger_sha256": ledgers, "prior_fingerprints": fingerprints,
        "prior_peak_cny": spent / NANO_PER_CNY, "available_cny": prior_budget_nano / NANO_PER_CNY,
        "max_api_attempts": max_attempts,
    }
    print(json.dumps({
        "test_batches": len(rows), "paired_comparisons": len(archived),
        "old_train_distinct_tags": len({t.strip() for r in prefix for t in r["target"].split(",")}),
        "extended_train_distinct_tags": len({t.strip() for r in prefix + orders[17] for t in r["target"].split(",")}),
        "remaining_budget_cny": prior_budget_nano / NANO_PER_CNY,
    }), flush=True)
    if not args.run:
        return
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "run_manifest.json", manifest)
    client = BudgetedDeepSeekClient(
        args.out / "usage.jsonl", budget_cny=Decimal(prior_budget_nano) / NANO_PER_CNY,
        max_attempts=max_attempts, max_input_tokens=INPUT_CAP, max_output_tokens=INITIAL_CAP, timeout_s=30,
    )
    processor = data_processor.DataProcessor("finer")
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(import_module("ace.ace"), "initialize_clients",
                                             lambda provider: (client, client, client)))
            for role in ("generator", "reflector", "curator"):
                stack.enter_context(patch.object(import_module(f"ace.core.{role}"), "timed_llm_call", ace_timed_llm_call))
            stack.enter_context(patch.object(data_processor, "eval", ast.literal_eval, create=True))
            with ((args.out / "training_console.log").open("w", encoding="utf-8") as log,
                  redirect_stdout(log), ThreadPoolExecutor(max_workers=3) as pool):
                futures = {pool.submit(extend, seed, c, orders[seed], processor,
                                       args.out / f"trial_{seed}", run_id): seed
                           for seed, c in checkpoints.items()}
                extended = {futures[f]: f.result() for f in as_completed(futures)}
            write_json(args.out / "trained_checkpoints.json", {
                seed: {k: v for k, v in book.items() if k != "playbook"} for seed, book in extended.items()
            })
            print("Training continuation complete", flush=True)
            generator = new_ace(INITIAL_CAP).generator
            jobs = [(i, row, seed) for i, row in enumerate(rows) for seed in checkpoints]

            def evaluate(job):
                i, row, seed = job
                old = archived[seed, row["row_id"]]
                result = {"row_id": row["row_id"], "seed": seed, "base": old["base"],
                          "archived_ace_12": old["ace_upstream"]}
                books = {OLD: checkpoints[seed]["playbook"], NEW: extended[seed]["playbook"]}
                order = (OLD, NEW) if (i + seed // 2) % 2 == 0 else (NEW, OLD)
                result["order"] = order
                for arm in order:
                    system = SimpleNamespace(generator=generator, playbook=books[arm])
                    result[arm] = evaluation_row(system, processor, "fixed_test", row, seed, arm, run_id)
                return result

            pairs = []
            with ((args.out / "paired_rows.jsonl").open("w", encoding="utf-8") as f,
                  ThreadPoolExecutor(max_workers=12) as pool):
                for pair in pool.map(evaluate, jobs):
                    pairs.append(pair)
                    append_json(f, pair)
                    if len(pairs) % 24 == 0 or len(pairs) == len(jobs):
                        print(f"Tested {len(pairs)}/{len(jobs)} paired batches", flush=True)
        usage = client.summary()
    finally:
        client.close()

    stats = summarize(pairs)
    ci = differences_interval(stats.pop("batch_delta"))
    complete = [p for p in pairs if all(
        p[arm]["status"] == "scored" and p[arm]["predicted_positions"] == 4 and not p[arm]["missing_positions"]
        for arm in ARMS
    )]
    complete_stats = summarize(complete) if complete else None
    if complete_stats:
        complete_stats.pop("batch_delta")
    archived_stats = summarize([{**pair, OLD: pair["archived_ace_12"]} for pair in pairs])
    archived_stats.pop("batch_delta")
    results = {
        "primary": stats, "both_arms_and_reference_complete": complete_stats,
        "comparison_to_archived_12": archived_stats,
        "difference_bootstrap_95pct_pp": ci,
        "trials": {seed: {k: v for k, v in summarize([p for p in pairs if p["seed"] == seed]).items()
                          if k != "batch_delta"} for seed in checkpoints},
        "completed_train_batches": {seed: TRAIN_COUNT + x["completed_additional"] for seed, x in extended.items()},
        "usage": usage,
    }
    write_json(args.out / "summary.json", results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
