"""Run the pinned serial ACE implementation on the frozen FiNER splits."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import hashlib
from importlib import import_module
import json
from pathlib import Path
import random
import statistics
import sys
import threading
import time
from unittest.mock import patch
from uuid import uuid4

from finer_data import load_splits


MODEL = "deepseek-flash"
UPSTREAM = "82709de050e1db6e6ef2f07bcb0393560b94992a"
SEEDS = (17, 29, 43)
ARMS = ("base", "ace_upstream", "ace_source")
LABELS_PER_BATCH = 4


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_json(handle, payload):
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def new_ace(max_tokens):
    from ace import ACE

    return ACE(
        api_provider="deepseek",
        generator_model=MODEL,
        reflector_model=MODEL,
        curator_model=MODEL,
        max_tokens=max_tokens,
        use_bulletpoint_analyzer=False,
    )


def evaluation_row(system, processor, phase, row, trial, arm, run_id):
    from finer_api import BudgetExceeded, LedgerError, request_scope
    from utils import extract_answer

    record = {
        "row_id": row["row_id"],
        "target": row["target"],
        "prediction": None,
        "correct_labels": 0,
        "total_labels": LABELS_PER_BATCH,
        "status": "error",
    }
    try:
        with request_scope(run_id=run_id, trial=trial, arm=arm, phase=phase, row_id=row["row_id"]):
            response, _, _ = system.generator.generate(
                question=row["question"],
                context=row["context"],
                playbook=system.playbook,
                reflection="(empty)",
                use_json_mode=True,
                call_id=f"{phase}_{row['row_id'].replace(':', '_')}",
            )
        prediction = extract_answer(response)
        if not isinstance(prediction, str) or not prediction.strip() or prediction == "No final answer found":
            record["status"] = "invalid_answer"
            return record
        record["prediction"] = prediction
        record["predicted_positions"] = len(prediction.split(","))
        record["missing_positions"] = LABELS_PER_BATCH - sum(
            bool(label.strip()) for label in prediction.split(",")[:LABELS_PER_BATCH]
        )
        count, total = processor._finer_answer_is_correct(prediction, row["target"], return_counts=True)
        if total != LABELS_PER_BATCH:
            raise ValueError("FiNER row does not contain four scoring positions")
        record["correct_labels"] = count
        record["status"] = "scored"
    except (BudgetExceeded, LedgerError) as error:
        record["status"] = "blocked"
        record["error_type"] = type(error).__name__
        record["stop_reason"] = str(error)
    except Exception as error:
        record["error_type"] = type(error).__name__
    return record


def evaluate(system, processor, rows, path, phase, trial, arm, pool, run_id):
    start = time.monotonic()
    jobs = {
        pool.submit(evaluation_row, system, processor, phase, row, trial, arm, run_id): row["row_id"]
        for row in rows
    }
    records = {}
    with path.with_suffix(".progress.jsonl").open("w", encoding="utf-8") as handle:
        for future in as_completed(jobs):
            record = future.result()
            records[jobs[future]] = record
            append_json(handle, record)
    ordered = [records[row["row_id"]] for row in rows]
    with path.with_suffix(".jsonl").open("w", encoding="utf-8") as handle:
        for record in ordered:
            append_json(handle, record)
    correct = sum(record["correct_labels"] for record in ordered)
    total = LABELS_PER_BATCH * len(rows)
    stats = {
        "accuracy": correct / total,
        "correct_labels": correct,
        "total_labels": total,
        "requested_batches": len(rows),
        "scored_batches": sum(record["status"] == "scored" for record in ordered),
        "invalid_answer_batches": sum(record["status"] == "invalid_answer" for record in ordered),
        "error_batches": sum(record["status"] == "error" for record in ordered),
        "blocked_batches": sum(record["status"] == "blocked" for record in ordered),
        "missing_positions": sum(record.get("missing_positions", LABELS_PER_BATCH) for record in ordered),
        "extra_position_batches": sum(record.get("predicted_positions", 0) > LABELS_PER_BATCH for record in ordered),
        "batch_exact_accuracy": sum(record["correct_labels"] == LABELS_PER_BATCH for record in ordered) / len(rows),
        "seconds": time.monotonic() - start,
        "playbook_sha256": hash_text(system.playbook),
    }
    write_json(path.with_suffix(".json"), stats)
    return stats


def fit_arm(system, processor, training_rows, validation_rows, test_rows, path, trial, arm, eval_pool, progress, run_id):
    from finer_api import BudgetExceeded, LedgerError, request_scope

    path.mkdir(parents=True, exist_ok=False)
    initial_playbook = system.playbook
    train_log_dir = path / "training_logs"
    train_log_dir.mkdir()
    config = system._extract_config_params({
        "task_name": "finer",
        "max_num_rounds": 3,
        "curator_frequency": 1,
        "eval_steps": len(training_rows),
        "json_mode": True,
        "no_ground_truth": False,
    })
    start = time.monotonic()
    completed = 0
    step = 0
    stop_reason = None
    progress(trial, arm, "training_started")
    with (path / "training_rows.jsonl").open("w", encoding="utf-8") as training_log:
        for step, row in enumerate(training_rows, 1):
            playbook_before = system.playbook
            next_id_before = system.next_global_id
            evidence = row.get("curator_evidence") if arm == "ace_source" else None
            if arm == "ace_source" and not evidence:
                raise ValueError(f"Missing curator evidence for {row['row_id']}")
            original_curate = system.curator.curate
            curator_observation = {}

            def curate(*args, **kwargs):
                if evidence is not None:
                    kwargs["question_context"] = evidence
                curator_observation["context_sha256"] = hash_text(kwargs["question_context"])
                curator_observation["context_characters"] = len(kwargs["question_context"])
                result = original_curate(*args, **kwargs)
                curator_observation["operations"] = len(result[2])
                return result

            row_for_training = {key: row[key] for key in ("context", "question", "target", "row_id")}
            record = {"step": step, "row_id": row["row_id"]}
            try:
                with (
                    patch.object(system.curator, "curate", new=curate),
                    request_scope(run_id=run_id, trial=trial, arm=arm, phase="train", row_id=row["row_id"]),
                ):
                    pre, post, detail = system._train_single_sample(
                        task_dict=row_for_training,
                        data_processor=processor,
                        step_id=f"train_e_1_s_{step}",
                        epoch=1,
                        step=step,
                        usage_log_path=str(path / "bullet_usage_log.jsonl"),
                        log_dir=str(train_log_dir),
                        config_params=config,
                        total_samples=len(training_rows),
                    )
                record.update(status="complete", pre_prediction=pre, post_prediction=post, detail=detail)
                completed += 1
            except Exception as error:
                # A failed transport call may follow an in-memory update. Keep the last complete state.
                system.playbook = playbook_before
                system.next_global_id = next_id_before
                record.update(status="error", error_type=type(error).__name__)
                if isinstance(error, (BudgetExceeded, LedgerError)):
                    stop_reason = type(error).__name__
                    record["status"] = "blocked"
                    record["stop_reason"] = str(error)
            record["curator"] = curator_observation
            record["playbook_sha256"] = hash_text(system.playbook)
            append_json(training_log, record)
            write_json(path / "checkpoint.json", {
                "trial": trial,
                "arm": arm,
                "last_attempted_step": step,
                "completed_training_batches": completed,
                "next_global_id": system.next_global_id,
                "playbook": system.playbook,
                "playbook_sha256": record["playbook_sha256"],
            })
            progress(trial, arm, "training_step", step=step, status=record["status"])
            if stop_reason:
                break

    train_seconds = time.monotonic() - start
    final_playbook = system.playbook
    (path / "final_playbook.txt").write_text(final_playbook, encoding="utf-8")
    summary = {
        "trial": trial,
        "arm": arm,
        "training_seconds": train_seconds,
        "training_requested_batches": len(training_rows),
        "training_attempted_batches": step,
        "training_completed_batches": completed,
        "training_error_batches": step - completed,
        "initial_playbook_sha256": hash_text(initial_playbook),
        "final_playbook_sha256": hash_text(final_playbook),
    }
    if stop_reason:
        summary.update(status="incomplete", error_type=stop_reason)
        write_json(path / "summary.json", summary)
        progress(trial, arm, "arm_interrupted", error_type=stop_reason)
        return summary
    progress(trial, arm, "validation_started")
    validation = evaluate(system, processor, validation_rows, path / "validation", "validation", trial, arm, eval_pool, run_id)
    summary["validation"] = validation
    if validation["blocked_batches"]:
        summary.update(status="incomplete", error_type="validation_blocked")
        write_json(path / "summary.json", summary)
        progress(trial, arm, "arm_interrupted", error_type="validation_blocked")
        return summary
    # With one terminal checkpoint the upstream strict improvement rule compares to zero.
    selected = "final" if validation["accuracy"] > 0 else "initial"
    system.best_playbook = final_playbook if selected == "final" else initial_playbook
    system.playbook = system.best_playbook
    (path / "selected_playbook.txt").write_text(system.playbook, encoding="utf-8")
    progress(trial, arm, "test_started", selected_checkpoint=selected)
    test = evaluate(system, processor, test_rows, path / "test", "test", trial, arm, eval_pool, run_id)
    summary.update(
        status="incomplete" if test["blocked_batches"] else "complete",
        selected_checkpoint=selected,
        selected_playbook_sha256=hash_text(system.playbook),
        test=test,
    )
    write_json(path / "summary.json", summary)
    progress(trial, arm, "arm_finished", accuracy=test["accuracy"], error_batches=test["error_batches"])
    return summary


def summarize(results, seeds):
    def stats(values):
        return {
            "values": values,
            "mean": statistics.mean(values) if values else None,
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        }

    indexed = {(result["trial"], result["arm"]): result for result in results if result.get("status") == "complete"}
    return {
        "complete_trial_seeds": [seed for seed in seeds if all((seed, arm) in indexed for arm in ARMS)],
        "incomplete_trial_seeds": [seed for seed in seeds if any((seed, arm) not in indexed for arm in ARMS)],
        "accuracy_by_arm": {
            arm: stats([indexed[(seed, arm)]["test"]["accuracy"] for seed in seeds if (seed, arm) in indexed])
            for arm in ARMS
        },
        "paired_differences_percentage_points": {
            f"{treatment}_minus_{control}": stats([
                100 * (indexed[(seed, treatment)]["test"]["accuracy"] - indexed[(seed, control)]["test"]["accuracy"])
                for seed in seeds if (seed, treatment) in indexed and (seed, control) in indexed
            ])
            for treatment, control in (("ace_upstream", "base"), ("ace_source", "ace_upstream"))
        },
        "runs": results,
    }


def usage_by_phase(ledger, run_id):
    """Join settled usage to its reserved attempt and retain unresolved attempts."""
    records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line]
    settled = {record["attempt_id"]: record for record in records if record["event"] == "settle"}
    groups = {}
    for record in records:
        if record["event"] != "reserve" or record.get("run_id") != run_id:
            continue
        key = (record["trial"], record["arm"], record["phase"])
        group = groups.setdefault(key, {
            "trial": key[0], "arm": key[1], "phase": key[2], "attempts": 0,
            "settled_attempts": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "observed_cache_hit_tokens": 0, "unknown_cache_breakdown_attempts": 0,
            "settled_peak_cost_nano_cny": 0, "unresolved_reserved_nano_cny": 0,
        })
        group["attempts"] += 1
        settlement = settled.get(record["attempt_id"])
        if settlement is None:
            group["unresolved_reserved_nano_cny"] += record["reserved_nano_cny"]
            continue
        usage = settlement["usage"]
        group["settled_attempts"] += 1
        group["prompt_tokens"] += usage["prompt_tokens"]
        group["completion_tokens"] += usage["completion_tokens"]
        group["observed_cache_hit_tokens"] += usage["prompt_cache_hit_tokens"] or 0
        group["unknown_cache_breakdown_attempts"] += usage["prompt_cache_hit_tokens"] is None
        group["settled_peak_cost_nano_cny"] += settlement["cost_peak_nano_cny"]
    return [groups[key] for key in sorted(groups)]


def run_experiment(options, api_client, splits):
    from eval.finance.data_processor import DataProcessor
    from finer_api import BASE_URL, ace_timed_llm_call

    seeds = options.seeds[:1] if options.pilot else options.seeds
    root = options.output_dir
    run_id = uuid4().hex
    processor = DataProcessor("finer")
    root.mkdir(parents=True, exist_ok=False)
    progress_lock = threading.Lock()
    row_orders = {}
    for seed in seeds:
        rows = list(splits["train"])
        random.Random(seed).shuffle(rows)
        row_orders[seed] = rows
        (root / f"trial_{seed}").mkdir()
    evaluation_splits = {
        split: [{key: row[key] for key in ("context", "question", "target", "row_id")} for row in splits[split]]
        for split in ("val", "test")
    }
    write_json(root / "run_manifest.json", {
        "upstream_commit": UPSTREAM,
        "run_id": run_id,
        "model": MODEL,
        "base_url": BASE_URL,
        "temperature": 0,
        "thinking": "disabled",
        "json_mode": True,
        "pilot": options.pilot,
        "scope": options.scope,
        "seeds": seeds,
        "train_order": {seed: [row["row_id"] for row in rows] for seed, rows in row_orders.items()},
        "validation_ids": [row["row_id"] for row in evaluation_splits["val"]],
        "test_ids": [row["row_id"] for row in evaluation_splits["test"]],
        "max_num_rounds": 3,
        "max_output_tokens": options.max_output_tokens,
        "max_input_tokens": options.max_input_tokens,
        "budget_cny": options.budget_cny,
        "fit_workers": options.fit_workers,
        "eval_workers": options.eval_workers,
        "ledger": str(options.ledger.resolve()),
    })

    def progress(trial, arm, event, **details):
        record = {"time": time.time(), "trial": trial, "arm": arm, "event": event, **details}
        with progress_lock, (root / "progress.jsonl").open("a", encoding="utf-8") as handle:
            append_json(handle, record)
            print(f"[{trial} {arm}] {event} {details}", flush=True)

    def baseline(seed, eval_pool):
        path = root / f"trial_{seed}" / "base"
        path.mkdir()
        progress(seed, "base", "test_started")
        system = new_ace(options.max_output_tokens)
        test = evaluate(system, processor, evaluation_splits["test"], path / "test", "test", seed, "base", eval_pool, run_id)
        summary = {"trial": seed, "arm": "base", "test": test,
                   "status": "incomplete" if test["blocked_batches"] else "complete"}
        write_json(path / "summary.json", summary)
        progress(seed, "base", "arm_finished", accuracy=test["accuracy"], error_batches=test["error_batches"])
        return summary

    def fit(seed, arm, eval_pool):
        return fit_arm(
            new_ace(options.max_output_tokens), processor,
            row_orders[seed], evaluation_splits["val"], evaluation_splits["test"],
            root / f"trial_{seed}" / arm, seed, arm, eval_pool, progress, run_id,
        )

    results = []

    def collect(jobs):
        for future in as_completed(jobs):
            seed, arm = jobs[future]
            try:
                results.append(future.result())
            except Exception as error:
                results.append({"trial": seed, "arm": arm, "status": "incomplete", "error_type": type(error).__name__})
                progress(seed, arm, "arm_interrupted", error_type=type(error).__name__)
            write_json(root / "summary.json", summarize(results, seeds))

    with ExitStack() as stack:
        stack.enter_context(patch.object(import_module("ace.ace"), "initialize_clients",
                                         new=lambda provider: (api_client, api_client, api_client)))
        for role in ("generator", "reflector", "curator"):
            stack.enter_context(patch.object(import_module(f"ace.core.{role}"), "timed_llm_call",
                                             new=ace_timed_llm_call))
        eval_pool = stack.enter_context(ThreadPoolExecutor(max_workers=options.eval_workers))
        arm_pool = stack.enter_context(ThreadPoolExecutor(max_workers=options.fit_workers))
        # Each trial receives exactly one fresh empty-playbook baseline before adaptation.
        collect({arm_pool.submit(baseline, seed, eval_pool): (seed, "base") for seed in seeds})
        ready_seeds = {result["trial"] for result in results if result["status"] == "complete"}
        jobs = {}
        for index, seed in enumerate(seeds):
            order = ARMS[1:] if index % 2 == 0 else tuple(reversed(ARMS[1:]))
            for arm in order:
                if seed not in ready_seeds:
                    results.append({"trial": seed, "arm": arm, "status": "incomplete",
                                    "error_type": "baseline_incomplete"})
                    progress(seed, arm, "arm_interrupted", error_type="baseline_incomplete")
                    continue
                jobs[arm_pool.submit(fit, seed, arm, eval_pool)] = (seed, arm)
        collect(jobs)

    results.sort(key=lambda result: (seeds.index(result["trial"]), ARMS.index(result["arm"])))
    summary = summarize(results, seeds)
    summary["usage_by_trial_arm_phase"] = usage_by_phase(options.ledger, run_id)
    summary["shared_budget"] = api_client.summary()
    write_json(root / "summary.json", summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="Use the reserved 2/2/2 pilot and one trial")
    parser.add_argument("--scope", choices=("compact", "standard"), default="compact")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--output-dir", "--out", type=Path)
    parser.add_argument("--ledger", type=Path, help="Reuse the same ledger for sequential pilot and formal runs")
    parser.add_argument("--env-file", type=Path, help="Read credentials into the process environment")
    parser.add_argument("--budget-cny", type=float, default=50)
    parser.add_argument("--max-attempts", type=int, default=2000)
    parser.add_argument("--max-input-tokens", type=int, default=16000)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--fit-workers", type=int, default=6)
    parser.add_argument("--eval-workers", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true", help="Validate the frozen split manifest without a model client")
    options = parser.parse_args()
    if len(set(options.seeds)) != len(options.seeds):
        parser.error("trial seeds must be distinct")
    if options.fit_workers < 1 or options.eval_workers < 1:
        parser.error("worker counts must be positive")
    if not options.dry_run and options.output_dir is None:
        parser.error("--output-dir is required")
    if options.ledger is None and options.output_dir is not None:
        options.ledger = options.output_dir.parent / "deepseek_ledger.jsonl"
    if options.output_dir is not None and options.ledger.resolve().is_relative_to(options.output_dir.resolve()):
        parser.error("--ledger must be outside the new output directory so it can be shared across runs")
    return options


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    options = parse_args()
    splits = load_splits(pilot=options.pilot, scope=options.scope)
    if options.dry_run:
        counts = {split: len(rows) for split, rows in splits.items()}
        trials = 1 if options.pilot else len(options.seeds)
        limit = trials * (18 * counts["train"] + 2 * counts["val"] + 3 * counts["test"])
        print(json.dumps({"split_counts": counts, "trials": trials, "max_logical_calls": limit}))
        return 0
    if options.env_file is not None:
        from dotenv import load_dotenv

        load_dotenv(options.env_file, override=False)
    from finer_api import BudgetedDeepSeekClient

    client = BudgetedDeepSeekClient(
        options.ledger,
        budget_cny=options.budget_cny,
        max_attempts=options.max_attempts,
        max_input_tokens=options.max_input_tokens,
        max_output_tokens=options.max_output_tokens,
        timeout_s=options.timeout,
    )
    try:
        summary = run_experiment(options, client, splits)
    finally:
        client.close()
    for result in summary["runs"]:
        if result["status"] != "complete" or result["test"]["error_batches"]:
            return 2
        if result.get("training_error_batches") or result.get("validation", {}).get("error_batches"):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

