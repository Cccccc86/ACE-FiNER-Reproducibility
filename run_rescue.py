"""Compare frozen ACE with a single output-cap rescue on new FiNER batches."""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from runpy import run_path
import statistics
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from finer_api import (
    BASE_URL, MISS_PRICE, NANO_PER_CNY, OUTPUT_PRICE, BudgetedDeepSeekClient,
    BudgetExceeded, LedgerError, TruncatedCompletion, ace_timed_llm_call, request_scope,
)
from run_finer import LABELS_PER_BATCH, MODEL, SEEDS, UPSTREAM, append_json, hash_text, usage_by_phase, write_json


ROOT = Path(__file__).resolve().parent
INITIAL_CAP = 1536
RESCUE_CAP = 3072
INPUT_CAP = 32000
MAX_ATTEMPTS = 2000
BUDGET_CNY = 50
BATCHES = 64


def load_playbooks(checkpoints_dir):
    manifest = json.loads((checkpoints_dir / "run_manifest.json").read_text(encoding="utf-8"))
    expected = {
        "upstream_commit": UPSTREAM, "model": MODEL, "base_url": BASE_URL,
        "temperature": 0, "thinking": "disabled", "json_mode": True,
        "seeds": list(SEEDS), "max_output_tokens": INITIAL_CAP, "pilot": False,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("The original run manifest does not match the frozen ACE protocol")
    playbooks = {}
    for seed in SEEDS:
        directory = checkpoints_dir / f"trial_{seed}" / "ace_upstream"
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        playbook = (directory / "selected_playbook.txt").read_text(encoding="utf-8")
        if (summary.get("trial") != seed or summary.get("arm") != "ace_upstream"
                or summary.get("status") != "complete" or not playbook.strip()
                or summary.get("selected_playbook_sha256") != hash_text(playbook)):
            raise ValueError(f"Invalid frozen ACE checkpoint for trial {seed}")
        playbooks[seed] = playbook
    return playbooks


def capacity_plan(rows, playbooks):
    # This constants-only module can be read without initializing ACE's tokenizer.
    template = run_path(str(ROOT / "ace" / "prompts" / "generator.py"))["GENERATOR_PROMPT"]
    bounds = [
        544 + len(template.format(book, "(empty)", row["question"], row["context"]).encode("utf-8"))
        for book in playbooks.values() for row in rows
    ]
    # One ordinary transport retry is allowed at each of the two output caps.
    worst_nano = sum(2 * (2 * bound * MISS_PRICE + (INITIAL_CAP + RESCUE_CAP) * OUTPUT_PRICE)
                     for bound in bounds)
    return {
        "shared_initial_calls": len(bounds), "maximum_rescue_calls": len(bounds),
        "maximum_api_attempts": 4 * len(bounds), "maximum_input_bound": max(bounds),
        "maximum_incremental_reservation_cny": worst_nano / NANO_PER_CNY,
    }


def evaluate_attempt(generator, processor, row, playbook, seed, phase, run_id):
    from utils import extract_answer

    started = time.monotonic()
    record = {"status": "error", "prediction": None, "correct_labels": 0}
    try:
        with request_scope(run_id=run_id, trial=seed, arm="ace_output_rescue", phase=phase, row_id=row["row_id"]):
            response, _, _ = generator.generate(
                question=row["question"], context=row["context"], playbook=playbook,
                reflection="(empty)", use_json_mode=True,
                call_id=f"{phase}_{row['row_id'].replace(':', '_')}",
            )
        prediction = extract_answer(response)
        if not isinstance(prediction, str) or not prediction.strip() or prediction == "No final answer found":
            record["status"] = "invalid_answer"
        else:
            correct, total = processor._finer_answer_is_correct(prediction, row["target"], return_counts=True)
            if total != LABELS_PER_BATCH:
                raise ValueError("FiNER row does not contain four scoring positions")
            positions = prediction.split(",")
            record.update(
                status="scored", prediction=prediction, correct_labels=correct,
                predicted_positions=len(positions),
                missing_positions=LABELS_PER_BATCH - sum(bool(label.strip()) for label in positions[:LABELS_PER_BATCH]),
            )
    except TruncatedCompletion:
        record.update(status="truncated", finish_reason="length")
    except (BudgetExceeded, LedgerError) as exc:
        record.update(status="blocked", error_type=type(exc).__name__)
    except Exception as exc:
        record["error_type"] = type(exc).__name__
    record["elapsed_seconds"] = time.monotonic() - started
    return record


def evaluate_pair(initial, rescue, processor, row, playbook, seed, run_id):
    control = evaluate_attempt(initial, processor, row, playbook, seed, "initial", run_id)
    retry = None
    if control["status"] == "truncated":
        retry = evaluate_attempt(rescue, processor, row, playbook, seed, "rescue", run_id)
    treatment = retry if retry is not None else control
    return {
        "trial": seed, "row_id": row["row_id"], "target": row["target"], "total_labels": LABELS_PER_BATCH,
        "control": control, "rescue": retry, "treatment_status": treatment["status"],
        "treatment_correct_labels": treatment["correct_labels"],
        "gained_correct_labels": treatment["correct_labels"] - control["correct_labels"],
    }


def summarize_trial(seed, pairs):
    controls = [row["control"] for row in pairs]
    rescues = [row["rescue"] for row in pairs if row["rescue"] is not None]
    total = sum(row["total_labels"] for row in pairs)
    control_correct = sum(row["correct_labels"] for row in controls)
    treatment_correct = sum(row["treatment_correct_labels"] for row in pairs)
    return {
        "trial": seed, "batches": len(pairs), "total_labels": total,
        "control_correct_labels": control_correct, "treatment_correct_labels": treatment_correct,
        "control_accuracy": control_correct / total, "treatment_accuracy": treatment_correct / total,
        "paired_gain_pp": 100 * (treatment_correct - control_correct) / total,
        "initial_truncations": len(rescues), "rescue_calls": len(rescues),
        "rescues_scored": sum(rescue["status"] == "scored" for rescue in rescues),
        "rescue_truncations": sum(rescue["status"] == "truncated" for rescue in rescues),
        "improved_batches": sum(row["gained_correct_labels"] > 0 for row in pairs),
        "recovered_correct_labels": treatment_correct - control_correct,
        "blocked_attempts": sum(attempt["status"] == "blocked" for attempt in controls + rescues),
        "other_error_attempts": sum(attempt["status"] == "error" for attempt in controls + rescues),
        "invalid_answer_attempts": sum(attempt["status"] == "invalid_answer" for attempt in controls + rescues),
        "added_logical_latency_seconds": sum(rescue["elapsed_seconds"] for rescue in rescues),
    }


def run_experiment(client, rows, playbooks, output_dir, ledger, workers, *, show_progress=True):
    from ace.core import generator as generator_module
    from eval.finance import data_processor

    started = time.monotonic()
    plan = capacity_plan(rows, playbooks)
    before = client.summary()
    if (plan["maximum_input_bound"] > INPUT_CAP
            or plan["maximum_incremental_reservation_cny"] > before["remaining_cny"]
            or plan["maximum_api_attempts"] + before["attempts"] > before["max_attempts"]
            or before["usage_bound_violations"]):
        raise BudgetExceeded("The shared ledger cannot safely cover the frozen worst-case plan")
    output_dir.mkdir(parents=True, exist_ok=False)
    run_id = uuid4().hex
    write_json(output_dir / "run_manifest.json", {
        "run_id": run_id, "upstream_commit": UPSTREAM, "model": MODEL, "base_url": BASE_URL,
        "temperature": 0, "thinking": "disabled", "json_mode": True, "workers": workers,
        "seeds": list(playbooks), "test_ids": [row["row_id"] for row in rows],
        "checkpoint_sha256": {seed: hash_text(book) for seed, book in playbooks.items()},
        "initial_max_tokens": INITIAL_CAP, "rescue_max_tokens": RESCUE_CAP, "input_cap": INPUT_CAP,
        "rescue_trigger": "finish_reason=length", "shared_initial_response": True,
        "capacity_plan": plan, "shared_budget_before": before,
    })
    initial = generator_module.Generator(client, "deepseek", MODEL, max_tokens=INITIAL_CAP)
    rescue = generator_module.Generator(client, "deepseek", MODEL, max_tokens=RESCUE_CAP)
    processor = data_processor.DataProcessor("finer")
    results = {}
    truncations = 0
    # Keep the original tag scorer while restricting it to literal parsing of untrusted text.
    with (
        patch.object(generator_module, "timed_llm_call", new=ace_timed_llm_call),
        patch.object(data_processor, "eval", new=ast.literal_eval, create=True),
        (output_dir / "progress.jsonl").open("w", encoding="utf-8") as progress,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        jobs = [pool.submit(evaluate_pair, initial, rescue, processor, row, book, seed, run_id)
                for seed, book in playbooks.items() for row in rows]
        for future in as_completed(jobs):
            pair = future.result()
            results[pair["trial"], pair["row_id"]] = pair
            append_json(progress, pair)
            truncations += pair["rescue"] is not None
            if show_progress and (len(results) % 12 == 0 or len(results) == len(jobs)):
                print(f"Paired batches {len(results)}/{len(jobs)}, truncations {truncations}", flush=True)

    trials = []
    for seed in playbooks:
        trial_dir = output_dir / f"trial_{seed}"
        trial_dir.mkdir()
        pairs = [results[seed, row["row_id"]] for row in rows]
        with (trial_dir / "paired_rows.jsonl").open("w", encoding="utf-8") as handle:
            for pair in pairs:
                append_json(handle, pair)
        trial = summarize_trial(seed, pairs)
        write_json(trial_dir / "summary.json", trial)
        trials.append(trial)

    usage = usage_by_phase(ledger, run_id)
    phase_usage = {}
    for phase in ("initial", "rescue"):
        groups = [group for group in usage if group["phase"] == phase]
        totals = {key: sum(group[key] for group in groups) for key in (
            "attempts", "settled_attempts", "prompt_tokens", "completion_tokens",
            "observed_cache_hit_tokens", "unknown_cache_breakdown_attempts",
            "settled_peak_cost_nano_cny", "unresolved_reserved_nano_cny",
        )}
        totals["committed_cny"] = (totals["settled_peak_cost_nano_cny"] + totals["unresolved_reserved_nano_cny"]) / NANO_PER_CNY
        phase_usage[phase] = totals

    def aggregate(key):
        values = [trial[key] for trial in trials]
        return {"values": values, "mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values) if len(values) > 1 else None}

    summary = {
        "run_id": run_id, "status": "incomplete" if any(trial["blocked_attempts"] for trial in trials) else "complete",
        "trials": trials, "control_accuracy": aggregate("control_accuracy"),
        "treatment_accuracy": aggregate("treatment_accuracy"), "paired_gain_pp": aggregate("paired_gain_pp"),
        "initial_truncations": sum(trial["initial_truncations"] for trial in trials),
        "rescues_scored": sum(trial["rescues_scored"] for trial in trials),
        "recovered_correct_labels": sum(trial["recovered_correct_labels"] for trial in trials),
        "usage_by_phase": phase_usage, "usage_by_trial_phase": usage,
        "shared_budget_before": before, "shared_budget_after": client.summary(),
        "wall_seconds": time.monotonic() - started,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def smoke_test():
    import tiktoken

    calls = {}
    answers = {"stop": ("stop", '{"final_answer":"Alpha,Beta,Missing,Delta"}'),
               "length": ("length", "{"), "blank": ("stop", ""),
               "filtered": ("content_filter", "{}"), "invalid": ("stop", "{}")}

    def create(**params):
        prompt = params["messages"][0]["content"]
        case = next(case for case in (*answers, "network", "length_twice", "missing_usage") if f"<case-{case}>" in prompt)
        calls.setdefault(case, []).append(params)
        cap = params["max_tokens"]
        if case == "network":
            raise ConnectionError("private fake provider detail")
        if case in ("length_twice", "missing_usage") or (case == "length" and cap == INITIAL_CAP):
            reason, content = answers["length"]
        else:
            reason, content = answers["stop"] if case == "length" else answers[case]
        usage = None if case == "missing_usage" else {
            "prompt_tokens": 80, "completion_tokens": cap if reason == "length" else 12,
        }
        return {"choices": [{"finish_reason": reason, "message": {"content": content}}], "usage": usage}

    with TemporaryDirectory(dir=Path.cwd()) as directory:
        root = Path(directory)
        ledger = root / "ledger.jsonl"
        fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        client = BudgetedDeepSeekClient(ledger, max_input_tokens=INPUT_CAP, max_output_tokens=RESCUE_CAP, client=fake)
        rows = [{"row_id": case, "question": f"<case-{case}>", "context": "", "target": "Alpha,Beta,Gamma,Delta"}
                for case in (*answers, "network", "length_twice", "missing_usage")]
        with patch.object(tiktoken, "get_encoding", new=lambda name: None):
            result = run_experiment(client, rows, {17: "frozen playbook"}, root / "out", ledger, 1, show_progress=False)
        assert result["initial_truncations"] == 2 and result["rescues_scored"] == 1
        assert result["recovered_correct_labels"] == 3
        assert result["trials"][0]["control_correct_labels"] == 3
        assert result["trials"][0]["treatment_correct_labels"] == 6
        assert result["usage_by_phase"]["initial"]["attempts"] == len(rows) + 1
        assert result["usage_by_phase"]["rescue"]["attempts"] == 2
        for case, params in calls.items():
            caps = [item["max_tokens"] for item in params]
            expected = [INITIAL_CAP, RESCUE_CAP] if case in ("length", "length_twice") else [INITIAL_CAP]
            if case == "network":
                expected.append(INITIAL_CAP)
            assert caps == expected, (case, caps)
            if case in ("length", "length_twice"):
                assert {key: value for key, value in params[0].items() if key != "max_tokens"} == {
                    key: value for key, value in params[1].items() if key != "max_tokens"
                }
        assert "private fake provider detail" not in ledger.read_text(encoding="utf-8")
    print("Offline smoke test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Send paid DeepSeek requests under the shared ledger")
    mode.add_argument("--dry-run", action="store_true", help="Verify rows, checkpoints and budget plan (default)")
    mode.add_argument("--smoke-test", action="store_true", help="Exercise the full runner with a fake provider")
    parser.add_argument("--output-dir", "--out", type=Path)
    parser.add_argument("--ledger", type=Path, help="Reuse the ledger from the prior pilot and formal run")
    parser.add_argument("--checkpoints-dir", type=Path, default=ROOT / "evidence" / "formal")
    parser.add_argument("--env-file", type=Path, help="Read credentials into the process environment")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.smoke_test:
        smoke_test()
        return 0
    if args.run:
        if args.output_dir is None or args.ledger is None:
            parser.error("--run requires --output-dir and --ledger")
        if args.output_dir.exists():
            parser.error("--output-dir must be a new path")
        if args.ledger.resolve().is_relative_to(args.output_dir.resolve()):
            parser.error("--ledger must be outside the output directory")
    from rescue_data import load_rescue_test

    rows = load_rescue_test()
    if len(rows) != BATCHES or len({row["row_id"] for row in rows}) != BATCHES:
        raise ValueError(f"Expected {BATCHES} distinct frozen rescue batches")
    playbooks = load_playbooks(args.checkpoints_dir)
    if not args.run:
        print(json.dumps(capacity_plan(rows, playbooks), indent=2))
        return 0
    if args.env_file is not None:
        from dotenv import load_dotenv

        load_dotenv(args.env_file, override=False)
    client = BudgetedDeepSeekClient(
        args.ledger, budget_cny=BUDGET_CNY, max_attempts=MAX_ATTEMPTS,
        max_input_tokens=INPUT_CAP, max_output_tokens=RESCUE_CAP,
    )
    try:
        result = run_experiment(client, rows, playbooks, args.output_dir, args.ledger, args.workers)
    finally:
        client.close()
    print(json.dumps({key: result[key] for key in (
        "status", "control_accuracy", "treatment_accuracy", "paired_gain_pp", "initial_truncations",
        "recovered_correct_labels", "usage_by_phase",
    )}, indent=2))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
