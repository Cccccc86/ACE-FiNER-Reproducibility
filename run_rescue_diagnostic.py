"""Replay observed FiNER truncations at both caps as a post-hoc diagnostic."""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from runpy import run_path
import time
from unittest.mock import patch
from uuid import uuid4

from finer_api import BASE_URL, NANO_PER_CNY, BudgetedDeepSeekClient, BudgetExceeded, ace_timed_llm_call, request_scope
from finer_data import load_splits
from rescue_data import load_rescue_test
from run_finer import MODEL, UPSTREAM, append_json, hash_text, usage_by_phase, write_json
from run_rescue import (
    BUDGET_CNY, INITIAL_CAP, INPUT_CAP, LABELS_PER_BATCH, MAX_ATTEMPTS, RESCUE_CAP, ROOT,
    capacity_plan, evaluate_attempt, load_playbooks,
)


REPEATS = 3
CAPS = (INITIAL_CAP, RESCUE_CAP)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def empty_playbook():
    # The pinned method returns a string literal. Read it without loading the tokenizer.
    tree = ast.parse((ROOT / "ace" / "ace.py").read_text(encoding="utf-8"))
    method = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "_initialize_empty_playbook")
    return ast.literal_eval(method.body[-1].value)


def load_cases(formal_dir, rescue_dir, ledger):
    formal = read_json(formal_dir / "run_manifest.json")
    rescue = read_json(rescue_dir / "run_manifest.json")
    playbooks = load_playbooks(formal_dir)
    expected = {"upstream_commit": UPSTREAM, "model": MODEL, "base_url": BASE_URL,
                "temperature": 0, "thinking": "disabled", "json_mode": True,
                "initial_max_tokens": INITIAL_CAP, "rescue_max_tokens": RESCUE_CAP,
                "shared_initial_response": True}
    if any(rescue.get(key) != value for key, value in expected.items()) or formal.get("scope") != "compact":
        raise ValueError("Archived experiment settings do not match the diagnostic protocol")
    formal_rows = {row["row_id"]: row for row in load_splits(pilot=False, scope="compact")["test"]}
    rescue_rows = {row["row_id"]: row for row in load_rescue_test()}
    if list(formal_rows) != formal["test_ids"] or list(rescue_rows) != rescue["test_ids"]:
        raise ValueError("Archived test rows differ from the frozen splits")
    origins = {formal["run_id"]: "formal", rescue["run_id"]: "rescue"}
    events = read_jsonl(ledger)
    reserves = {event["attempt_id"]: event for event in events
                if event["event"] == "reserve" and event.get("run_id") in origins}
    selected = []
    for event in events:
        if event["event"] != "settle" or event.get("finish_reason") != "length":
            continue
        reserve = reserves.get(event["attempt_id"])
        if reserve is None:
            continue
        cohort = origins[reserve["run_id"]]
        if reserve["phase"] != ("test" if cohort == "formal" else "initial"):
            continue
        if (reserve["role"] != "generator" or reserve["model_requested"] != MODEL
                or reserve["max_tokens"] != INITIAL_CAP or reserve["temperature"] != 0
                or reserve["thinking"] != "disabled" or not reserve["json_mode"]
                or event["usage"]["completion_tokens"] != INITIAL_CAP or event["limit_violation"]):
            raise ValueError("Unexpected ledger settings for an archived truncation")
        selected.append((cohort, reserve))
    if sum(cohort == "formal" for cohort, _ in selected) != 9 or sum(cohort == "rescue" for cohort, _ in selected) != 1:
        raise ValueError("Expected nine formal and one primary rescue truncation")

    cases = []
    for cohort, reserve in sorted(selected, key=lambda item: (item[0], item[1]["trial"], item[1]["arm"], item[1]["row_id"])):
        seed, arm, row_id = (reserve[key] for key in ("trial", "arm", "row_id"))
        if cohort == "formal":
            path = formal_dir / f"trial_{seed}" / arm
            summary = read_json(path / "summary.json")
            row = formal_rows[row_id]
            original = next(record for record in read_jsonl(path / "test.jsonl") if record["row_id"] == row_id)
            if (summary["status"] != "complete" or summary["trial"] != seed or summary["arm"] != arm
                    or original["status"] != "error" or original["correct_labels"] != 0):
                raise ValueError("Formal truncation does not match archived scoring")
            book = empty_playbook() if arm == "base" else (path / "selected_playbook.txt").read_text(encoding="utf-8")
            expected_hash = summary["test"]["playbook_sha256"]
        else:
            if arm != "ace_output_rescue":
                raise ValueError("Unexpected arm for the primary rescue truncation")
            row = rescue_rows[row_id]
            original = next(record for record in read_jsonl(rescue_dir / f"trial_{seed}" / "paired_rows.jsonl")
                            if record["row_id"] == row_id)
            control = original["control"]
            if control["status"] != "truncated" or control["finish_reason"] != "length" or control["correct_labels"] != 0:
                raise ValueError("Primary rescue truncation does not match archived scoring")
            book = playbooks[seed]
            expected_hash = rescue["checkpoint_sha256"][str(seed)]
        if row["target"] != original["target"] or hash_text(book) != expected_hash:
            raise ValueError("Archived diagnostic case failed label or playbook verification")
        cases.append({
            "case_id": f"{cohort}.{seed}.{arm}.{row_id}", "cohort": cohort, "trial": seed, "arm": arm,
            "archived_attempt_id": reserve["attempt_id"], "archived_run_id": reserve["run_id"],
            "row": row, "playbook": book,
        })
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate archived diagnostic case")
    return cases


def diagnostic_plan(cases):
    plans = [capacity_plan([case["row"]], {case["trial"]: case["playbook"]}) for case in cases]
    return {
        "cases": len(cases), "repeats_per_case": REPEATS,
        "logical_calls": len(cases) * REPEATS * len(CAPS),
        "maximum_api_attempts": REPEATS * sum(plan["maximum_api_attempts"] for plan in plans),
        "maximum_input_bound": max(plan["maximum_input_bound"] for plan in plans),
        "maximum_incremental_reservation_cny": REPEATS * sum(plan["maximum_incremental_reservation_cny"] for plan in plans),
    }


def replay_pair(generators, processor, case, repeat, order, run_id):
    row = case["row"]
    sample_id = f"{case['case_id']}.r{repeat}"
    with request_scope(condition=case["arm"], sample_id=sample_id):
        by_cap = {str(cap): evaluate_attempt(generators[cap], processor, row, case["playbook"],
                                            case["trial"], f"cap_{cap}", run_id) for cap in order}
    return {
        "case_id": case["case_id"], "sample_id": sample_id, "cohort": case["cohort"],
        "trial": case["trial"], "arm": case["arm"], "row_id": row["row_id"], "target": row["target"],
        "repeat": repeat, "cap_order": order, "by_cap": by_cap,
        "expanded_minus_old_correct_labels": by_cap[str(RESCUE_CAP)]["correct_labels"] - by_cap[str(INITIAL_CAP)]["correct_labels"],
    }


def summarize_pairs(pairs):
    by_cap = {}
    for cap in CAPS:
        attempts = [pair["by_cap"][str(cap)] for pair in pairs]
        correct = sum(attempt["correct_labels"] for attempt in attempts)
        denominator = len(attempts) * LABELS_PER_BATCH
        by_cap[str(cap)] = {
            "batches": len(attempts), "correct_labels": correct, "total_labels": denominator,
            "accuracy_on_selected_cases": correct / denominator,
            **{f"{status}_batches": sum(attempt["status"] == status for attempt in attempts)
               for status in ("scored", "truncated", "invalid_answer", "error", "blocked")},
        }
    differences = [pair["expanded_minus_old_correct_labels"] for pair in pairs]
    return {"by_cap": by_cap, "expanded_wins": sum(delta > 0 for delta in differences),
            "ties": sum(delta == 0 for delta in differences), "old_cap_wins": sum(delta < 0 for delta in differences)}


def budget_snapshot(client):
    return {key: value for key, value in client.summary().items() if key != "ledger_path"}


def run_diagnostic(client, cases, output_dir, ledger, workers):
    from ace.core import generator as generator_module
    from eval.finance import data_processor

    started = time.monotonic()
    plan = diagnostic_plan(cases)
    before = budget_snapshot(client)
    if (plan["maximum_input_bound"] > INPUT_CAP or before["usage_bound_violations"]
            or plan["maximum_incremental_reservation_cny"] > before["remaining_cny"]
            or plan["maximum_api_attempts"] + before["attempts"] > before["max_attempts"]):
        raise BudgetExceeded("Shared ledger cannot cover the diagnostic worst-case plan")
    output_dir.mkdir(parents=True, exist_ok=False)
    run_id = uuid4().hex
    template = run_path(str(ROOT / "ace" / "prompts" / "generator.py"))["GENERATOR_PROMPT"]
    write_json(output_dir / "run_manifest.json", {
        "run_id": run_id, "post_hoc_selected_on_truncation": True, "capacity_plan": plan,
        "model": MODEL, "base_url": BASE_URL, "upstream_commit": UPSTREAM,
        "temperature": 0, "thinking": "disabled", "json_mode": True, "caps": CAPS,
        "cases": [{**{key: value for key, value in case.items() if key not in ("row", "playbook")},
                   "row_id": case["row"]["row_id"], "playbook_sha256": hash_text(case["playbook"]),
                   "prompt_sha256": hash_text(template.format(case["playbook"], "(empty)",
                                                              case["row"]["question"], case["row"]["context"]))}
                  for case in cases],
    })
    generators = {cap: generator_module.Generator(client, "deepseek", MODEL, max_tokens=cap) for cap in CAPS}
    processor = data_processor.DataProcessor("finer")
    pairs = []
    with (
        patch.object(generator_module, "timed_llm_call", new=ace_timed_llm_call),
        patch.object(data_processor, "eval", new=ast.literal_eval, create=True),
        (output_dir / "progress.jsonl").open("w", encoding="utf-8") as progress,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        jobs = [pool.submit(replay_pair, generators, processor, case, repeat,
                            CAPS if (index + repeat) % 2 else CAPS[::-1], run_id)
                for index, case in enumerate(cases) for repeat in range(1, REPEATS + 1)]
        for future in as_completed(jobs):
            pair = future.result()
            pairs.append(pair)
            append_json(progress, pair)
            if len(pairs) % 5 == 0 or len(pairs) == len(jobs):
                print(f"Diagnostic pairs {len(pairs)}/{len(jobs)}", flush=True)

    pairs.sort(key=lambda pair: (pair["case_id"], pair["repeat"]))
    with (output_dir / "paired_rows.jsonl").open("w", encoding="utf-8") as handle:
        for pair in pairs:
            append_json(handle, pair)
    usage = usage_by_phase(ledger, run_id)
    scores = summarize_pairs(pairs)
    summary = {
        "run_id": run_id, "post_hoc_selected_on_truncation": True,
        "status": "incomplete" if any(cap["blocked_batches"] for cap in scores["by_cap"].values()) else "complete",
        **scores,
        "by_case": {case["case_id"]: summarize_pairs([pair for pair in pairs if pair["case_id"] == case["case_id"]])
                    for case in cases},
        "usage_by_trial_phase": usage,
        "committed_cny_by_cap": {str(cap): sum(group["settled_peak_cost_nano_cny"] + group["unresolved_reserved_nano_cny"]
                                              for group in usage if group["phase"] == f"cap_{cap}") / NANO_PER_CNY for cap in CAPS},
        "shared_budget_before": before, "shared_budget_after": budget_snapshot(client),
        "wall_seconds": time.monotonic() - started,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Send paid requests under the existing ledger")
    mode.add_argument("--dry-run", action="store_true", help="Verify archives and print the budget plan (default)")
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--rescue-dir", required=True, type=Path)
    parser.add_argument("--formal-dir", type=Path, default=ROOT / "evidence" / "formal")
    parser.add_argument("--output-dir", "--out", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.run and (args.output_dir is None or args.output_dir.exists()):
        parser.error("--run requires a new --output-dir")
    if args.run and args.ledger.resolve().is_relative_to(args.output_dir.resolve()):
        parser.error("--ledger must be outside the output directory")
    cases = load_cases(args.formal_dir, args.rescue_dir, args.ledger)
    if not args.run:
        print(json.dumps(diagnostic_plan(cases), indent=2))
        return 0
    if args.env_file is not None:
        from dotenv import load_dotenv

        load_dotenv(args.env_file, override=False)
    client = BudgetedDeepSeekClient(args.ledger, budget_cny=BUDGET_CNY, max_attempts=MAX_ATTEMPTS,
                                  max_input_tokens=INPUT_CAP, max_output_tokens=RESCUE_CAP)
    try:
        summary = run_diagnostic(client, cases, args.output_dir, args.ledger, args.workers)
    finally:
        client.close()
    print(json.dumps({key: summary[key] for key in ("status", "by_cap", "expanded_wins", "ties", "old_cap_wins",
                                                   "committed_cny_by_cap")}, indent=2))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
