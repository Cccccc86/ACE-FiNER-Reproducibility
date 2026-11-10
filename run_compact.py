"""Select and test a compact inference-only FiNER output contract."""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from hashlib import sha256
import json
from pathlib import Path
import re
from runpy import run_path
import statistics
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from finer_api import (
    BASE_URL, MISS_PRICE, NANO_PER_CNY, OUTPUT_PRICE, BudgetedDeepSeekClient,
    BudgetExceeded, ace_timed_llm_call, request_scope,
)
from run_finer import LABELS_PER_BATCH, MODEL, UPSTREAM, append_json, hash_text, usage_by_phase, write_json
from run_rescue import BUDGET_CNY, INITIAL_CAP, INPUT_CAP, MAX_ATTEMPTS, RESCUE_CAP, ROOT, evaluate_attempt, load_playbooks


DEV_SEED = 29
DEV_BATCHES = 8
FROZEN_TEST_BATCHES = 32
TEST_BATCHES = 31  # The final frozen row is reserved to keep the all-retry plan within the shared cap.
VERBOSE = "verbose"
CANDIDATES = ("compact_rationale", "answer_only")
ARMS = (VERBOSE, *CANDIDATES)
SAVINGS_THRESHOLD = 0.25
TAG = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_ROUTE = ContextVar("compact_prompt_route")


def prompt_templates():
    original = run_path(str(ROOT / "ace" / "prompts" / "generator.py"))["GENERATOR_PROMPT"]
    prefix, tail = original.split("**Playbook:**", 1)
    protected, _ = tail.split("**Answer in this exact JSON format:**", 1)
    instructions, _ = prefix.split("Your output should be a json object", 1)
    for line in ("- Show your reasoning step-by-step\n", "- Be concise but thorough in your analysis\n"):
        if instructions.count(line) != 1:
            raise ValueError("Pinned ACE output instruction changed")
        instructions = instructions.replace(line, "")
    final_answer_rule = "The final_answer must be one string of exactly four comma-separated tags in question order."
    rationale = (
        "Return one JSON object with reasoning, bullet_ids, and final_answer. "
        "Use one concise rationale of at most 40 words for reasoning. Include relevant playbook IDs in bullet_ids. "
        + final_answer_rule
    )
    only = "Return one JSON object with final_answer as its only field. " + final_answer_rule
    protected = "\n\n\n**Playbook:**" + protected + "**Answer in this exact JSON format:**\n"
    return {
        VERBOSE: original,
        "compact_rationale": instructions + rationale + protected
        + '{{\n  "reasoning": "[Concise rationale]",\n  "bullet_ids": ["calc-00001"],\n'
          '  "final_answer": "Tag1,Tag2,Tag3,Tag4"\n}}\n\n---\n',
        "answer_only": instructions + only + protected
        + '{{\n  "final_answer": "Tag1,Tag2,Tag3,Tag4"\n}}\n\n---\n',
    }


def render(template, row, playbook):
    return template.format(playbook, "(empty)", row["question"], row["context"])


def capacity_plan(splits, playbooks, templates):
    def bound(arm, rows, books):
        inputs = [544 + len(render(templates[arm], row, book).encode("utf-8")) for book in books for row in rows]
        # Each policy can call at both caps, with up to two transport attempts per cap.
        return {
            "policies": len(inputs), "maximum_input_bound": max(inputs),
            "worst_nano_cny": sum(2 * (2 * count * MISS_PRICE + (INITIAL_CAP + RESCUE_CAP) * OUTPUT_PRICE)
                                 for count in inputs),
        }

    dev = {arm: bound(arm, splits["dev"], [playbooks[DEV_SEED]]) for arm in ARMS}
    test = {arm: bound(arm, splits["test"], list(playbooks.values())) for arm in ARMS}
    dev_policies = sum(plan["policies"] for plan in dev.values())
    test_policies = test[VERBOSE]["policies"] + test[CANDIDATES[0]]["policies"]
    worst = sum(plan["worst_nano_cny"] for plan in dev.values()) + test[VERBOSE]["worst_nano_cny"]
    worst += max(test[arm]["worst_nano_cny"] for arm in CANDIDATES)
    return {
        "dev_policies": dev_policies, "test_policies": test_policies,
        "maximum_logical_calls": 2 * (dev_policies + test_policies),
        "maximum_api_attempts": 4 * (dev_policies + test_policies),
        "maximum_input_bound": max(plan["maximum_input_bound"] for plan in (*dev.values(), *test.values())),
        "maximum_incremental_reservation_nano_cny": worst,
        "maximum_incremental_reservation_cny": worst / NANO_PER_CNY,
        "dev_by_arm": dev, "test_possible_by_arm": test,
    }


def output_shape(response, arm):
    try:
        value = json.loads(response)
    except json.JSONDecodeError:
        value = None
    result = {"valid_json_object": isinstance(value, dict), "valid_four_tag_output": False,
              "follows_arm_contract": False, "rationale_words": None}
    if isinstance(value, dict):
        answer = value.get("final_answer")
        tags = answer.split(",") if isinstance(answer, str) else []
        result["valid_four_tag_output"] = len(tags) == LABELS_PER_BATCH and all(TAG.fullmatch(tag.strip()) for tag in tags)
        reasoning = value.get("reasoning")
        words = len(reasoning.split()) if isinstance(reasoning, str) else None
        result["rationale_words"] = words
        if arm == "answer_only":
            compliant = set(value) == {"final_answer"}
        else:
            compliant = set(value) == {"reasoning", "bullet_ids", "final_answer"} and isinstance(value["bullet_ids"], list)
            compliant = compliant and words is not None and (arm == VERBOSE or 0 < words <= 40)
        result["follows_arm_contract"] = result["valid_four_tag_output"] and compliant
    return result


def routed_call(client, provider, model, prompt, **kwargs):
    expected, replacement, arm, observation = _ROUTE.get()
    if prompt != expected:
        raise ValueError("Generator did not render the verified original FiNER prompt")
    response, info = ace_timed_llm_call(client, provider, model, replacement, **kwargs)
    observation.update(output_shape(response, arm))
    return response, info


def evaluate_policy(generators, processor, row, playbook, seed, split, arm, templates, run_id):
    original = render(templates[VERBOSE], row, playbook)
    replacement = render(templates[arm], row, playbook)

    def attempt(stage):
        observation = {}
        token = _ROUTE.set((original, replacement, arm, observation))
        try:
            with request_scope(condition=arm, split=split):
                result = evaluate_attempt(generators[stage], processor, row, playbook, seed,
                                          f"{split}.{arm}.{stage}", run_id)
        finally:
            _ROUTE.reset(token)
        if observation:
            result["output_shape"] = observation
        return result

    initial = attempt("initial")
    retry = attempt("rescue") if initial["status"] == "truncated" else None
    final = retry if retry is not None else initial
    return {"initial": initial, "rescue": retry, "final_status": final["status"],
            "correct_labels": final["correct_labels"], "output_shape": final.get("output_shape")}


def evaluate_row(generators, processor, row, playbook, seed, split, order, templates, run_id):
    return {
        "row_id": row["row_id"], "trial": seed, "split": split, "target": row["target"], "arm_order": order,
        "arms": {arm: evaluate_policy(generators, processor, row, playbook, seed, split, arm, templates, run_id)
                 for arm in order},
    }


def collect_phase(pool, generators, processor, rows, playbooks, split, arms, templates, run_id, output_dir):
    rotations = [arms[index:] + arms[:index] for index in range(len(arms))] if len(arms) == 3 else [arms]
    orders = [order for rotation in rotations for order in (rotation, rotation[::-1])]
    jobs = {}
    for seed_index, (seed, playbook) in enumerate(playbooks.items()):
        for index, row in enumerate(rows):
            order = orders[(index + seed_index) % len(orders)]
            future = pool.submit(evaluate_row, generators, processor, row, playbook, seed, split, order, templates, run_id)
            jobs[future] = (seed_index, index)
    records = {}
    with (output_dir / f"{split}_progress.jsonl").open("w", encoding="utf-8") as progress:
        for future in as_completed(jobs):
            record = future.result()
            records[jobs[future]] = record
            append_json(progress, record)
            if len(records) % 8 == 0 or len(records) == len(jobs):
                print(f"{split} paired rows {len(records)}/{len(jobs)}", flush=True)
    ordered = [records[key] for key in sorted(records)]
    with (output_dir / f"{split}_rows.jsonl").open("w", encoding="utf-8") as handle:
        for record in ordered:
            append_json(handle, record)
    return ordered


def arm_stats(records, usage, arm):
    policies = [row["arms"][arm] for row in records]
    count = sum(policy["correct_labels"] for policy in policies)
    denominator = len(policies) * LABELS_PER_BATCH
    groups = [group for group in usage if group["phase"].split(".")[1] == arm]
    keys = ("attempts", "settled_attempts", "prompt_tokens", "completion_tokens", "observed_cache_hit_tokens",
            "unknown_cache_breakdown_attempts", "settled_peak_cost_nano_cny", "unresolved_reserved_nano_cny")
    totals = {key: sum(group[key] for group in groups) for key in keys}
    totals["unresolved_attempts"] = totals["attempts"] - totals["settled_attempts"]
    totals["committed_cny"] = (totals["settled_peak_cost_nano_cny"] + totals["unresolved_reserved_nano_cny"]) / NANO_PER_CNY
    totals["uncached_equivalent_cny_for_settled"] = (
        totals["prompt_tokens"] * MISS_PRICE + totals["completion_tokens"] * OUTPUT_PRICE
    ) / NANO_PER_CNY
    return {
        "correct_labels": count, "total_labels": denominator, "accuracy": count / denominator,
        "incomplete_batches": sum(policy["final_status"] != "scored" for policy in policies),
        "malformed_four_tag_batches": sum(policy["output_shape"] is not None
                                           and not policy["output_shape"]["valid_four_tag_output"] for policy in policies),
        "contract_compliant_batches": sum(bool(policy["output_shape"] and policy["output_shape"]["follows_arm_contract"])
                                          for policy in policies),
        "initial_truncations": sum(policy["initial"]["status"] == "truncated" for policy in policies),
        "final_truncations": sum(policy["final_status"] == "truncated" for policy in policies),
        "rescue_calls": sum(policy["rescue"] is not None for policy in policies),
        "blocked_batches": sum(policy["final_status"] == "blocked" for policy in policies),
        "logical_latency_seconds": sum(policy["initial"]["elapsed_seconds"] +
                                       (policy["rescue"]["elapsed_seconds"] if policy["rescue"] else 0)
                                       for policy in policies),
        "usage_including_retries": totals,
    }


def adoption_gate(control, candidate):
    old = control["usage_including_retries"]
    new = candidate["usage_including_retries"]
    known = not (old["unresolved_attempts"] or new["unresolved_attempts"])
    savings = 1 - new["completion_tokens"] / old["completion_tokens"] if known and old["completion_tokens"] else None
    checks = {
        "no_label_loss": candidate["correct_labels"] >= control["correct_labels"],
        "no_extra_incomplete": candidate["incomplete_batches"] <= control["incomplete_batches"],
        "no_extra_malformed": candidate["malformed_four_tag_batches"] <= control["malformed_four_tag_batches"],
        "no_blocked": not (control["blocked_batches"] or candidate["blocked_batches"]),
        "output_tokens_saved_at_least_25_percent": (savings is not None
                                                   and 4 * new["completion_tokens"] <= 3 * old["completion_tokens"]),
    }
    return {"passed": all(checks.values()), "checks": checks, "completion_token_savings_fraction": savings,
            "completion_token_comparison_complete": known}


def choose_candidate(dev):
    checks = {arm: adoption_gate(dev[VERBOSE], dev[arm]) for arm in CANDIDATES}
    eligible = [arm for arm in CANDIDATES if checks[arm]["passed"]]

    def rank(arm):
        result = dev[arm]
        usage = result["usage_including_retries"]
        tie_break = usage["completion_tokens"] if eligible else usage["committed_cny"]
        return -result["correct_labels"], tie_break, arm

    return {"selected": min(eligible or CANDIDATES, key=rank), "dev_gate_passed": bool(eligible),
            "exploratory_only": not eligible, "candidate_gates": checks}


def summarize_test(records, usage, candidate):
    def paired(rows):
        differences = [row["arms"][candidate]["correct_labels"] - row["arms"][VERBOSE]["correct_labels"] for row in rows]
        return {"gain_pp": 100 * sum(differences) / (len(rows) * LABELS_PER_BATCH),
                "wins": sum(delta > 0 for delta in differences), "ties": differences.count(0),
                "losses": sum(delta < 0 for delta in differences)}

    arms = (VERBOSE, candidate)
    overall = {arm: arm_stats(records, usage, arm) for arm in arms}
    trials = []
    for seed in dict.fromkeys(row["trial"] for row in records):
        rows = [row for row in records if row["trial"] == seed]
        trial_usage = [group for group in usage if group["trial"] == seed]
        trials.append({"trial": seed, "arms": {arm: arm_stats(rows, trial_usage, arm) for arm in arms}, "paired": paired(rows)})

    def spread(values):
        return {"values": values, "mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values) if len(values) > 1 else None}

    return {"candidate": candidate, "arms": overall, "trials": trials, "paired": paired(records),
            "paired_gain_pp_by_trial": spread([trial["paired"]["gain_pp"] for trial in trials]),
            "accuracy_by_trial": {arm: spread([trial["arms"][arm]["accuracy"] for trial in trials]) for arm in arms},
            "gate": adoption_gate(overall[VERBOSE], overall[candidate]), "usage_by_trial_phase": usage}


def budget_snapshot(client):
    return {key: value for key, value in client.summary().items() if key != "ledger_path"}


def verify_capacity(plan, before):
    if (plan["maximum_input_bound"] > INPUT_CAP or before["usage_bound_violations"]
            or plan["maximum_incremental_reservation_nano_cny"] > round(before["remaining_cny"] * NANO_PER_CNY)
            or before["attempts"] + plan["maximum_api_attempts"] > before["max_attempts"]):
        raise BudgetExceeded("Shared ledger cannot cover both phases and every allowed retry")


def run_experiment(client, splits, playbooks, templates, output_dir, ledger, workers, data_hash=None, reserved_test_ids=()):
    from ace.core import generator as generator_module
    from eval.finance import data_processor

    started = time.monotonic()
    plan = capacity_plan(splits, playbooks, templates)
    before = budget_snapshot(client)
    verify_capacity(plan, before)
    output_dir.mkdir(parents=True, exist_ok=False)
    run_id = uuid4().hex
    write_json(output_dir / "run_manifest.json", {
        "run_id": run_id, "upstream_commit": UPSTREAM, "model": MODEL, "base_url": BASE_URL,
        "temperature": 0, "thinking": "disabled", "json_mode": True, "workers": workers,
        "compact_data_manifest_sha256": data_hash,
        "split_ids": {split: [row["row_id"] for row in rows] for split, rows in splits.items()},
        "frozen_but_unused_test_ids": reserved_test_ids,
        "checkpoint_sha256": {seed: hash_text(book) for seed, book in playbooks.items()},
        "prompt_template_sha256": {arm: hash_text(template) for arm, template in templates.items()},
        "initial_cap": INITIAL_CAP, "rescue_cap": RESCUE_CAP, "rescue_trigger": "finish_reason=length",
        "maximum_transport_attempts_per_logical_call": 2, "dev_checkpoint": DEV_SEED,
        "dev_arms": ARMS, "test_arms": [VERBOSE, "selected_on_dev"],
        "output_savings_threshold": SAVINGS_THRESHOLD, "capacity_plan": plan, "shared_budget_before": before,
    })
    generators = {stage: generator_module.Generator(client, "deepseek", MODEL, max_tokens=cap)
                  for stage, cap in (("initial", INITIAL_CAP), ("rescue", RESCUE_CAP))}
    processor = data_processor.DataProcessor("finer")
    with (
        patch.object(generator_module, "timed_llm_call", new=routed_call),
        patch.object(data_processor, "eval", new=ast.literal_eval, create=True),
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        dev_rows = collect_phase(pool, generators, processor, splits["dev"], {DEV_SEED: playbooks[DEV_SEED]},
                                 "dev", ARMS, templates, run_id, output_dir)
        dev_usage = usage_by_phase(ledger, run_id)
        dev_stats = {arm: arm_stats(dev_rows, dev_usage, arm) for arm in ARMS}
        selection = choose_candidate(dev_stats)
        dev = {"arms": dev_stats, "selection": selection, "usage_by_trial_phase": dev_usage}
        write_json(output_dir / "dev_summary.json", dev)
        candidate = selection["selected"]
        print(f"Dev selected {candidate}; gate passed: {selection['dev_gate_passed']}", flush=True)
        test = None
        if not any(result["blocked_batches"] for result in dev_stats.values()):
            test_rows = collect_phase(pool, generators, processor, splits["test"], playbooks,
                                      "test", (VERBOSE, candidate), templates, run_id, output_dir)
            test_usage = [group for group in usage_by_phase(ledger, run_id) if group["phase"].startswith("test.")]
            test = summarize_test(test_rows, test_usage, candidate)
            write_json(output_dir / "test_summary.json", test)

    complete = test is not None and not any(result["blocked_batches"] for result in test["arms"].values())
    adopt = complete and selection["dev_gate_passed"] and test["gate"]["passed"]
    summary = {
        "run_id": run_id, "status": "complete" if complete else "incomplete",
        "selected_candidate": candidate, "recommended_policy": candidate if adopt else VERBOSE,
        "adopt_candidate": adopt, "dev": dev, "test": test,
        "shared_budget_before": before, "shared_budget_after": budget_snapshot(client),
        "wall_seconds": time.monotonic() - started,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def smoke_test():
    import tiktoken

    templates = prompt_templates()
    captured = []

    def create(**params):
        prompt = params["messages"][0]["content"]
        captured.append((prompt, params["max_tokens"]))
        compact = "at most 40 words" in prompt
        only = "as its only field" in prompt
        truncate = compact and "<test-retry>" in prompt and params["max_tokens"] == INITIAL_CAP
        answer = "Alpha,Beta,Wrong,Delta" if only else "Alpha,Beta,Gamma,Delta"
        fields = {"final_answer": answer}
        if not only:
            fields.update(reasoning="Concise rationale.", bullet_ids=[])
        return {"choices": [{"finish_reason": "length" if truncate else "stop",
                             "message": {"content": "{" if truncate else json.dumps(fields)}}],
                "usage": {"prompt_tokens": 200, "completion_tokens": 12 if compact else 100}}

    def row(name):
        return {"row_id": name, "context": "<protected-context>", "question": f"<{name}>",
                "target": "Alpha,Beta,Gamma,Delta"}

    splits = {"dev": [row("dev")], "test": [row("test-retry"), row("test-normal")]}
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        root = Path(directory)
        ledger = root / "ledger.jsonl"
        fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        client = BudgetedDeepSeekClient(ledger, max_input_tokens=INPUT_CAP, max_output_tokens=RESCUE_CAP, client=fake)
        with patch.object(tiktoken, "get_encoding", new=lambda name: None):
            result = run_experiment(client, splits, {DEV_SEED: "<protected-playbook>"}, templates, root / "out", ledger, 2)
        assert result["selected_candidate"] == "compact_rationale" and result["adopt_candidate"]
        assert result["test"]["arms"]["compact_rationale"]["rescue_calls"] == 1
        assert result["shared_budget_after"]["attempts"] == 8
        assert all("<protected-context>" in prompt and "<protected-playbook>" in prompt for prompt, _ in captured)
        retried = [(prompt, cap) for prompt, cap in captured if "<test-retry>" in prompt and "at most 40 words" in prompt]
        assert retried[0][0] == retried[1][0] and [cap for _, cap in retried] == [INITIAL_CAP, RESCUE_CAP]
        assert choose_candidate({arm: result["dev"]["arms"][VERBOSE] for arm in ARMS})["exploratory_only"]
    print("Offline compact output-contract smoke test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Send paid requests under the existing ledger")
    mode.add_argument("--dry-run", action="store_true", help="Verify frozen inputs and print the budget plan (default)")
    mode.add_argument("--smoke-test", action="store_true", help="Exercise both phases with an offline provider")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--out", "--output-dir", type=Path)
    parser.add_argument("--checkpoints-dir", type=Path, default=ROOT / "evidence" / "formal")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.smoke_test:
        smoke_test()
        return 0
    if args.run:
        if args.out is None or args.out.exists() or args.ledger is None or not args.ledger.is_file():
            parser.error("--run requires a new --out directory and an existing --ledger")
        if args.ledger.resolve().is_relative_to(args.out.resolve()):
            parser.error("--ledger must be outside the output directory")
    from compact_data import load_compact_splits

    splits = load_compact_splits()
    if {split: len(rows) for split, rows in splits.items()} != {"dev": DEV_BATCHES, "test": FROZEN_TEST_BATCHES}:
        raise ValueError("Frozen dev and test counts differ from the protocol")
    reserved_test_ids = [row["row_id"] for row in splits["test"][TEST_BATCHES:]]
    splits["test"] = splits["test"][:TEST_BATCHES]
    playbooks = load_playbooks(args.checkpoints_dir)
    templates = prompt_templates()
    if not args.run:
        print(json.dumps({"split_counts": {split: len(rows) for split, rows in splits.items()},
                          "frozen_but_unused_test_ids": reserved_test_ids,
                          **capacity_plan(splits, playbooks, templates)}, indent=2))
        return 0
    if args.env_file is not None:
        from dotenv import load_dotenv

        load_dotenv(args.env_file, override=False)
    client = BudgetedDeepSeekClient(args.ledger, budget_cny=BUDGET_CNY, max_attempts=MAX_ATTEMPTS,
                                  max_input_tokens=INPUT_CAP, max_output_tokens=RESCUE_CAP)
    try:
        digest = sha256((ROOT / "compact_manifest.json").read_bytes()).hexdigest()
        summary = run_experiment(client, splits, playbooks, templates, args.out, args.ledger, args.workers,
                                 digest, reserved_test_ids)
    finally:
        client.close()
    print(json.dumps({key: summary[key] for key in ("status", "selected_candidate", "recommended_policy", "adopt_candidate")}, indent=2))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
