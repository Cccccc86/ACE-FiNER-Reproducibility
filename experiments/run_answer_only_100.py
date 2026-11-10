"""Compare original and label-only ACE output on an unused FiNER validation holdout."""

import argparse
import ast
import json
import os
import random
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from retest_finer import RUNS, WORK, selection
from finer_api import BASE_URL, MISS_PRICE, MODEL, NANO_PER_CNY, OUTPUT_PRICE, BudgetedDeepSeekClient
from finer_data import _sha256
from run_compact import VERBOSE, _ROUTE, arm_stats, prompt_templates, render, routed_call
from run_finer import LABELS_PER_BATCH, append_json, usage_by_phase, write_json
from run_rescue import INITIAL_CAP, evaluate_attempt


ARMS = (VERBOSE, "answer_only")
SEEDS = (17, 29, 43)
BATCHES = 64
PRIOR_RUNS = ("fresh_retest", "evidence_first", "training_dose", "training_scale", "confirm_100")
RANK_PREFIX = "ACE-FINER-ANSWER-ONLY-100-VAL-V1|"
MARGIN_PP = 2.0
MIN_TOKEN_SAVINGS = 0.25
BOOTSTRAP_SEED = 20261111
WORKERS = 12
OUTPUT_DIR = RUNS / "answer_only_100"


def prepare():
    rows, manifest = selection(
        batch_count=BATCHES, rank_prefix=RANK_PREFIX, holdout_split="val",
        extra_prior_runs=tuple(WORK / name for name in PRIOR_RUNS),
    )
    scale = WORK / "training_scale"
    confirm = json.loads((WORK / "confirm_100" / "run_manifest.json").read_bytes())
    templates = {arm: prompt_templates()[arm] for arm in ARMS}
    settings = {
        "model": MODEL, "base_url": BASE_URL, "temperature": 0, "thinking": "disabled",
        "json_mode": True, "max_output_tokens": INITIAL_CAP, "length_retry": False,
        "maximum_transport_attempts": 2,
    }
    if (any(confirm[key] != value for key, value in settings.items())
            or _sha256(templates[VERBOSE]) != confirm["generator_prompt_sha256"]):
        raise ValueError("The control must match the 100-batch confirmation run")

    books = {}
    for seed in SEEDS:
        checkpoint = json.loads((scale / f"trial_{seed}" / "checkpoint_100.json").read_bytes())
        book = checkpoint["playbook"]
        if (checkpoint["seed"] != seed or checkpoint["completed_total"] != 100
                or _sha256(book) != confirm["playbook_sha256"][str(seed)]):
            raise ValueError(f"The 100-batch checkpoint for seed {seed} differs from confirmation")
        books[seed] = book

    for marker in ("**Playbook:**", "**Answer in this exact JSON format:**"):
        if any(template.count(marker) != 1 for template in templates.values()):
            raise ValueError(f"Unexpected template structure at {marker}")
    protected = {
        template.split("**Playbook:**", 1)[1].split("**Answer in this exact JSON format:**", 1)[0]
        for template in templates.values()
    }
    if len(protected) != 1:
        raise ValueError("The playbook, reflection, question and context must match across arms")

    bounds = [544 + len(render(templates[arm], row, book).encode("utf-8"))
              for row in rows for book in books.values() for arm in ARMS]
    budget_nano = 50 * NANO_PER_CNY - manifest["prior_committed_nano_cny"]
    if max(bounds) > confirm["max_input_tokens"] or budget_nano <= 0:
        raise ValueError("Input bound or remaining budget does not permit the comparison")
    manifest.update({
        **settings, "holdout_split": "val", "unique_holdout_batches": len(rows),
        "selection_rule": "SHA-256 rank among unused released validation rows. Exclude any normalized "
                          "sentence in released train/test, previously selected or requested validation rows, "
                          "or earlier rows in this holdout.",
        "max_input_tokens": confirm["max_input_tokens"], "maximum_request_byte_bound": max(bounds),
        "playbook_sha256": {seed: _sha256(book) for seed, book in books.items()},
        "template_sha256": {arm: _sha256(template) for arm, template in templates.items()},
        "planned_completions": len(bounds), "max_new_attempts": 2 * len(bounds),
        "available_cny": budget_nano / NANO_PER_CNY,
        "maximum_uncached_byte_bound_cny": (
            2 * sum(bound * MISS_PRICE + INITIAL_CAP * OUTPUT_PRICE for bound in bounds) / NANO_PER_CNY
        ),
        "budget_rule": "Reserve each request under the remaining total budget. A blocked run is incomplete.",
        "order": "Alternate which arm is first by holdout index and checkpoint index.",
        "accuracy_margin_pp": MARGIN_PP, "minimum_completion_token_savings_fraction": MIN_TOKEN_SAVINGS,
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_draws": 10000,
        "accuracy_interval_unit": "Unique holdout row, with all checkpoint results kept together.",
        "scoring": "Original ACE FiNER positional scorer. Check JSON and tag count separately.",
    })
    return rows, books, templates, manifest, budget_nano


def evaluate_pair(generator, processor, row, seed, book, order, templates, run_id):
    pair = {"row_id": row["row_id"], "trial": seed, "target": row["target"], "arm_order": order, "arms": {}}
    original = render(templates[VERBOSE], row, book)
    for arm in order:
        shape = {}
        route = _ROUTE.set((original, render(templates[arm], row, book), arm, shape))
        try:
            attempt = evaluate_attempt(generator, processor, row, book, seed, f"holdout.{arm}.initial", run_id)
        finally:
            _ROUTE.reset(route)
        pair["arms"][arm] = {
            "initial": attempt, "rescue": None, "final_status": attempt["status"],
            "correct_labels": attempt["correct_labels"], "output_shape": shape or None,
        }
    return pair


def summarize(pairs, ledger, run_id, manifest, usage):
    grouped = {}
    transitions = Counter()
    for pair in pairs:
        arms = pair["arms"]
        expected = [tag.strip().lower() for tag in pair["target"].split(",")]
        position_flags = []
        for arm in ARMS:
            prediction = arms[arm]["initial"]["prediction"] or ""
            tags = [tag.strip().lower() for tag in prediction.split(",")] + [""] * LABELS_PER_BATCH
            flags = [int(tag == gold) for tag, gold in zip(tags, expected)]
            if sum(flags) != arms[arm]["correct_labels"]:
                raise ValueError("Position-level audit disagrees with the upstream scorer")
            position_flags.append(flags)
        transitions.update(f"{old}{new}" for old, new in zip(*position_flags))
        grouped.setdefault(pair["row_id"], {})[pair["trial"]] = (
            arms[ARMS[1]]["correct_labels"] - arms[VERBOSE]["correct_labels"]
        )
    expected_ids = {record["row_id"] for record in manifest["selection"]}
    if (set(grouped) != expected_ids or len(pairs) != len(expected_ids) * len(SEEDS)
            or any(set(deltas) != set(SEEDS) for deltas in grouped.values())):
        raise ValueError("The paired evidence does not cover the frozen holdout and checkpoints")

    per_row_pp = [100 * statistics.mean(deltas.values()) / LABELS_PER_BATCH for deltas in grouped.values()]
    rng = random.Random(BOOTSTRAP_SEED)
    draws = sorted(statistics.mean(rng.choices(per_row_pp, k=len(per_row_pp))) for _ in range(10000))
    usage_groups = usage_by_phase(ledger, run_id)
    stats = {arm: arm_stats(pairs, usage_groups, arm) for arm in ARMS}
    control, treatment = (stats[arm] for arm in ARMS)
    old, new = (values["usage_including_retries"] for values in (control, treatment))
    known_usage = not (old["unresolved_attempts"] or new["unresolved_attempts"])
    savings = 1 - new["completion_tokens"] / old["completion_tokens"] if known_usage and old["completion_tokens"] else None
    status = {
        arm: dict(Counter(p["arms"][arm]["final_status"] for p in pairs)) for arm in ARMS
    }
    complete = all(set(counts) <= {"scored", "truncated", "invalid_answer"} for counts in status.values())
    checks = {
        "accuracy_noninferiority_at_margin": draws[500] > -MARGIN_PP,
        "output_token_savings": savings is not None and savings >= MIN_TOKEN_SAVINGS,
        "no_extra_incomplete": treatment["incomplete_batches"] <= control["incomplete_batches"],
        "no_extra_malformed": treatment["malformed_four_tag_batches"] <= control["malformed_four_tag_batches"],
        "no_extra_truncations": treatment["final_truncations"] <= control["final_truncations"],
    }
    events = [json.loads(line) for line in ledger.read_bytes().splitlines()]
    fingerprints = sorted({e["system_fingerprint"] for e in events
                           if e["event"] == "settle" and e.get("system_fingerprint")})
    return {
        "status": "complete" if complete else "incomplete", "arms": stats, "status_by_arm": status,
        "checkpoints": {
            seed: {arm: arm_stats([p for p in pairs if p["trial"] == seed],
                                 [g for g in usage_groups if g["trial"] == seed], arm)
                   for arm in ARMS} for seed in SEEDS
        },
        "paired_accuracy_change_pp": statistics.mean(per_row_pp),
        "paired_batch_bootstrap_95pct_interval_pp": [draws[250], draws[9749]],
        "paired_batch_bootstrap_one_sided_95pct_lower_pp": draws[500],
        "label_correctness_transitions_verbose_to_answer_only": dict(sorted(transitions.items())),
        "completion_token_savings_fraction": savings, "decision_checks": checks,
        "passes_predeclared_gate": complete and all(checks.values()),
        "observed_model_fingerprints": fingerprints, "usage": usage,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run and args.out.resolve() != OUTPUT_DIR:
        parser.error("Live inference must use the fixed output directory so this run cannot be repeated under the same budget.")
    if args.run and not os.environ.get("DEEPSEEK_API_KEY"):
        parser.error("Set DEEPSEEK_API_KEY in the local environment before running live inference.")
    rows, books, templates, manifest, budget_nano = prepare()
    print(json.dumps({key: manifest[key] for key in (
        "unique_holdout_batches", "eligible_batches", "planned_completions", "available_cny",
        "maximum_request_byte_bound",
    )}), flush=True)
    if not args.run:
        return

    from ace.core import generator as generator_module
    from eval.finance import data_processor

    args.out.mkdir(parents=True, exist_ok=False)
    run_id = uuid4().hex
    write_json(args.out / "run_manifest.json", {"run_id": run_id, **manifest})
    ledger = args.out / "usage.jsonl"
    client = BudgetedDeepSeekClient(
        ledger, budget_cny=Decimal(budget_nano) / NANO_PER_CNY, max_attempts=manifest["max_new_attempts"],
        max_input_tokens=manifest["max_input_tokens"], max_output_tokens=INITIAL_CAP, timeout_s=45,
    )
    generator = generator_module.Generator(client, "deepseek", MODEL, max_tokens=INITIAL_CAP)
    processor = data_processor.DataProcessor("finer")
    jobs = [(row, seed, book, ARMS if (index + book_index) % 2 == 0 else ARMS[::-1])
            for index, row in enumerate(rows) for book_index, (seed, book) in enumerate(books.items())]

    def run_pair(job):
        return evaluate_pair(generator, processor, *job, templates, run_id)

    try:
        with (patch.object(generator_module, "timed_llm_call", routed_call),
              patch.object(data_processor, "eval", ast.literal_eval, create=True),
              (args.out / "paired_rows.jsonl").open("w", encoding="utf-8") as log,
              ThreadPoolExecutor(max_workers=WORKERS) as pool):
            first = run_pair(jobs[0])
            append_json(log, first)
            if any(record["final_status"] in {"error", "blocked"} for record in first["arms"].values()):
                raise RuntimeError("First paired request failed. See recorded provider status.")
            pairs = [first]
            for pair in pool.map(run_pair, jobs[1:]):
                append_json(log, pair)
                pairs.append(pair)
                if len(pairs) % 24 == 0:
                    print(f"Completed {len(pairs)}/{len(jobs)} paired evaluations", flush=True)
        usage = client.summary()
    finally:
        client.close()
    result = summarize(pairs, ledger, run_id, manifest, usage)
    write_json(args.out / "summary.json", result)
    print(json.dumps({key: result[key] for key in (
        "status", "paired_accuracy_change_pp", "paired_batch_bootstrap_95pct_interval_pp",
        "completion_token_savings_fraction", "passes_predeclared_gate",
    )}), flush=True)
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
