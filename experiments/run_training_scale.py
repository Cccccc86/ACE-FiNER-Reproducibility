"""Continue three FiNER playbooks to 100 batches and test fixed checkpoints."""

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
import statistics
import sys
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from run_training_dose import frozen_inputs, prior_budget, read_rows
from retest_finer import (
    BASE_URL, INITIAL_CAP, MODEL, NANO_PER_CNY, ROOT, UPSTREAM, WORK,
    BudgetedDeepSeekClient, _sha256, append_json, differences_interval, evaluation_row, selection, write_json,
)
from finer_api import ProviderRequestError, TruncatedCompletion, ace_timed_llm_call, request_scope
from rescue_data import _source_rows
from run_finer import new_ace

PRIOR_DOSE = WORK / "training_dose"
DOSES = (24, 48, 100)
ARMS = ("base", *(f"ace_{dose}" for dose in DOSES))
HOLDOUT_COUNT = 64
RANK_PREFIX = "ACE-FINER-TRAINING-SCALE-V1|"
INPUT_BOUND = 131072
STEP_ATTEMPTS = 3
RECOVERY_CAP = 3072
WORKERS = 12


def prepare():
    _, _, _, used_orders, prefix, original_test, _ = frozen_inputs()
    prior = json.loads((PRIOR_DOSE / "run_manifest.json").read_bytes())
    summary = json.loads((PRIOR_DOSE / "trained_checkpoints.json").read_bytes())
    used = prefix + used_orders[17]
    forbidden_rows = {row["row_id"] for row in used}
    original_manifest = json.loads((ROOT / "finer_manifest.json").read_bytes())
    forbidden_rows.update(r["row_id"] for r in original_manifest["selections"]["pilot"]["train"])
    sources = original_test["sources"]
    blocked_sentences = {
        s for split in ("val", "test") for rec, _ in _source_rows(split, sources[split])
        for s in rec["sentence_sha256"]
    }
    ranked = []
    for rec, row in _source_rows("train", sources["train"]):
        hashes = set(rec["sentence_sha256"])
        if rec["row_id"] in forbidden_rows:
            blocked_sentences.update(hashes)
        else:
            rank = _sha256(RANK_PREFIX + "train|" + rec["original_context_sha256"])
            ranked.append((rank, rec["physical_line_1based"], hashes, row))
    extra = []
    for _, _, hashes, row in sorted(ranked):
        if blocked_sentences.isdisjoint(hashes):
            blocked_sentences.update(hashes)
            extra.append(row)
        if len(extra) == DOSES[-1] - DOSES[0]:
            break
    if len(extra) != DOSES[-1] - DOSES[0]:
        raise ValueError("Insufficient clean training batches")

    rows, holdout = selection(
        batch_count=HOLDOUT_COUNT, rank_prefix=RANK_PREFIX,
        extra_prior_runs=(WORK / "fresh_retest", WORK / "evidence_first"),
    )
    checkpoint24, orders = {}, {}
    for seed in used_orders:
        path = PRIOR_DOSE / f"trial_{seed}"
        state = json.loads((path / "checkpoint.json").read_bytes())
        recorded = read_rows(path / "training_rows.jsonl")
        ids = [row["row_id"] for row in used_orders[seed]]
        if (state["seed"] != seed or state["last_step"] != DOSES[0] or state["completed_total"] != DOSES[0]
                or _sha256(state["playbook"]) != state["playbook_sha256"]
                or summary[str(seed)] != {"completed_additional": 12, "sha256": state["playbook_sha256"]}
                or state["playbook"] != (path / "playbook_24.txt").read_text(encoding="utf-8")
                or ids != prior["extension_order"][str(seed)]
                or ids != [row["row_id"] for row in recorded if row["status"] == "complete"]):
            raise ValueError(f"Trial {seed} lacks a verified 24-batch checkpoint")
        checkpoint24[seed] = state
        first = extra[:DOSES[1] - DOSES[0]]
        rest = extra[DOSES[1] - DOSES[0]:]
        rng = random.Random(seed)
        rng.shuffle(first)
        rng.shuffle(rest)
        orders[seed] = first + rest
    cost, fingerprints, ledgers = prior_budget((PRIOR_DOSE / "usage.jsonl",))
    return rows, holdout, checkpoint24, orders, extra, used, cost, fingerprints, ledgers, prior


def train(seed, original, rows, processor, directory, run_id):
    directory.mkdir(exist_ok=True)
    logs = directory / "role_logs"
    logs.mkdir(exist_ok=True)
    resume_path = directory / "state.json"
    state = json.loads(resume_path.read_bytes()) if resume_path.exists() else original
    start = state["last_step"]
    if (state["seed"] != seed or state["completed_total"] != start
            or not DOSES[0] <= start <= DOSES[-1] or _sha256(state["playbook"]) != state["playbook_sha256"]):
        raise ValueError(f"Invalid continuation checkpoint for {seed}")
    system = new_ace(INITIAL_CAP)
    system.playbook, system.next_global_id = state["playbook"], state["next_global_id"]
    config = system._extract_config_params({
        "task_name": "finer", "max_num_rounds": 3, "curator_frequency": 1,
        "json_mode": True, "no_ground_truth": False,
    })
    with (directory / "training_rows.jsonl").open("a", encoding="utf-8") as progress:
        for step in range(start + 1, DOSES[-1] + 1):
            row = rows[step - DOSES[0] - 1]
            before, next_id = system.playbook, system.next_global_id
            for attempt in range(1, STEP_ATTEMPTS + 1):
                try:
                    with request_scope(run_id=run_id, trial=seed, arm="training_scale", phase="train",
                                       row_id=row["row_id"], condition=f"step_{step}_attempt_{attempt}"):
                        pre, post, _ = system._train_single_sample(
                            task_dict={k: row[k] for k in ("row_id", "context", "question", "target")},
                            data_processor=processor, step_id=f"train_e_1_s_{step}",
                            epoch=1, step=step, usage_log_path=str(directory / "bullet_usage_log.jsonl"),
                            log_dir=str(logs), config_params=config, total_samples=DOSES[-1],
                        )
                    break
                except (TruncatedCompletion, ProviderRequestError) as exc:
                    system.playbook, system.next_global_id = before, next_id
                    append_json(progress, {"step": step, "row_id": row["row_id"], "attempt": attempt,
                                           "status": "retry_or_stop", "error_type": type(exc).__name__})
                    retryable = isinstance(exc, TruncatedCompletion) or exc.retryable
                    if attempt == STEP_ATTEMPTS or not retryable:
                        raise
            state = {
                "seed": seed, "last_step": step, "completed_total": step,
                "playbook": system.playbook, "next_global_id": system.next_global_id,
                "playbook_sha256": _sha256(system.playbook),
            }
            append_json(progress, {"step": step, "row_id": row["row_id"], "status": "complete",
                                   "attempt": attempt, "pre_prediction": pre, "post_prediction": post,
                                   "playbook_sha256": state["playbook_sha256"]})
            if step in DOSES:
                write_json(directory / f"checkpoint_{step}.json", state)
            write_json(resume_path, state)
            if step % 8 == 0 or step in DOSES:
                print(f"Trial {seed}: {step}/{DOSES[-1]} training batches", file=sys.stderr, flush=True)
    books = {24: original["playbook"]}
    for dose in DOSES[1:]:
        checkpoint = json.loads((directory / f"checkpoint_{dose}.json").read_bytes())
        book = checkpoint["playbook"]
        if checkpoint["last_step"] != dose or checkpoint["playbook_sha256"] != _sha256(book):
            raise ValueError(f"Checkpoint {seed}/{dose} failed verification")
        books[dose] = book
    return books


def metrics(pairs, intervals=False):
    if not pairs:
        return None
    counts = Counter()
    by_row = defaultdict(lambda: defaultdict(list))
    statuses = {arm: Counter() for arm in ARMS}
    for pair in pairs:
        target = [tag.strip().lower() for tag in pair["base"]["target"].split(",")]
        correct = {}
        for arm in ARMS:
            rec = pair[arm]
            pred = (rec["prediction"] or "").lower().split(",") + [""] * 4
            correct[arm] = [a.strip() == b for a, b in zip(pred, target)]
            if len(target) != 4 or sum(correct[arm]) != rec["correct_labels"]:
                raise ValueError("Independent scoring differs from recorded labels")
            statuses[arm][rec.get("error_type", rec["status"])] += 1
        for b, c24, c48, c100 in zip(*(correct[arm] for arm in ARMS)):
            flags = (b, c24, c48, c100)
            counts[flags] += 1
            for dose, value in ((48, c48), (100, c100)):
                by_row[pair["row_id"]][dose].append(value - c24)
    result = {"positions": 4 * len(pairs), "arms": {}}
    for index, arm in enumerate(ARMS):
        stats = {"correct": sum(n for flags, n in counts.items() if flags[index]),
                 "statuses": dict(statuses[arm])}
        if arm != "base":
            stats.update(
                base_right_to_wrong=sum(n for f, n in counts.items() if f[0] and not f[index]),
                base_wrong_to_right=sum(n for f, n in counts.items() if not f[0] and f[index]),
            )
        result["arms"][arm] = stats
    if intervals:
        result["bootstrap_95pct_difference_vs_24_pp"] = {
            dose: differences_interval([100 * statistics.mean(group[dose]) for group in by_row.values()])
            for dose in DOSES[1:]
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recover-seed", type=int, help="Resume only a seed stopped by training truncation")
    args = parser.parse_args()
    if args.recover_seed is not None and not (args.run and args.resume):
        parser.error("--recover-seed requires --run --resume")
    rows, holdout, originals, orders, extra, used, cost, fingerprints, ledgers, prior = prepare()
    remaining = 50 * NANO_PER_CNY - cost
    if remaining <= 0:
        raise ValueError("Prior calls used the authorized budget")
    max_attempts = 2 * (STEP_ATTEMPTS * 9 * len(extra) * len(orders) + len(rows) * len(orders) * len(ARMS))
    stages = {dose: len({t.strip() for r in used + extra[:dose-24] for t in r["target"].split(",")})
              for dose in DOSES}
    plan = {
        "upstream_commit": UPSTREAM, "model": MODEL, "base_url": BASE_URL,
        "temperature": 0, "thinking": "disabled", "json_mode": True,
        "output_cap": INITIAL_CAP, "administrative_input_bound": INPUT_BOUND, "reflection_rounds_cap": 3,
        "curator_continuation_total": DOSES[-1], "training_step_attempt_cap": STEP_ATTEMPTS,
        "checkpoints": list(DOSES), "checkpoint_24_sha256": {str(s): c["playbook_sha256"] for s,c in originals.items()},
        "previous_dose_manifest_sha256": _sha256((PRIOR_DOSE / "run_manifest.json").read_bytes()),
        "generator_prompt_sha256": prior["prompt_sha256"], "distinct_training_labels": {str(k): v for k, v in stages.items()},
        "extension_order": {str(seed): [row["row_id"] for row in train_rows] for seed, train_rows in orders.items()},
        "holdout": holdout, "test_batches": len(rows), "prior_fingerprints": fingerprints,
        "prior_ledger_sha256": ledgers, "prior_cost_cny": cost / NANO_PER_CNY,
        "remaining_cny": remaining / NANO_PER_CNY, "max_api_attempts": max_attempts,
        "training_selection_rule": "Rank released train rows by SHA256 of namespace, train separator and context hash. "
                                   "Exclude prior training rows and normalized sentences in train selections, val or test.",
    }
    print(json.dumps({"test_batches": len(rows), "train_targets": stages,
                      "remaining_budget_cny": plan["remaining_cny"]}), flush=True)
    if not args.run:
        return
    from eval.finance import data_processor
    from ace.prompts.generator import GENERATOR_PROMPT
    if _sha256(GENERATOR_PROMPT) != plan["generator_prompt_sha256"]:
        raise ValueError("Generator prompt changed")
    manifest_path = args.out / "run_manifest.json"
    if args.resume:
        previous = json.loads(manifest_path.read_bytes())
        run_id = previous.pop("run_id")
        if json.dumps(previous, sort_keys=True) != json.dumps(plan, sort_keys=True):
            raise ValueError("Frozen run plan differs from the resume plan")
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        run_id = uuid4().hex
        write_json(manifest_path, {"run_id": run_id, **plan})
    def client_at_cap(cap):
        return BudgetedDeepSeekClient(
            args.out / "usage.jsonl", budget_cny=Decimal(remaining) / NANO_PER_CNY,
            max_attempts=max_attempts, max_input_tokens=INPUT_BOUND, max_output_tokens=cap, timeout_s=45,
        )

    client = client_at_cap(INITIAL_CAP)
    processor = data_processor.DataProcessor("finer")
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(import_module("ace.ace"), "initialize_clients",
                                             lambda provider: (client, client, client)))
            role_call = ace_timed_llm_call
            if args.recover_seed is not None:
                seed = args.recover_seed
                if seed not in originals:
                    raise ValueError("Unknown recovery seed")
                directory = args.out / f"trial_{seed}"
                last = read_rows(directory / "training_rows.jsonl")[-1]
                if (last.get("error_type") != "TruncatedCompletion" or last.get("attempt") != STEP_ATTEMPTS):
                    raise ValueError("Seed is not stopped at a truncation retry limit")
                rescue = client_at_cap(RECOVERY_CAP)
                stack.callback(rescue.close)

                def role_call(api_client, provider, model, prompt, **kwargs):
                    try:
                        return ace_timed_llm_call(api_client, provider, model, prompt, **kwargs)
                    except TruncatedCompletion:
                        return ace_timed_llm_call(rescue, provider, model, prompt, **{
                            **kwargs, "max_tokens": RECOVERY_CAP,
                            "call_id": kwargs["call_id"] + "_cap_rescue",
                        })

            for role in ("generator", "reflector", "curator"):
                stack.enter_context(patch.object(import_module(f"ace.core.{role}"), "timed_llm_call", role_call))
            stack.enter_context(patch.object(data_processor, "eval", ast.literal_eval, create=True))
            if args.recover_seed is not None:
                write_json(directory / "recovery_policy.json", {
                    "seed": seed, "resume_at_step": last["step"], "initial_cap": INITIAL_CAP,
                    "truncation_retry_cap": RECOVERY_CAP, "scope": "training only",
                })
                with ((directory / "recovery_console.log").open("a", encoding="utf-8") as log, redirect_stdout(log)):
                    train(seed, originals[seed], orders[seed], processor, directory, run_id)
                print(f"Trial {seed} recovery finished", flush=True)
                return
            with ((args.out / "training_console.log").open("a", encoding="utf-8") as log,
                  redirect_stdout(log), ThreadPoolExecutor(max_workers=3) as pool):
                futures = {
                    pool.submit(train, seed, c, orders[seed], processor, args.out / f"trial_{seed}", run_id): seed
                    for seed, c in originals.items()
                }
                books = {futures[f]: f.result() for f in as_completed(futures)}
            write_json(args.out / "checkpoint_hashes.json", {
                seed: {dose: _sha256(book) for dose, book in stages.items()} for seed, stages in books.items()
            })
            print("All training checkpoints ready", flush=True)
            system = new_ace(INITIAL_CAP)
            empty, generator = system.playbook, system.generator
            jobs = [(i, row, k, seed) for i, row in enumerate(rows) for k, seed in enumerate(originals)]
            rotations = [ARMS[k:] + ARMS[:k] for k in range(len(ARMS))]
            permutations = [p for order in rotations for p in (order, order[::-1])]

            def evaluate(job):
                i, row, k, seed = job
                order = permutations[(i + k) % len(permutations)]
                pair = {"row_id": row["row_id"], "seed": seed, "arm_order": order}
                for arm in order:
                    book = empty if arm == "base" else books[seed][int(arm.split("_")[1])]
                    current = SimpleNamespace(generator=generator, playbook=book)
                    pair[arm] = evaluation_row(current, processor, "fresh_test", row, seed, arm, run_id)
                return pair

            output = args.out / "paired_rows.jsonl"
            pairs = read_rows(output) if args.resume and output.exists() else []
            done = {(p["row_id"], p["seed"]) for p in pairs}
            if len(done) != len(pairs) or not done.issubset({(r["row_id"], s) for _, r, _, s in jobs}):
                raise ValueError("Invalid resumed test records")
            todo = [job for job in jobs if (job[1]["row_id"], job[3]) not in done]
            with (output.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=WORKERS) as pool):
                for pair in pool.map(evaluate, todo):
                    pairs.append(pair)
                    append_json(f, pair)
                    if len(pairs) % 24 == 0 or len(pairs) == len(jobs):
                        print(f"Tested {len(pairs)}/{len(jobs)} matched batches", flush=True)
        usage = client.summary()
    finally:
        client.close()
    complete = [p for p in pairs if all(
        p[arm]["status"] == "scored" and p[arm]["predicted_positions"] == 4 and not p[arm]["missing_positions"]
        for arm in ARMS
    )]
    events = read_rows(args.out / "usage.jsonl")
    settled = [e for e in events if e["event"] == "settle"]
    result = {
        "primary": metrics(pairs, intervals=True), "all_four_complete": metrics(complete),
        "trials": {seed: metrics([p for p in pairs if p["seed"] == seed]) for seed in originals},
        "test_prompt_tokens": {arm: sum(e["usage"]["prompt_tokens"] for e in settled
                                       if e["phase"] == "fresh_test" and e["arm"] == arm) for arm in ARMS},
        "observed_fingerprints": sorted({e["system_fingerprint"] for e in settled if e.get("system_fingerprint")}),
        "usage": usage,
    }
    write_json(args.out / "summary.json", result)
    print(json.dumps(result["primary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
