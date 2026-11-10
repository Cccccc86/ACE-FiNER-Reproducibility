"""Verify the report's final three experiments offline, using saved predictions."""

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
import unicodedata


ROOT = Path(__file__).resolve().parent
SEEDS = (17, 29, 43)
SCALE, CONFIRM, COMPACT = "training_scale", "confirm_100", "answer_only_100"
ARMS = {
    SCALE: ("base", "ace_24", "ace_48", "ace_100"),
    CONFIRM: ("base", "ace_upstream"),
    COMPACT: ("verbose", "answer_only"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def tags(text):
    return [tag.strip().casefold() for tag in (text or "").split(",")]


def source_data(pins):
    data = {}
    sentences = defaultdict(set)
    for split, pin in pins.items():
        path = (ROOT / pin["file"]).resolve()
        require(path.is_relative_to(ROOT), f"source path outside repository: {split}")
        payload = path.read_bytes()
        lines = payload.splitlines()
        require(sha(payload) == pin["sha256"] and len(lines) == pin["rows"], f"source mismatch: {split}")
        for index, line in enumerate(lines, 1):
            raw = json.loads(line)
            context = raw["context"]
            body = context.rstrip().rsplit("\n", 1)[0]
            evidence = re.split(r"(?m)^[1-4]\. What is best tag for entity ", body)[1:]
            hashes = []
            for part in evidence:
                match = re.fullmatch(r'".+?" in sentence: "(.+)"', part.strip(), re.S)
                require(match is not None, f"source question malformed: {split}/{index}")
                hashes.append(sha(" ".join(unicodedata.normalize("NFKC", match[1]).split())))
            gold = tags(raw["target"])
            require(len(hashes) == len(gold) == 4 and all(gold), f"invalid source row: {split}/{index}")
            row_id = f"finer:{split}:{index:06d}"
            data[row_id] = (sha(line), sha(context), hashes, gold)
            sentences[split].update(hashes)
    return data, sentences


def verify_selection(manifest, split, pins, data, sentences, reserved):
    require(manifest["sources"] == pins, "run uses different sources")
    selected = manifest["selection"]
    require(len(selected) == 64, "expected 64 held-out batches")
    blocked = set().union(*(values for name, values in sentences.items() if name != split))
    blocked.update(h for row_id in reserved if row_id.startswith(f"finer:{split}:") for h in data[row_id][2])
    candidates = sorted((sha(manifest["rank_prefix"] + value[1]), row_id)
                        for row_id, value in data.items()
                        if row_id.startswith(f"finer:{split}:") and row_id not in reserved)
    ranked, seen = [], blocked.copy()
    for _, row_id in candidates:
        hashes = data[row_id][2]
        if seen.isdisjoint(hashes):
            ranked.append(row_id)
            seen.update(hashes)
    require(len(ranked) == manifest["eligible_batches"]
            and [r["row_id"] for r in selected] == ranked[:len(selected)],
            "selected rows differ from the frozen hash ranking")
    result, ranks = {}, []
    for record in selected:
        row_id = record["row_id"]
        require(row_id == f"finer:{split}:{record['physical_line_1based']:06d}" and row_id not in reserved,
                f"invalid or previously selected row: {row_id}")
        line_hash, context_hash, hashes, gold = data[row_id]
        rank = sha(manifest["rank_prefix"] + context_hash)
        require((record["jsonl_line_sha256"], record["original_context_sha256"],
                 record["sentence_sha256"], record["selection_rank_sha256"])
                == (line_hash, context_hash, hashes, rank), f"selection hash mismatch: {row_id}")
        require(blocked.isdisjoint(hashes), f"repeated source sentence: {row_id}")
        blocked.update(hashes)
        result[row_id] = gold
        ranks.append(rank)
    require(ranks == sorted(ranks), "selection order differs from recorded hash ranking")
    return result


def verify_checkpoints(path, manifest):
    hashes = load(path / "checkpoint_hashes.json")
    for seed in SEEDS:
        recorded = hashes[str(seed)]
        earlier = load(path.parent / "training_dose" / f"trial_{seed}" / "checkpoint.json")
        require(earlier["completed_total"] == 24
                and sha(earlier["playbook"]) == earlier["playbook_sha256"]
                == recorded["24"] == manifest["checkpoint_24_sha256"][str(seed)],
                f"24-batch hash mismatch: seed {seed}")
        for dose in (48, 100):
            checkpoint = load(path / f"trial_{seed}" / f"checkpoint_{dose}.json")
            require((checkpoint["seed"], checkpoint["last_step"], checkpoint["completed_total"]) == (seed, dose, dose)
                    and sha(checkpoint["playbook"]) == checkpoint["playbook_sha256"] == recorded[str(dose)],
                    f"checkpoint mismatch: seed {seed}, dose {dose}")
    return {str(seed): hashes[str(seed)]["100"] for seed in SEEDS}


def verify_usage(path, manifest, summary, expected, arms, name):
    entries = rows(path / "usage.jsonl")
    config = entries[0]
    require(config["event"] == "config" and config["currency"] == "CNY"
            and config["model"] == manifest["model"] and config["base_url"] == manifest["base_url"]
            and config["budget_nano_cny"] == round(manifest.get("available_cny", manifest.get("remaining_cny")) * 1e9),
            "ledger config mismatch")
    hit_price, miss_price, out_price = (config[key] for key in (
        "peak_hit_nano_per_token", "peak_miss_nano_per_token", "peak_output_nano_per_token",
    ))
    reserved, settled, unknown = {}, set(), set()
    test_calls, truncations, fingerprint = Counter(), Counter(), set()
    token_totals = defaultdict(Counter)
    spent = 0
    for event in entries[1:]:
        key = event["attempt_id"]
        kind = event["event"]
        require(event["run_id"] == manifest["run_id"], "ledger run ID mismatch")
        if kind == "reserve":
            require(key not in reserved and event["model_requested"] == manifest["model"]
                    and event["max_tokens"] == 1536 and event["temperature"] == 0
                    and event["thinking"] == "disabled" and event["json_mode"] is True,
                    "duplicate reserve or changed decoding settings")
            estimate = event["input_token_upper_bound"] * miss_price + event["max_tokens"] * out_price
            require(estimate == event["reserved_nano_cny"], "incorrect request reservation")
            reserved[key] = event
            continue
        require(key in reserved and key not in settled | unknown, "unmatched or duplicate ledger completion")
        if kind == "unknown":
            unknown.add(key)
            continue
        require(kind == "settle" and event["model_reported"] == manifest["model"]
                and not event["limit_violation"], "invalid request completion")
        settled.add(key)
        fingerprint.add(event["system_fingerprint"])
        usage = event["usage"]
        require(usage["prompt_tokens"] == usage["prompt_cache_hit_tokens"] + usage["prompt_cache_miss_tokens"]
                and 0 <= usage["prompt_tokens"] <= reserved[key]["input_token_upper_bound"]
                and 0 <= usage["completion_tokens"] <= 1536, "token bounds or cache accounting mismatch")
        cost = (usage["prompt_cache_hit_tokens"] * hit_price + usage["prompt_cache_miss_tokens"] * miss_price
                + usage["completion_tokens"] * out_price)
        require(cost == event["cost_peak_nano_cny"], "request cost mismatch")
        spent += cost
        if event["phase"] == "train":
            require(name == SCALE and event["row_id"] in manifest["extension_order"][str(event["trial"])],
                    "unexpected training request")
            continue
        arm = event["phase"].split(".")[1] if name == COMPACT else event["arm"]
        require(arm in arms and event["role"] == "generator", "unexpected evaluation request")
        test_calls[event["row_id"], event["trial"], arm] += 1
        truncations[arm] += event["finish_reason"] == "length"
        token_totals[arm].update({"completion": usage["completion_tokens"], "input": usage["prompt_tokens"],
                                 "cost": cost})
    require(test_calls == Counter((row, seed, arm) for row, seed in expected for arm in arms),
            "ledger does not cover every paired evaluation")
    require(len(fingerprint) == 1 and None not in fingerprint, "model fingerprint missing or changed within run")
    pending = sum(event["reserved_nano_cny"] for key, event in reserved.items() if key not in settled)
    recorded = summary["usage"]
    require((len(reserved), len(settled), len(unknown)) == (
        recorded["attempts"], recorded["settled_attempts"], recorded["unknown_attempts"])
        and spent == round(recorded["settled_peak_cost_cny"] * 1e9)
        and pending == round(recorded["unresolved_reserved_cny"] * 1e9)
        and spent + pending <= config["budget_nano_cny"], "run usage or budget mismatch")
    return token_totals, truncations, spent, pending, fingerprint


def score(path, name, gold):
    arms = ARMS[name]
    expected = {(row, seed) for row in gold for seed in SEEDS}
    observed, orders = set(), Counter()
    totals, complete = Counter(), Counter()
    grouped = defaultdict(lambda: defaultdict(list))
    statuses = defaultdict(Counter)
    compliant, malformed = Counter(), Counter()
    for pair in rows(path / "paired_rows.jsonl"):
        row_id = pair["row_id"]
        seed = pair["trial"] if name == COMPACT else pair["seed"]
        key = row_id, seed
        require(key in expected and key not in observed, f"invalid pair: {key}")
        observed.add(key)
        order = tuple(pair["arm_order"])
        require(len(order) == len(arms) and set(order) == set(arms), f"invalid arm order: {key}")
        orders[order] += 1
        answers, all_complete = [], True
        for arm in arms:
            record = pair["arms"][arm] if name == COMPACT else pair[arm]
            attempt = record["initial"] if name == COMPACT else record
            target = pair["target"] if name == COMPACT else record["target"]
            require(tags(target) == gold[row_id], f"target mismatch: {key}, {arm}")
            predicted = tags(attempt["prediction"])
            flags = tuple(a == b for a, b in zip(predicted + [""] * 4, gold[row_id]))
            require(sum(flags) == record["correct_labels"], f"score mismatch: {key}, {arm}")
            status = attempt["status"]
            statuses[arm][attempt.get("error_type", status)] += 1
            all_complete &= status == "scored" and len(predicted) == 4 and all(predicted)
            if name == COMPACT:
                require(record["rescue"] is None and record["final_status"] == status,
                        f"unexpected compact-output retry: {key}")
                shape = record["output_shape"]
                compliant[arm] += bool(shape and shape["follows_arm_contract"])
                malformed[arm] += bool(shape and not shape["valid_four_tag_output"])
            answers.append(flags)
        patterns = list(zip(*answers))
        totals.update(patterns)
        if all_complete:
            complete.update(patterns)
        baseline = 1 if name == SCALE else 0
        for index in range(baseline + 1, len(arms)):
            grouped[arms[index]][row_id].extend(int(p[index]) - int(p[baseline]) for p in patterns)
    require(observed == expected and len(set(orders.values())) == 1, "incomplete or unbalanced paired evaluation")
    return totals, complete, grouped, statuses, compliant, malformed, expected


def summarize_counts(patterns, arms):
    result = {}
    for index, arm in enumerate(arms):
        result[arm] = {"correct": sum(n for mask, n in patterns.items() if mask[index])}
        if index:
            result[arm].update(
                base_right_to_wrong=sum(n for mask, n in patterns.items() if mask[0] and not mask[index]),
                base_wrong_to_right=sum(n for mask, n in patterns.items() if not mask[0] and mask[index]),
            )
    return result


def check_counts(patterns, arms, recorded):
    counts = summarize_counts(patterns, arms)
    require(sum(patterns.values()) == recorded.get("positions", recorded.get("denominator")), "denominator mismatch")
    for arm in arms:
        for field, value in counts[arm].items():
            if field in recorded["arms"][arm]:
                require(value == recorded["arms"][arm][field], f"summary mismatch: {arm}/{field}")
    return counts


def interval(group, seed):
    values = [100 * statistics.mean(x) for x in group.values()]
    rng = random.Random(seed)
    draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(10000))
    return [draws[250], draws[9749]], draws[500]


def check_interval(actual, recorded):
    require(all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(actual, recorded, strict=True)),
            "paired bootstrap interval mismatch")


def prior_usage(evidence, reserved):
    compact = ROOT / "evidence" / "deepseek_ledger_compact.jsonl"
    files = [compact]
    for name in ("fresh_retest", "evidence_first", "training_dose"):
        path = evidence / name
        manifest = load(path / "run_manifest.json")
        reserved.update(r["row_id"] for r in manifest.get("selection", ()))
        reserved.update(manifest.get("test_row_ids", ()))
        files.append(path / "usage.jsonl")
    committed = 0
    for path in files:
        events = rows(path)
        held = {e["attempt_id"]: e for e in events if e["event"] == "reserve"}
        paid = {e["attempt_id"]: e for e in events if e["event"] == "settle"}
        reserved.update(e["row_id"] for e in held.values() if "row_id" in e)
        committed += sum(paid[key]["cost_peak_nano_cny"] if key in paid else e["reserved_nano_cny"]
                         for key, e in held.items())
    return committed


def verify(evidence):
    finer = load(ROOT / "finer_manifest.json")
    rescue = load(ROOT / "rescue_manifest.json")
    compact = load(ROOT / "compact_manifest.json")
    pins = finer["sources"]
    data, sentences = source_data(pins)
    reserved = {record["row_id"] for phase in finer["selections"].values()
                for split in phase.values() for record in split}
    reserved.update(record["row_id"] for record in rescue["records"])
    reserved.update(record["row_id"] for split in compact["selections"].values() for record in split)
    committed = prior_usage(evidence, reserved)
    books, fingerprint = None, None
    generator_source = ROOT / "ace" / "prompts" / "generator.py"
    assignments = ast.parse(generator_source.read_text(encoding="utf-8")).body
    prompt_hash = sha(next(ast.literal_eval(node.value) for node in assignments
                           if isinstance(node, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == "GENERATOR_PROMPT"
                                   for t in node.targets)))
    for name, arms in ARMS.items():
        path = evidence / name
        manifest, summary = (load(path / file) for file in ("run_manifest.json", "summary.json"))
        require(manifest["upstream_commit"] == finer["upstream_commit"], f"{name}: upstream commit mismatch")
        recorded_prompt = manifest["template_sha256"]["verbose"] if name == COMPACT else manifest["generator_prompt_sha256"]
        require(recorded_prompt == prompt_hash, f"{name}: original Generator prompt changed")
        if name == SCALE:
            books = verify_checkpoints(path, manifest)
            require(round(manifest["prior_cost_cny"] * 1e9) == committed, "scale starting budget mismatch")
        else:
            require(manifest["playbook_sha256"] == books, f"{name}: changed 100-batch checkpoints")
            require(manifest["prior_committed_nano_cny"] == committed, f"{name}: carry-forward budget mismatch")
            for previous in (SCALE, CONFIRM):
                if previous == name:
                    break
                pins_previous = manifest["extra_prior_runs"][previous]
                require(sha((evidence / previous / "run_manifest.json").read_bytes()) == pins_previous["manifest_sha256"]
                        and sha((evidence / previous / "usage.jsonl").read_bytes()) == pins_previous["ledger_sha256"],
                        f"{name}: prior run manifest or ledger changed")
        gold = verify_selection(manifest["holdout"] if name == SCALE else manifest,
                                "val" if name == COMPACT else "test", pins, data, sentences, reserved)
        reserved.update(gold)
        totals, complete, grouped, statuses, compliant, malformed, expected = score(path, name, gold)
        tokens, truncated, spent, pending, observed_fp = verify_usage(path, manifest, summary, expected, arms, name)
        if fingerprint is not None:
            require(observed_fp == fingerprint, f"{name}: model fingerprint changed")
        fingerprint = observed_fp
        committed += spent + pending
        require(committed <= 50_000_000_000, "cumulative budget exceeded")
        print(f"{name}: {sum(totals.values())} positions")
        if name == SCALE:
            check_counts(totals, arms, summary["primary"])
            check_counts(complete, arms, summary["all_four_complete"])
            require(all(statuses[a] == summary["primary"]["arms"][a]["statuses"]
                        and tokens[a]["input"] == summary["test_prompt_tokens"][a] for a in arms),
                    "scale response or input usage mismatch")
            for arm, group in grouped.items():
                check_interval(interval(group, 20261110)[0],
                               summary["primary"]["bootstrap_95pct_difference_vs_24_pp"][arm.split("_")[1]])
        elif name == CONFIRM:
            check_counts(totals, arms, summary["pooled"])
            check_counts(complete, arms, summary["both_complete"])
            transitions = {"".join(str(int(flag)) for flag in mask): n for mask, n in totals.items()}
            require(transitions == summary["base_ace_correct_transitions"]
                    and all(statuses[a] == summary["pooled"]["arms"][a]["status"] for a in arms),
                    "confirmation response or transition mismatch")
            bounds, _ = interval(grouped[arms[1]], manifest["bootstrap_seed"])
            check_interval(bounds, summary["paired_batch_bootstrap_95pct_delta_pp"])
            change = 100 * (transitions.get("01", 0) - transitions.get("10", 0)) / sum(totals.values())
            require(math.isclose(change, summary["pooled"]["delta_pp"]), "confirmation difference mismatch")
            print(f"  Paired difference {change:+.2f} pp (95% interval {bounds[0]:+.2f}, {bounds[1]:+.2f}).")
        else:
            counts = summarize_counts(totals, arms)
            for arm in arms:
                rec = summary["arms"][arm]
                require(counts[arm]["correct"] == rec["correct_labels"]
                        and sum(totals.values()) == rec["total_labels"]
                        and compliant[arm] == rec["contract_compliant_batches"]
                        and malformed[arm] == rec["malformed_four_tag_batches"]
                        and statuses[arm] == summary["status_by_arm"][arm]
                        and truncated[arm] == rec["final_truncations"]
                        and tokens[arm]["completion"] == rec["usage_including_retries"]["completion_tokens"]
                        and tokens[arm]["cost"] == rec["usage_including_retries"]["settled_peak_cost_nano_cny"],
                        f"compact output or usage mismatch: {arm}")
            bounds, lower = interval(grouped[arms[1]], manifest["bootstrap_seed"])
            check_interval(bounds, summary["paired_batch_bootstrap_95pct_interval_pp"])
            savings = 1 - tokens[arms[1]]["completion"] / tokens[arms[0]]["completion"]
            require(math.isclose(lower, summary["paired_batch_bootstrap_one_sided_95pct_lower_pp"])
                    and math.isclose(savings, summary["completion_token_savings_fraction"]),
                    "compact noninferiority or token-savings summary mismatch")
            old, new = arms
            checks = {
                "accuracy_noninferiority_at_margin": lower > -manifest["accuracy_margin_pp"],
                "output_token_savings": savings >= manifest["minimum_completion_token_savings_fraction"],
                "no_extra_incomplete": statuses[new]["scored"] >= statuses[old]["scored"],
                "no_extra_malformed": malformed[new] <= malformed[old],
                "no_extra_truncations": truncated[new] <= truncated[old],
            }
            require(checks == summary["decision_checks"]
                    and all(checks.values()) == summary["passes_predeclared_gate"], "compact decision gate mismatch")
            print(f"  Completion tokens {tokens[arms[0]]['completion']:,} -> {tokens[arms[1]]['completion']:,}"
                  f" ({savings:.2%} saved). One-sided 95% lower bound {lower:+.2f} pp.")
            complete_counts = summarize_counts(complete, arms)
            print(f"  Complete pairs: {complete_counts[old]['correct']} vs {complete_counts[new]['correct']}"
                  f" correct out of {sum(complete.values())} positions per arm.")
        for index, arm in enumerate(arms):
            correct = sum(n for mask, n in totals.items() if mask[index])
            print(f"  {arm}: {correct}/{sum(totals.values())} ({correct / sum(totals.values()):.2%}),"
                  f" {truncated[arm]} truncated batches")
        print(f"  Peak estimated cost CNY {spent / 1e9:.4f}, reserved unknown CNY {pending / 1e9:.4f}.")
    print(f"Verified three experiments. Cumulative committed CNY {committed / 1e9:.4f}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT / "evidence", help="parent of the archived run directories")
    args = parser.parse_args()
    try:
        verify(args.evidence)
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration, SyntaxError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
