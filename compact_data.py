"""Verify and load the frozen FiNER prompt-development and evaluation sets."""

from __future__ import annotations

import json

from finer_data import ROOT, _sha256
from rescue_data import _source_rows


MANIFEST = "compact_manifest.json"
PRIOR_MANIFESTS = ("finer_manifest.json", "rescue_manifest.json")
PRIOR_LEDGER = "evidence/deepseek_ledger_combined.jsonl"
RANK_PREFIX = "ACE-FINER-COMPACT-V1|"
COUNTS = {"dev": 8, "test": 32}


def _ranked_rows(split, source):
    rows = list(_source_rows(split, source))
    for record, _ in rows:
        digest = record["original_context_sha256"]
        record["selection_rank_sha256"] = _sha256(f"{RANK_PREFIX}{split}|{digest}")
    return sorted(rows, key=lambda pair: (
        pair[0]["selection_rank_sha256"], pair[0]["physical_line_1based"],
    ))


def _sentence_hashes(records):
    return {digest for record in records for digest in record["sentence_sha256"]}


def _select_disjoint(rows, excluded_ids, excluded_sentences, *, require_four_distinct):
    selected = []
    seen = excluded_sentences.copy()
    for record, row in rows:
        sentences = set(record["sentence_sha256"])
        if record["row_id"] in excluded_ids or sentences & seen:
            continue
        if require_four_distinct and len(sentences) != 4:
            continue
        selected.append((record, row))
        seen.update(sentences)
    return selected


def _verify_tags(row):
    menu = row["question"].split("tags options:", 1)[1].split(". Answer the following", 1)[0]
    options = {tag.strip().casefold() for tag in menu.split(",") if tag.strip()}
    tags = [tag.strip().casefold() for tag in row["target"].split(",")]
    if len(tags) != 4 or not set(tags).issubset(options):
        raise ValueError(f"Expected four candidate-menu tags in {row['row_id']}")


def _build_selection(ledger_prefix_bytes):
    prior_bytes = [(ROOT / name).read_bytes() for name in PRIOR_MANIFESTS]
    finer, rescue = [json.loads(payload) for payload in prior_bytes]
    sources = finer["sources"]
    if rescue["upstream_commit"] != finer["upstream_commit"] or rescue["sources"] != sources:
        raise ValueError("Earlier manifests do not pin the same FiNER sources")
    ranked = {split: _ranked_rows(split, source) for split, source in sources.items()}
    records = {record["row_id"]: record for rows in ranked.values() for record, _ in rows}

    ledger = (ROOT / PRIOR_LEDGER).read_bytes()
    if not 0 < ledger_prefix_bytes <= len(ledger):
        raise ValueError("The archived ledger is missing the frozen prior prefix")
    ledger = ledger[:ledger_prefix_bytes]
    if not ledger.endswith(b"\n"):
        raise ValueError("The prior ledger prefix must end at a complete JSONL record")
    prior_exposed = set()
    for line in ledger.splitlines():
        event = json.loads(line)
        row_id = event.get("row_id")
        if event.get("event") == "reserve" and isinstance(row_id, str) and row_id.startswith("finer:"):
            prior_exposed.add(row_id)
    prior_reserved = {
        record["row_id"] for phase in finer["selections"].values()
        for split in phase.values() for record in split
    } | {record["row_id"] for record in rescue["records"]}
    excluded_ids = prior_exposed | prior_reserved
    if not excluded_ids.issubset(records):
        raise ValueError("An earlier manifest or ledger references an unknown FiNER row")
    excluded_sentences = _sentence_hashes(records[row_id] for row_id in excluded_ids)
    dev = _select_disjoint(
        ranked["val"], excluded_ids,
        excluded_sentences | _sentence_hashes(record for record, _ in ranked["test"]),
        require_four_distinct=True,
    )
    test = _select_disjoint(
        ranked["test"], excluded_ids,
        excluded_sentences | _sentence_hashes(
            record for split in ("train", "val") for record, _ in ranked[split]
        ),
        require_four_distinct=False,
    )

    manifest = {
        "upstream_commit": finer["upstream_commit"],
        "rank_prefix": RANK_PREFIX,
        "selection_rule": (
            "Rank by SHA256(rank_prefix + source split + '|' + original context SHA256), "
            "breaking ties by physical line. Exclude prior manifested and API-exposed rows "
            "and their normalized sentences from both sets. Development uses validation "
            "batches with four distinct sentences and no sentence overlap with the full test "
            "file. Evaluation uses test batches with no sentence overlap with the full train "
            "or validation files. Greedily exclude sentence overlap between accepted batches "
            "in each set. Evaluation permits separate entity questions sharing one sentence "
            "inside an original batch. Sentence normalization is NFKC and collapsed whitespace."
        ),
        "prior_manifests": {name: _sha256(payload) for name, payload in zip(PRIOR_MANIFESTS, prior_bytes)},
        "prior_ledger": {"file": PRIOR_LEDGER, "prefix_bytes": len(ledger), "sha256": _sha256(ledger)},
        "prior_reserved_row_count": len(prior_reserved),
        "prior_exposed_row_count": len(prior_exposed),
        "sources": sources,
        "eligible_batches": {"dev": len(dev), "test": len(test)},
        "selections": {},
        "ordered_row_ids_sha256": {},
    }
    splits = {}
    for split, pool in (("dev", dev), ("test", test)):
        count = COUNTS[split]
        if len(pool) < count:
            raise ValueError(f"Expected at least {count} eligible FiNER {split} batches")
        selected, rows = map(list, zip(*pool[:count]))
        for row in rows:
            _verify_tags(row)
        manifest["selections"][split] = selected
        manifest["ordered_row_ids_sha256"][split] = _sha256("\n".join(row["row_id"] for row in rows))
        splits[split] = rows
    return manifest, splits


def load_compact_splits() -> dict[str, list[dict]]:
    """Recompute the frozen eight development and 32 evaluation batches."""
    frozen = json.loads((ROOT / MANIFEST).read_bytes())
    computed, splits = _build_selection(frozen["prior_ledger"]["prefix_bytes"])
    if computed != frozen:
        raise ValueError("FiNER compact selection differs from the frozen manifest")
    return splits
