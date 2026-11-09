"""Verify and load the outcome-blind FiNER rescue holdout."""

from __future__ import annotations

import json

from finer_data import ROOT, SPLITS, _parse_evidence, _sha256


RANK_PREFIX = "ACE-FINER-RESCUE-V1|"
TEST_COUNT = 64
PRIOR_MANIFEST = "finer_manifest.json"
PRIOR_LEDGER = "evidence/deepseek_ledger.jsonl"
RESCUE_MANIFEST = "rescue_manifest.json"


def _source_rows(split, source):
    payload = (ROOT / source["file"]).read_bytes()
    lines = payload.splitlines()
    if _sha256(payload) != source["sha256"] or len(lines) != source["rows"]:
        raise ValueError(f"FiNER source verification failed: {source['file']}")
    for number, line in enumerate(lines, 1):
        raw = json.loads(line)
        question = raw["context"]
        if "Instruction: " in question and "Input: " in question:
            raise ValueError(f"Unexpected FiNER preprocessing format: {split}/{number}")
        evidence, sentences = _parse_evidence(question)
        context_hash = _sha256(question)
        record = {
            "row_id": f"finer:{split}:{number:06d}",
            "physical_line_1based": number,
            "original_context_sha256": context_hash,
            "jsonl_line_sha256": _sha256(line),
            "curator_evidence_sha256": _sha256(evidence),
            "sentence_sha256": [_sha256(sentence) for sentence in sentences],
            "selection_rank_sha256": _sha256(RANK_PREFIX + context_hash),
        }
        row = {
            "row_id": record["row_id"],
            "context": "",
            "question": question,
            "target": raw["target"],
            "curator_evidence": evidence,
        }
        yield record, row


def _select_test(ledger_prefix_size=None):
    prior_manifest_bytes = (ROOT / PRIOR_MANIFEST).read_bytes()
    prior = json.loads(prior_manifest_bytes)
    ledger = (ROOT / PRIOR_LEDGER).read_bytes()
    if ledger_prefix_size is not None:
        if not 0 < ledger_prefix_size <= len(ledger):
            raise ValueError("The archived ledger is missing the frozen prior prefix")
        ledger = ledger[:ledger_prefix_size]
    if not ledger.endswith(b"\n"):
        raise ValueError("The prior ledger prefix must end at a complete JSONL record")

    ledger_test_ids = set()
    for line in ledger.splitlines():
        event = json.loads(line)
        row_id = event.get("row_id")
        if event.get("event") == "reserve" and isinstance(row_id, str) and row_id.startswith("finer:test:"):
            ledger_test_ids.add(row_id)

    reserved = [record for phase in prior["selections"].values()
                for split in phase.values() for record in split]
    excluded_ids = {record["row_id"] for record in reserved} | ledger_test_ids
    excluded_sentences = {digest for record in reserved for digest in record["sentence_sha256"]}
    sources = prior["sources"]
    for split in SPLITS:
        if split != "test":
            for record, _ in _source_rows(split, sources[split]):
                excluded_sentences.update(record["sentence_sha256"])

    test = list(_source_rows("test", sources["test"]))
    if not ledger_test_ids.issubset({record["row_id"] for record, _ in test}):
        raise ValueError("The prior ledger references unknown FiNER test rows")
    for record, _ in test:
        if record["row_id"] in excluded_ids:
            excluded_sentences.update(record["sentence_sha256"])

    test.sort(key=lambda item: (item[0]["selection_rank_sha256"], item[0]["physical_line_1based"]))
    eligible = []
    seen = excluded_sentences.copy()
    for record, row in test:
        sentences = set(record["sentence_sha256"])
        if record["row_id"] in excluded_ids or len(sentences) != 4 or sentences & seen:
            continue
        eligible.append((record, row))
        seen.update(sentences)
    if len(eligible) < TEST_COUNT:
        raise ValueError(f"Expected {TEST_COUNT} fresh test batches, found {len(eligible)}")

    records, rows = map(list, zip(*eligible[:TEST_COUNT]))
    for row in rows:
        labels = row["target"].split(",")
        if len(labels) != 4 or any(not label.strip() for label in labels):
            raise ValueError(f"Expected four labels in {row['row_id']}")

    manifest = {
        "upstream_commit": prior["upstream_commit"],
        "rank_prefix": RANK_PREFIX,
        "selection_rule": (
            "Rank by SHA256(rank_prefix + original context SHA256), breaking ties by physical line. "
            "Exclude previously manifested rows and ledger-seen test rows, and normalized sentences "
            "from all reserved rows, all ledger-seen test rows, and the full train and validation files. "
            "Greedily accept batches with four distinct normalized sentences absent from earlier "
            "accepted batches. Freeze the first 64. Sentence normalization is NFKC and collapsed whitespace."
        ),
        "prior_manifest": {"file": PRIOR_MANIFEST, "sha256": _sha256(prior_manifest_bytes)},
        "prior_ledger": {"file": PRIOR_LEDGER, "prefix_bytes": len(ledger), "sha256": _sha256(ledger)},
        "prior_ledger_test_ids": sorted(ledger_test_ids),
        "sources": sources,
        "eligible_batches": len(eligible),
        "test_batches": TEST_COUNT,
        "ordered_row_ids_sha256": _sha256("\n".join(row["row_id"] for row in rows)),
        "records": records,
    }
    return manifest, rows


def load_rescue_test() -> list[dict]:
    """Return the 64 frozen test rows after reproducing the selection locally."""
    frozen = json.loads((ROOT / RESCUE_MANIFEST).read_bytes())
    computed, rows = _select_test(frozen["prior_ledger"]["prefix_bytes"])
    if computed != frozen:
        raise ValueError("FiNER rescue selection differs from the frozen manifest")
    return rows


if __name__ == "__main__":
    rows = load_rescue_test()
    print(json.dumps({"test_batches": len(rows), "label_positions": 4 * len(rows)}))
