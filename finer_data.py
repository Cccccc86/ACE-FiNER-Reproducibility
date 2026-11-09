"""Load the frozen FiNER samples and their training evidence."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
COUNTS = {
    "pilot": (2, 2, 2),
    "compact": (12, 8, 48),
    "standard": (24, 16, 64),
}
HEADING = re.compile(r"(?m)^([1-4])\. What is best tag for entity ")
QUESTION = re.compile(r'([1-4])\. What is best tag for entity "(.+?)" in sentence: "(.+)"', re.DOTALL)


def _sha256(payload: bytes | str) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_evidence(question: str) -> tuple[str, list[str]]:
    body, separator, cue = question.rstrip().rpartition("\n")
    headings = list(HEADING.finditer(body))
    if not separator or cue.strip() != "Output US GAAP tags:":
        raise ValueError("Missing the empty final FiNER output cue")
    if [match[1] for match in headings] != list("1234"):
        raise ValueError("Expected exactly four FiNER questions numbered 1 through 4")
    prefix = body[:headings[0].start()]
    if "tags options:" not in prefix or ". Answer the following" not in prefix:
        raise ValueError("Missing the FiNER candidate menu and instructions")

    sentences = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(body)
        match = QUESTION.fullmatch(body[heading.start():end].strip())
        if not match or not match[2].strip() or not match[3].strip():
            raise ValueError(f"Malformed entity or sentence in FiNER question {index + 1}")
        sentences.append(" ".join(unicodedata.normalize("NFKC", match[3]).split()))
    return body[headings[0].start():].strip(), sentences


def load_splits(pilot: bool, scope: str) -> dict[str, list[dict]]:
    """Load compact or standard rows, or the reserved pilot when pilot is true."""
    if scope not in ("compact", "standard"):
        raise ValueError(f"Unknown FiNER scope: {scope!r}")

    manifest = json.loads((ROOT / "finer_manifest.json").read_text(encoding="utf-8"))
    selected = {"pilot": {}, "standard": {}}
    seen_sentences: set[str] = set()
    for split_index, split in enumerate(SPLITS):
        source = manifest["sources"][split]
        payload = (ROOT / source["file"]).read_bytes()
        if _sha256(payload) != source["sha256"]:
            raise ValueError(f"FiNER source SHA-256 mismatch: {source['file']}")
        lines = payload.splitlines()
        if len(lines) != source["rows"]:
            raise ValueError(f"FiNER source line count mismatch: {source['file']}")

        for phase in selected:
            records = manifest["selections"][phase][split]
            ranks = [record["selection_rank_sha256"] for record in records]
            if len(records) != COUNTS[phase][split_index] or ranks != sorted(ranks):
                raise ValueError(f"Invalid count or rank order in FiNER {phase}/{split}")

            processed = []
            for record in records:
                line = record["physical_line_1based"]
                row_id = f"finer:{split}:{line:06d}"
                if not 1 <= line <= len(lines) or row_id != record["row_id"]:
                    raise ValueError(f"Invalid FiNER row reference: {record['row_id']}")
                raw_line = lines[line - 1]
                if _sha256(raw_line) != record["jsonl_line_sha256"]:
                    raise ValueError(f"FiNER row SHA-256 mismatch: {row_id}")

                raw = json.loads(raw_line)
                question, target = raw["context"], raw["target"]
                if "Input: " in question and "Instruction: " in question:
                    raise ValueError(f"Unexpected upstream preprocessing format: {row_id}")
                labels = target.split(",")
                if len(labels) != 4 or any(not label.strip() for label in labels):
                    raise ValueError(f"Expected four FiNER targets: {row_id}")
                evidence, sentences = _parse_evidence(question)
                context_hash = _sha256(question)
                rank = _sha256(f"{manifest['selection_seed']}|{split}|{context_hash}")
                sentence_hashes = [_sha256(sentence) for sentence in sentences]
                if (
                    context_hash != record["original_context_sha256"]
                    or rank != record["selection_rank_sha256"]
                    or _sha256(evidence) != record["curator_evidence_sha256"]
                    or sentence_hashes != record["sentence_sha256"]
                ):
                    raise ValueError(f"FiNER text SHA-256 mismatch: {row_id}")
                if seen_sentences.intersection(sentence_hashes):
                    raise ValueError(f"Sentence repeats across selected FiNER rows: {row_id}")
                seen_sentences.update(sentence_hashes)

                # The upstream parser keeps these complete questions in `question`.
                processed.append({
                    "row_id": row_id,
                    "context": "",
                    "question": question,
                    "target": target,
                    "curator_evidence": evidence,
                })
            selected[phase][split] = processed

    if pilot:
        return selected["pilot"]
    return {
        split: selected["standard"][split][:count]
        for split, count in zip(SPLITS, COUNTS[scope])
    }
