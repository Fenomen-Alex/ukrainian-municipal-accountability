"""Deterministic Gemini batch preparation for the gold annotation set.

Reads ``ml/data/gold/annotation_set.jsonl`` (400 records, in order), slices
it into fixed-size batches (default 40) and writes one JSON file per batch
plus a ``manifest.json`` under ``ml/data/gold/gemini_batches/``.

Each batch contains only what Gemini needs to annotate: record ``id``,
``text`` and ``source_kind`` (as weak-supervision context), plus a compact
instruction block.

Run::

    python3 -m ml.gold.prepare_gemini --batch-size 40

Output is fully deterministic: no timestamps, no random values; repeated runs
are byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path

SCHEMA_VERSION = "gold-schema-1.0"
PREPARE_VERSION = "1.0.0"

INSTRUCTIONS = textwrap.dedent(
    """\
    You are annotating citizen appeals to a Ukrainian municipality for a gold
    evaluation set. The complaint texts are in Ukrainian. Annotate the
    semantics of what is written -- do not reproduce surface labels.

    WEAK SUPERVISION WARNING
    -------------------------
    "source_kind" is weak supervision from the source dataset. It may be
    wrong, too coarse, or machine-assigned. Classify each complaint from the
    text, not from source_kind. If they disagree, the text wins.

    TASK
    ----
    For every supplied record produce EXACTLY ONE output entry:

      {"id": "<same id>", "annotation": {"topics": [...]}}

    Every output "id" must exactly equal a supplied record id.

    TOPICS
    ------
    "topics" is an array; it may be empty.
    Add one topic per INDEPENDENT complaint/request in the text. Independent
    issues are separate topics even when they appear in the same paragraph.
    Each topic object has exactly these fields:
      - domain          : one value from the controlled vocabulary below.
      - issue           : the concrete problem, stated neutrally (short).
      - object          : the target (address, facility, service, body) --
                          a short string or small object.
      - requested_action: what the complainant asks the municipality to do.
      - attributes      : extra structured facts ACTUALLY present in the text
                          (addresses, dates, building numbers, quantities);
                          empty object {} when there are none.
    Use "topics": [] ONLY when no meaningful municipal issue can be
    identified from the text (for example, pure administrative docket text).

    RULES
    -----
    - Do not invent facts. Never infer dates, amounts, history or intent that
      is not in the text.
    - Do not modify or paraphrase the complaint text; copy IDs exactly.
    - Avoid unnecessary PII: omit names and phone numbers unless directly
      relevant to the complaint.
    - Leave fields empty when the information is unavailable.
    - Do not treat source_kind as ground truth.
    - Return raw JSON only: no markdown fences, no commentary, no trailing
      prose.
    - Emit exactly one annotation object per supplied record id. Never omit a
      record. Never add records.

    CONTROLLED VOCABULARY (domain)
    ------------------------------
    roads, water, heating, housing, transport, sanitation, electricity,
    construction, benefits, government, commerce, payments, other

    FULL SCHEMA & GUIDE
    -------------------
    The authoritative schema is ml/data/gold/annotation_schema.json and the
    annotation guide is ml/data/gold/annotation_guide.md in the same
    repository. Follow them for exact field semantics.
    """
)


def sha256_hex(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_annotation_records(data_dir: Path) -> list[dict]:
    path = data_dir / "gold" / "annotation_set.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"annotation set not found: {path}")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def prepare_batches(
    data_dir: Path, batch_size: int = 40
) -> tuple[list[dict], dict]:
    """Slice the annotation set into deterministic batches.

    Returns ``(batches, manifest)`` where each batch is a dict to be written
    as ``batch-NNN.json``.
    """
    records = load_annotation_records(data_dir)
    total = len(records)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    batches = []
    for start in range(0, total, batch_size):
        chunk = records[start : start + batch_size]
        number = start // batch_size + 1
        batch_id = f"batch-{number:03d}"
        batches.append(
            {
                "batch_id": batch_id,
                "schema_version": SCHEMA_VERSION,
                "instructions": INSTRUCTIONS,
                "records": [
                    {
                        "id": rec["id"],
                        "text": rec["text"],
                        "source_kind": rec["source_kind"],
                    }
                    for rec in chunk
                ],
            }
        )

    source_sha = sha256_hex(data_dir / "gold" / "annotation_set.jsonl")
    manifest = {
        "total_records": total,
        "batch_size": batch_size,
        "num_batches": len(batches),
        "batch_ids": [b["batch_id"] for b in batches],
        "records_per_batch": {
            b["batch_id"]: len(b["records"]) for b in batches
        },
        "ids_per_batch": {b["batch_id"]: [r["id"] for r in b["records"]] for b in batches},
        "source_sha256": source_sha,
        "schema_version": SCHEMA_VERSION,
        "generation": {
            "prepare_script_version": PREPARE_VERSION,
            "note": "deterministic; no timestamps by design",
            "record_fields": ["id", "text", "source_kind"],
        },
    }
    return batches, manifest


def write_outputs(data_dir: Path, batch_size: int = 40) -> dict:
    out_dir = data_dir / "gold" / "gemini_batches"
    out_dir.mkdir(parents=True, exist_ok=True)

    batches, manifest = prepare_batches(data_dir, batch_size)
    for batch in batches:
        path = out_dir / f"{batch['batch_id']}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(batch, fh, ensure_ascii=False, indent=2)
            fh.write("\n")

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python3 -m ml.gold.prepare_gemini",
        description="Build deterministic Gemini annotation batches.",
    )
    parser.add_argument("--data-dir", default="ml/data", type=Path)
    parser.add_argument("--batch-size", default=40, type=int)
    args = parser.parse_args()

    manifest = write_outputs(args.data_dir, args.batch_size)
    print(
        f"Wrote {manifest['num_batches']} batches "
        f"(size {manifest['batch_size']}, {manifest['total_records']} records) -> "
        f"{args.data_dir / 'gold' / 'gemini_batches'}"
    )
    print("Manifest:", args.data_dir / "gold" / "gemini_batches" / "manifest.json")


if __name__ == "__main__":
    main()