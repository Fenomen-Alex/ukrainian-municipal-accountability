"""Deterministic label mapping for the embedding experiment.

The embedding baseline must use exactly the same ``{kind: index}`` mapping and
label ordering as the TF-IDF baseline, so its outputs are directly comparable.
The mapping is produced from ``ml/data/labels.json`` (never rebuilt here).
"""

from __future__ import annotations

import json
import os


def load_label_map(path: str) -> dict[str, int]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def label_order(label_map: dict[str, int]) -> list[str]:
    """Labels sorted by their integer index (matches ``baseline.json``)."""
    return sorted(label_map, key=lambda k: label_map[k])


def to_indices(kinds: list[str], label_map: dict[str, int]) -> list[int]:
    """Map each record's ``kind`` to its integer label index."""
    return [label_map[k] for k in kinds]


def load_records(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def label_vectors_from_file(data_dir: str) -> dict[str, list[int]]:
    """Return ``{split: [label-index, ...]}`` for the three data files.

    The label map is read from ``labels.json`` exactly as the TF-IDF baseline
    does; every row maps to a label index in ``label_order``.
    """
    label_map = load_label_map(os.path.join(data_dir, "labels.json"))
    out: dict[str, list[int]] = {}
    for split in ("train", "validation", "test"):
        records = load_records(os.path.join(data_dir, f"{split}.jsonl"))
        out[split] = to_indices([r["kind"] for r in records], label_map)
    return out