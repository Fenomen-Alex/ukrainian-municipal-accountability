"""Disk cache for frozen per-split embedding matrices.

Caching is correctness-critical: the experiment reuses a matrix only when it
was produced for the exact same records (same uids, kinds and contents) and the
exact same model. A content-addressable fingerprint guards against stale or
mixed-up splits.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np


def fingerprint(records: list[dict]) -> str:
    """Order-independent sha256 over the semantic identity of each record."""
    identity = sorted(
        (r.get("uid"), r.get("kind"), r.get("content", "")) for r in records
    )
    h = hashlib.sha256()
    for uid, kind, content in identity:
        h.update(f"{uid}\0{kind}\0{content}\0".encode("utf-8"))
    return h.hexdigest()


def _matrix_path(cache_dir: str, split: str) -> str:
    return os.path.join(cache_dir, f"{split}.npy")


def _meta_path(cache_dir: str, split: str) -> str:
    return os.path.join(cache_dir, f"{split}.meta.json")


def save_embeddings(
    matrix: np.ndarray,
    model_id: str,
    split: str,
    records: list[dict],
    cache_dir: str,
) -> dict:
    """Persist a split's embedding matrix plus a validating metadata file."""
    os.makedirs(cache_dir, exist_ok=True)
    matrix = np.asarray(matrix, dtype=np.float32)
    meta = {
        "model_id": model_id,
        "split": split,
        "n": int(matrix.shape[0]),
        "dim": int(matrix.shape[1]),
        "fingerprint": fingerprint(records),
    }
    np.save(_matrix_path(cache_dir, split), matrix)
    with open(_meta_path(cache_dir, split), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)
    return meta


def load_embeddings(
    model_id: str,
    split: str,
    records: list[dict],
    cache_dir: str,
) -> np.ndarray | None:
    """Return the cached matrix only if it is valid for these records.

    Invalidates on: missing files, wrong model, wrong split marker, row-count
    or fingerprint mismatch. Returns ``None`` so the caller regenerates.
    """
    if not os.path.exists(_matrix_path(cache_dir, split)) or not os.path.exists(
        _meta_path(cache_dir, split)
    ):
        return None
    try:
        with open(_meta_path(cache_dir, split), encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    expect = {
        "model_id": model_id,
        "split": split,
        "n": len(records),
        "dim": meta.get("dim"),
        "fingerprint": fingerprint(records),
    }
    for key, value in expect.items():
        if meta.get(key) != value:
            return None
    try:
        matrix = np.load(_matrix_path(cache_dir, split))
    except (OSError, ValueError):
        return None
    if matrix.shape[0] != len(records):
        return None
    return np.asarray(matrix, dtype=np.float32)