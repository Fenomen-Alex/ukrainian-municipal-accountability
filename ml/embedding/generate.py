"""Semantic-embedding generation for the Qwen3 Ukrainian embedding model.

Wraps ``sentence_transformers``: loads the model once, runs it on Apple Metal
(MPS) when available otherwise on CPU, and returns frozen float32 matrices.
Seeding is applied around encoding so a given input text maps to the same
embedding across runs of this module.
"""

from __future__ import annotations

import numpy as np


def pick_device(prefer: str = "auto") -> str:
    """Return the device to run inference on.

    ``"mps"`` (Apple Metal) when available, else ``"cpu"``. An explicit
    ``prefer`` value is honoured verbatim.
    """
    if prefer != "auto":
        return prefer
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # torch not importable / no backend
        pass
    return "cpu"


def load_embedder(model_id: str, device: str):
    """Load a SentenceTransformer model onto ``device``."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id, device=device)


def embed_texts(
    embedder,
    texts: list[str],
    seed: int = 0,
    batch_size: int = 32,
    device: str | None = None,
) -> np.ndarray:
    """Return normalized float32 embeddings for ``texts``.

    Deliberately seeds the RNGs before encoding so the mapped vectors are
    reproducible for the same model/inputs on the same device.
    """
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    kwargs = {"batch_size": batch_size, "show_progress_bar": True,
              "normalize_embeddings": True}
    if device:
        kwargs["device"] = device
    mat = embedder.encode(texts, **kwargs)
    return np.asarray(mat, dtype=np.float32)