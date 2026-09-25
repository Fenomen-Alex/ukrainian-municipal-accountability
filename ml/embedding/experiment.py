"""Embedding-baseline experiment: run the full reproducible pipeline.

Loads the Qwen3 Ukrainian embedding model (via ``sentence-transformers``, MPS
when available), produces frozen per-split embeddings under ``ml/data/
embeddings/``, tunes a class-weighted logistic regression on the validation
set, evaluates on the test set, and compares with the TF-IDF baseline.

All numbers are reproducible: fixed seed, deterministic label mapping, cached
embedding matrices, and a deterministic classifier.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ml.embedding import cache, classifier as clfmod, evaluate as ev, generate, labels as el

DEFAULT_MODEL = "G37A/Qwen3-Embedding-0.6B-80k-squad"


def fit_weighted_lr_impl(x, y, c, seed):
    return clfmod.fit_weighted_lr(x, y, c=c, seed=seed)


def select_hyperparameter_impl(x_train, y_train, x_val, y_val, c_grid, seed):
    return clfmod.select_hyperparameter(x_train, y_train, x_val, y_val,
                                        c_grid=c_grid, seed=seed)


class _RealEmbedder:
    """Wraps the sentence-transformers model with experiment-level seeding."""

    def __init__(self, model_id: str, device: str, seed: int, batch_size: int):
        self._model = generate.load_embedder(model_id, device)
        self._device = device
        self._seed = seed
        self._batch_size = batch_size

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        return generate.embed_texts(
            self._model, texts, seed=self._seed,
            batch_size=batch_size or self._batch_size, device=self._device,
        )


def default_embedder_factory(model_id: str, device: str, seed: int,
                             batch_size: int):
    def factory(requested: str = ""):
        actual = requested or device
        return _RealEmbedder(model_id, actual, seed, batch_size)
    return factory


def _embed_split(
    records: list[dict],
    model_id: str,
    split: str,
    cache_dir: str,
    embedder,
) -> np.ndarray:
    cached = cache.load_embeddings(model_id, split, records, cache_dir)
    if cached is not None:
        return cached
    texts = [r["content"] for r in records]
    matrix = np.asarray(embedder.encode(texts), dtype=np.float32)
    cache.save_embeddings(matrix, model_id, split, records, cache_dir)
    return matrix


def _tfidf_baseline(data_dir: str) -> dict:
    """The comparison target for the embedding run.

    Prefers the saved ``baseline.json``; if absent, recomputes the deterministic
    numpy-only TF-IDF baseline from the same data files so the comparison
    always exists.
    """
    saved = ev.load_tfidf_baseline(data_dir)
    if saved is not None:
        return saved
    from ml.baseline import run_baseline

    return run_baseline(
        os.path.join(data_dir, "train.jsonl"),
        os.path.join(data_dir, "validation.jsonl"),
        os.path.join(data_dir, "test.jsonl"),
        os.path.join(data_dir, "labels.json"),
        seed=0,
    )


def run_experiment(
    data_dir: str,
    model_id: str = DEFAULT_MODEL,
    cache_dir: str | None = None,
    device: str = "auto",
    seed: int = 0,
    batch_size: int = 32,
    embedder_factory=None,
    out_json: str | None = None,
    out_md: str | None = None,
    c_grid=clfmod.DEFAULT_C_GRID,
) -> dict:
    cache_dir = cache_dir or os.path.join(data_dir, "embeddings")
    os.makedirs(cache_dir, exist_ok=True)

    label_map = el.load_label_map(os.path.join(data_dir, "labels.json"))
    labels = el.label_order(label_map)
    n_labels = len(labels)

    y = el.label_vectors_from_file(data_dir)
    records = {split: el.load_records(os.path.join(data_dir, f"{split}.jsonl"))
               for split in ("train", "validation", "test")}

    actual_device = generate.pick_device(device)
    bootstrapper = embedder_factory or default_embedder_factory(
        model_id, actual_device, seed, batch_size)
    embedder = bootstrapper(actual_device)

    splits = {}
    for split in ("train", "validation", "test"):
        splits[split] = _embed_split(
            records[split], model_id, split, cache_dir, embedder)

    best_c, c_scores = select_hyperparameter_impl(
        splits["train"], np.asarray(y["train"], dtype=int),
        splits["validation"], np.asarray(y["validation"], dtype=int),
        c_grid, seed,
    )

    clf = fit_weighted_lr_impl(
        splits["train"], np.asarray(y["train"], dtype=int), best_c, seed,
    )
    y_pred = clfmod.predict(clf, splits["test"])

    test_metrics = ev.evaluate_from_indices(y["test"], y_pred.tolist(), labels)
    tfidf_metrics = _tfidf_baseline(data_dir)
    comparison = ev.compare_to_baseline(test_metrics, tfidf_metrics)

    embedding_collapse = ev.majority_class_fraction(
        test_metrics["confusion_matrix"])
    tfidf_collapse = ev.majority_class_fraction(
        tfidf_metrics["confusion_matrix"])
    collapse = {
        "embedding": embedding_collapse,
        "tfidf": tfidf_collapse,
        "reduced": bool(
            (tfidf_collapse or {}).get("fraction", 1.0)
            > (embedding_collapse.get("fraction") or 0.0)
        ),
    }

    report = {
        "model": model_id,
        "device": actual_device,
        "seed": seed,
        "cache_dir": cache_dir,
        "labels": labels,
        "n_labels": n_labels,
        "n_train": len(records["train"]),
        "n_validation": len(records["validation"]),
        "n_test": len(records["test"]),
        "label_map": label_map,
        "hyperparameters": {
            "best_C": best_c,
            "C_scores": c_scores,
            "class_weight": "balanced",
        },
        "test_metrics": test_metrics,
        "vs_baseline": comparison,
        "majority_class_collapse": collapse,
    }

    out_json = out_json or os.path.join(data_dir, "embedding_baseline.json")
    out_md = out_md or os.path.join(data_dir, "embedding_baseline.md")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(format_markdown(report))
    return report


def format_markdown(report: dict) -> str:
    m = report["test_metrics"]
    acc = m["accuracy"]
    macro = m["macro_f1"]
    weighted = m["weighted_f1"]

    lines = [
        "# Embedding Baseline — Municipal Appeals (UK embeddings)",
        "",
        f"- Model: **`{report['model']}`**",
        f"- Device: **{report['device']}**  ·  Seed: **{report['seed']}**",
        f"- Train / validation / test rows: **{report['n_train']} / "
        f"{report['n_validation']} / {report['n_test']}**",
        f"- Accuracy: **{acc:.4f}**  ·  Macro-F1: **{macro:.4f}**  ·  "
        f"Weighted-F1: **{weighted:.4f}**",
        "",
    ]
    if report.get("vs_baseline"):
        c = report["vs_baseline"]
        lines += [
            "## Comparison with TF-IDF baseline",
            "",
            f"- Accuracy delta: **{c['accuracy_delta']:+.4f}** "
            f"({'better' if c['accuracy_improved'] else 'worse'})",
            f"- Macro-F1 delta: **{c['macro_f1_delta']:+.4f}** "
            f"({'better' if c['macro_f1_improved'] else 'worse'})",
            f"- Weighted-F1 delta: **{c['weighted_f1_delta']:+.4f}** "
            f"({'better' if c['weighted_f1_improved'] else 'worse'})",
            "",
        ]
    collapse = report.get("majority_class_collapse")
    if collapse:
        lines += [
            "## Majority-class collapse (share of test predictions on the largest-support label)",
            "",
        ]
        emb = collapse["embedding"]
        tf = collapse["tfidf"]
        emb_label = report["labels"][emb["majority_index"]] if emb["majority_index"] is not None else "?"
        tf_label = report["labels"][tf["majority_index"]] if tf and tf["majority_index"] is not None else "?"
        lines.append(f"- TF-IDF baseline: **{tf['count']}/{report['n_test']} "
                     f"({tf['fraction']:.1%})** → *{tf_label}*")
        lines.append(f"- Embeddings: **{emb['count']}/{report['n_test']} "
                     f"({emb['fraction']:.1%})** → *{emb_label}*")
        lines.append(f"- Collapse reduced: **{collapse['reduced']}**")
        lines.append("")
    lines += [
        "# Per-class metrics (test set)",
        "",
        "| # | kind | precision | recall | F1 | support |",
        "|---|------|-----------|--------|----|---------|",
    ]
    labels = report["labels"]
    for i, label in enumerate(labels):
        lines.append(
            f"| {i} | {label} | {m['per_class_precision'][i]:.4f} | "
            f"{m['per_class_recall'][i]:.4f} | {m['per_class_f1'][i]:.4f} | "
            f"{m['per_class_support'][i]} |"
        )
    lines += ["", "# Hyperparameters", ""]
    lines.append(f"- Best C: **{report['hyperparameters']['best_C']}**")
    for c, score in report["hyperparameters"]["C_scores"].items():
        lines.append(f"- C={c}: {score}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join("ml", "data"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    report = run_experiment(
        data_dir=args.data_dir,
        model_id=args.model,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    print(json.dumps({
        "model": report["model"],
        "device": report["device"],
        "best_C": report["hyperparameters"]["best_C"],
        "test_metrics": {
            k: v for k, v in report["test_metrics"].items()
            if k != "confusion_matrix"
        },
        "vs_baseline": report["vs_baseline"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()