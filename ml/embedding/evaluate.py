"""Evaluation metrics and TF-IDF comparison for the embedding baseline.

Metrics are computed identically to ``ml.baseline.compute_metrics`` so that
``embedding_baseline.json`` and ``baseline.json`` are directly comparable
(accuracy, macro-F1, weighted-F1, confusion matrix, per-class support).
"""

from __future__ import annotations

import json
import os

import numpy as np


def evaluate_from_indices(
    y_true: list[int],
    y_pred: list[int],
    labels: list[str],
) -> dict:
    """Return a metrics dict in the same shape as ``ml/baseline.py``.

    ``labels`` are the kind names in label-index order; rows of the confusion
    matrix follow ``labels`` order.
    """
    n = len(labels)
    index = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((n, n), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[index[labels[true]], index[labels[pred]]] += 1

    predicted_qty = cm.sum(axis=0)
    true_qty = cm.sum(axis=1)
    per_class_support = [int(true_qty[i]) for i in range(n)]

    per_class_precision: list[float] = []
    per_class_recall: list[float] = []
    per_class_f1: list[float] = []
    for i in range(n):
        tp = int(cm[i, i])
        fp = int(predicted_qty[i]) - tp
        fn = int(true_qty[i]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)

    accuracy = float(np.trace(cm)) / float(cm.sum()) if cm.sum() else 0.0
    macro_f1 = float(np.mean(per_class_f1)) if n else 0.0
    weighted_f1 = float(
        sum(f1 * support for f1, support in zip(per_class_f1, per_class_support))
        / sum(per_class_support)
    ) if sum(per_class_support) else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class_precision": [round(v, 4) for v in per_class_precision],
        "per_class_recall": [round(v, 4) for v in per_class_recall],
        "per_class_f1": [round(v, 4) for v in per_class_f1],
        "per_class_support": per_class_support,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


def load_baseline(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def majority_class_fraction(
    cm_matrix: list[list[int]], majority_index: int | None = None
) -> dict:
    """Share of test *predictions* assigned to the largest support label.

    The TF-IDF baseline collapses onto ``Санітарний стан`` (index 10), so this
    is the quantitative measure of whether the embedding model still does.
    """
    n = len(cm_matrix)
    col_sums = [sum(row[j] for row in cm_matrix) for j in range(n)]
    total = sum(col_sums)
    if total == 0:
        return {"majority_index": None, "fraction": 0.0, "count": 0}
    if majority_index is None:
        row_sums = [sum(row) for row in cm_matrix]
        majority_index = max(range(n), key=lambda i: row_sums[i])
    count = col_sums[majority_index]
    return {
        "majority_index": majority_index,
        "fraction": round(count / total, 4),
        "count": int(count),
    }


def compare_to_baseline(embedding_metrics: dict, tfidf_metrics: dict) -> dict:
    """Deltas and improvement flags between this run and the TF-IDF baseline."""
    return {
        "accuracy_delta": round(
            embedding_metrics["accuracy"] - tfidf_metrics["accuracy"], 4),
        "macro_f1_delta": round(
            embedding_metrics["macro_f1"] - tfidf_metrics["macro_f1"], 4),
        "weighted_f1_delta": round(
            embedding_metrics["weighted_f1"] - tfidf_metrics["weighted_f1"], 4),
        "accuracy_improved": embedding_metrics["accuracy"] > tfidf_metrics["accuracy"],
        "macro_f1_improved": embedding_metrics["macro_f1"] > tfidf_metrics["macro_f1"],
        "weighted_f1_improved": (
            embedding_metrics["weighted_f1"] > tfidf_metrics["weighted_f1"]),
    }


def load_tfidf_baseline(data_dir: str) -> dict | None:
    path = os.path.join(data_dir, "baseline.json")
    if not os.path.exists(path):
        return None
    return load_baseline(path)