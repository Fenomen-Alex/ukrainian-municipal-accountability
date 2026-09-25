"""Baseline classifier: numpy-only TF-IDF + multinomial softmax (linear).

Deliberately contains zero external-ML dependencies (numpy only) so the
benchmark runs on the M1 Mac with only a stock Python.  The point of a
baseline is not state-of-the-art accuracy but a sane, reproducible floor to
compare future models against.
"""

from __future__ import annotations

import math
import os
import random
import re
from collections import Counter

import numpy as np

_TOKEN_RE = re.compile(r"[a-zа-яіїєґ']+")


def tokenize(text: str | None) -> list[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def confusion_matrix(
    y_true: list[int], y_pred: list[int], labels: list[int]
) -> np.ndarray:
    n = len(labels)
    index = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((n, n), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[index[true], index[pred]] += 1
    return cm


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    cm: np.ndarray,
    labels: list[int],
) -> dict:
    n = len(labels)
    predicted_qty = cm.sum(axis=0)
    preds_per_class = {label: predicted_qty[i] for i, label in enumerate(labels)}
    true_qty = cm.sum(axis=1)
    per_class_support = [int(true_qty[i]) for i in range(n)]

    per_class_f1: list[float] = []
    for i, label in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(preds_per_class[label]) - tp
        fn = int(true_qty[i]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
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
        "per_class_support": per_class_support,
        "per_class_f1": [round(f1, 4) for f1 in per_class_f1],
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


class TfidfVectorizer:
    def fit(self, documents: list[str], min_df: int = 2) -> None:
        df: Counter[str] = Counter()
        for doc in documents:
            df.update(set(tokenize(doc)))
        self.vocab = sorted(
            tok for tok, count in df.items() if count >= min_df and len(tok) > 2
        )
        self.idf = {
            tok: math.log(len(documents) / (1 + count))
            for tok, count in df.items()
            if tok in self.vocab
        }
        self.index = {tok: i for i, tok in enumerate(self.vocab)}

    def transform(self, documents: list[str]) -> np.ndarray:
        rows = np.zeros((len(documents), len(self.vocab)))
        for row_i, doc in enumerate(documents):
            counts = Counter(tokenize(doc))
            for tok, count in counts.items():
                if tok in self.index:
                    tf = 1 + math.log(count)
                    rows[row_i, self.index[tok]] = tf * self.idf[tok]
        return normalize_rows(rows)


class SoftmaxClassifier:
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        num_classes: int,
        seed: int = 0,
        epochs: int = 200,
        learning_rate: float = 0.5,
    ) -> None:
        rng = np.random.default_rng(seed)
        num_docs, num_feats = x.shape
        self.weight = rng.normal(0.0, 1e-2, size=(num_feats, num_classes))
        one_hot = np.zeros((num_docs, num_classes))
        one_hot[np.arange(num_docs), y] = 1.0
        for _ in range(epochs):
            score = x @ self.weight
            score -= score.max(axis=1, keepdims=True)
            exp = np.exp(score)
            prob = exp / exp.sum(axis=1, keepdims=True)
            grad = x.T @ (prob - one_hot)
            self.weight -= learning_rate * grad / num_docs

    def predict(self, x: np.ndarray) -> list[int]:
        score = x @ self.weight
        return [int(i) for i in score.argmax(axis=1)]


def run_baseline(
    train_path: str,
    val_path: str,
    test_path: str,
    labels_path: str,
    seed: int = 0,
) -> dict:
    import json

    def load(path: str) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    train = load(train_path)
    test = load(test_path)
    with open(labels_path, encoding="utf-8") as f:
        label_map = json.load(f)
    labels = sorted(label_map, key=lambda k: label_map[k])
    labels_int = [label_map[k] for k in labels]

    # exclude OOV classes from training if the label split produced none
    label_ids = {label_map[r["kind"]] for r in train}
    labels = [k for k in labels if label_map[k] in label_ids]
    labels_int = [label_map[k] for k in labels]

    vectorizer = TfidfVectorizer()
    vectorizer.fit([r["content"] for r in train])
    x_train = vectorizer.transform([r["content"] for r in train])
    y_train = np.array([label_map[r["kind"]] for r in train])

    y_map = {old: new for new, old in enumerate(labels_int)}
    y_train = np.array([y_map[old] for old in y_train.tolist()])

    clf = SoftmaxClassifier()
    clf.fit(x_train, y_train, num_classes=len(labels), seed=seed)

    x_test = vectorizer.transform([r["content"] for r in test])
    y_test = [y_map[label_map[r["kind"]]] for r in test]
    y_pred = clf.predict(x_test)

    cm = confusion_matrix(y_test, y_pred, list(range(len(labels))))
    metrics = compute_metrics(y_test, y_pred, cm, list(range(len(labels))))
    metrics["labels"] = labels
    return metrics


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    metrics = run_baseline(
        os.path.join(args.data_dir, "train.jsonl"),
        os.path.join(args.data_dir, "validation.jsonl"),
        os.path.join(args.data_dir, "test.jsonl"),
        os.path.join(args.data_dir, "labels.json"),
        seed=args.seed,
    )
    out_path = args.out or os.path.join(args.data_dir, "baseline.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(
        {k: v for k, v in metrics.items() if k != "confusion_matrix"},
        ensure_ascii=False, indent=2, sort_keys=True,
    ))
    print("Saved to", out_path)


if __name__ == "__main__":
    main()