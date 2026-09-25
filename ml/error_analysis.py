"""Reproducible error analysis for the municipal-appeal baseline.

The analysis is deliberately read-only over the *existing* datasets and the
*saved* baseline``ml/data/baseline.json``.  The exact deterministic baseline
(seed 0) is re-run only to obtain per-example predictions and confidence
scores that ``baseline.json`` does not store.

Nothing here should be confused with improving the model: the goal is to
decide whether the current ``kind`` taxonomy is a usable supervised target
and what change should come next.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

import numpy as np

from ml.baseline import SoftmaxClassifier, TfidfVectorizer
from ml.cleaner import normalize_text


def load_records(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_label_map(path: str) -> dict[str, int]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def label_order(label_map: dict[str, int]) -> list[str]:
    """Labels sorted by their integer index, matching ``baseline.json``."""
    return sorted(label_map, key=lambda k: label_map[k])


def _softmax_scores(score: np.ndarray) -> np.ndarray:
    score = score - score.max(axis=1, keepdims=True)
    exp = np.exp(score)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_and_predict(
    train_path: str,
    test_path: str,
    labels_path: str,
    seed: int = 0,
) -> tuple[list[dict], list[str]]:
    """Re-run the baseline (seed 0) and return per-test predictions.

    Mirrors ``ml.baseline.run_baseline`` exactly (excluding OOV classes from
    training, remapping indices, same vectorizer/classifier) and additionally
    returns per-example softmax confidence alongside the prediction.
    """
    train = load_records(train_path)
    test = load_records(test_path)
    label_map = load_label_map(labels_path)
    labels = label_order(label_map)

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
    score = x_test @ clf.weight
    probs = _softmax_scores(score)
    y_pred = [int(i) for i in score.argmax(axis=1)]

    predictions: list[dict] = []
    for record, pred_idx in zip(test, y_pred):
        predictions.append({
            "uid": record.get("uid"),
            "content": record.get("content"),
            "true_kind": record.get("kind"),
            "predicted_kind": labels[pred_idx],
            "confidence": round(float(probs[len(predictions), pred_idx]), 4),
        })
    return predictions, labels


def confusion_matrix_from_predictions(
    predictions: list[dict], labels: list[str]
) -> np.ndarray:
    index = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for p in predictions:
        cm[index[p["true_kind"]], index[p["predicted_kind"]]] += 1
    return cm


def per_class_metrics(
    predictions: list[dict], labels: list[str]
) -> list[dict]:
    """Per-class precision/recall/F1/support, ordered by F1 descending."""
    cm = confusion_matrix_from_predictions(predictions, labels)
    predicted_qty = cm.sum(axis=0)
    true_qty = cm.sum(axis=1)
    rows = []
    for i, label in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(predicted_qty[i]) - tp
        fn = int(true_qty[i]) - tp
        support = int(true_qty[i])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        rows.append({
            "kind": label,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "tp": tp,
        })
    return sorted(rows, key=lambda r: (-r["f1"], -r["support"], r["kind"]))


def most_common_confusions(
    predictions: list[dict], top_n: int = 20
) -> list[dict]:
    """Most frequent (true -> predicted) pairs where prediction was wrong."""
    counter: Counter[tuple[str, str]] = Counter()
    for p in predictions:
        if p["true_kind"] != p["predicted_kind"]:
            counter[(p["true_kind"], p["predicted_kind"])] += 1
    return [
        {"true": true, "predicted": predicted, "count": count}
        for (true, predicted), count in counter.most_common(top_n)
    ]


def _is_error(p: dict) -> bool:
    return p["true_kind"] != p["predicted_kind"]


def misclassified_by_label(
    predictions: list[dict],
    labels: list[str],
    max_examples: int = 5,
) -> dict[str, list[dict]]:
    """For every final label, the highest-confidence misclassified examples."""
    by_label: dict[str, list[dict]] = {label: [] for label in labels}
    errors = sorted(
        (p for p in predictions if _is_error(p)),
        key=lambda p: -p["confidence"],
    )
    for p in errors:
        bucket = by_label[p["true_kind"]]
        if len(bucket) < max_examples:
            bucket.append(p)
    return by_label


def top_confidence_errors(predictions: list[dict], n: int = 20) -> list[dict]:
    errors = sorted(
        (p for p in predictions if _is_error(p)),
        key=lambda p: -p["confidence"],
    )
    return errors[:n]


def find_error_duplicates(
    predictions: list[dict],
    near_threshold: float = 0.0,
) -> dict:
    """Exact and normalized duplicate groups among misclassified texts.

    * exact groups: identical raw ``content`` (same as the pipeline's dedup key
      minus the kind dimension — here restricted to *errors*).
    * normalized groups: identical after whale-space collapse + lowercasing
      (uses ``ml.cleaner.normalize_text``).
    """
    errors = [p for p in predictions if _is_error(p)]

    exact: dict[str, list[dict]] = defaultdict(list)
    normalized: dict[str, list[dict]] = defaultdict(list)
    for p in errors:
        exact[p["content"]].append(p)
        normalized[normalize_text(p["content"])].append(p)

    def groups(mapping):
        out = []
        for text, members in mapping.items():
            if len(members) > 1:
                out.append({
                    "content": text,
                    "count": len(members),
                    "true_kind": members[0]["true_kind"],
                    "predicted_kind": members[0]["predicted_kind"],
                })
        return sorted(out, key=lambda g: -g["count"])

    return {
        "exact_duplicate_groups": groups(exact),
        "normalized_duplicate_groups": groups(normalized),
    }


def _markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def format_markdown(report: dict) -> str:
    lines: list[str] = [
        "# Error Analysis — Municipal Appeals Baseline",
        "",
        f"- Test records: **{report['n_test']}**",
        f"- Misclassified: **{report['n_errors']}** "
        f"({report['error_rate']:.1%})",
        f"- Accuracy: **{report['accuracy']:.4f}**  ·  "
        f"Macro-F1: **{report['macro_f1']:.4f}**  ·  "
        f"Weighted-F1: **{report['weighted_f1']:.4f}**",
        "",
        "## Confusion matrix (rows = true, columns = predicted)",
        "",
    ]
    labels = [c["kind"] for c in report["per_class"]]
    # label header uses the baseline order for stable column layout
    order = report["confusion_matrix"]["labels"]
    short = {label: f"{label[:38]}…" if len(label) > 40 else label
             for label in order}
    rows = []
    for i, true_label in enumerate(report["confusion_matrix"]["labels"]):
        row = [f"{short[true_label]}"]
        row += [str(v) for v in report["confusion_matrix"]["matrix"][i]]
        rows.append(row)
    lines.append(_markdown_table([""] + [str(j) for j in range(len(order))], rows))
    lines += [
        "",
        "Column index → label:",
        "",
    ]
    for j, label in enumerate(order):
        lines.append(f"{j}. {label}")
    lines += [
        "",
        "## Per-class metrics (ordered by F1)",
        "",
    ]
    lines.append(_markdown_table(
        ["kind", "precision", "recall", "F1", "support"],
        [[c["kind"], f"{c['precision']:.4f}", f"{c['recall']:.4f}",
          f"{c['f1']:.4f}", str(c["support"])] for c in report["per_class"]],
    ))
    lines += [
        "",
        "## Most common confusion pairs",
        "",
    ]
    if report["most_common_confusions"]:
        lines.append(_markdown_table(
            ["true kind", "predicted kind", "count"],
            [[c["true"], c["predicted"], str(c["count"])]
             for c in report["most_common_confusions"]],
        ))
    else:
        lines.append("_None — perfect predictions._")
    lines += [
        "",
        "## Misclassified examples by true label",
        "",
    ]
    for label, examples in report["misclassified_by_label"].items():
        lines.append(f"### {label}")
        lines.append("")
        if not examples:
            lines.append("_No misclassified test examples._")
            lines.append("")
            continue
        rows = []
        for e in examples:
            rows.append([
                str(e["confidence"]),
                e["predicted_kind"],
                e["content"],
            ])
        lines.append(_markdown_table(
            ["confidence", "predicted kind", "content"], rows))
        lines.append("")
    lines += [
        "## Top-confidence incorrect predictions",
        "",
    ]
    if report["top_confidence_errors"]:
        rows = []
        for e in report["top_confidence_errors"]:
            rows.append([
                str(e["confidence"]), e["true_kind"], e["predicted_kind"],
                e["content"],
            ])
        lines.append(_markdown_table(
            ["confidence", "true kind", "predicted kind", "content"], rows))
    else:
        lines.append("_None._")
    dup = report["duplicate_patterns"]
    lines += [
        "",
        "## Duplicate / near-duplicate patterns among errors",
        "",
        f"- Exact-duplicate groups: **{len(dup['exact_duplicate_groups'])}**",
        f"- Normalized-duplicate groups: "
        f"**{len(dup['normalized_duplicate_groups'])}**",
        "",
    ]
    if dup["exact_duplicate_groups"]:
        lines.append("### Exact duplicates")
        lines.append("")
        lines.append(_markdown_table(
            ["count", "kind", "content"],
            [[str(g["count"]), g["true_kind"], g["content"]]
             for g in dup["exact_duplicate_groups"]],
        ))
        lines.append("")
    if dup["normalized_duplicate_groups"]:
        lines.append("### Near duplicates (normalized text)")
        lines.append("")
        lines.append(_markdown_table(
            ["count", "kind", "content"],
            [[str(g["count"]), g["true_kind"], g["content"]]
             for g in dup["normalized_duplicate_groups"]],
        ))
    return "\n".join(lines) + "\n"


def analyze(
    data_dir: str,
    out_json: str,
    out_md: str,
    seed: int = 0,
) -> dict:
    """Run the full error-analysis and write ``error_analysis.json``/``.md``."""
    train_path = os.path.join(data_dir, "train.jsonl")
    test_path = os.path.join(data_dir, "test.jsonl")
    labels_path = os.path.join(data_dir, "labels.json")
    baseline_path = os.path.join(data_dir, "baseline.json")

    predictions, labels = fit_and_predict(train_path, test_path, labels_path,
                                          seed=seed)

    cm = confusion_matrix_from_predictions(predictions, labels)
    n_test = len(predictions)
    n_errors = sum(1 for p in predictions if _is_error(p))
    correct = int(sum(cm[i, i] for i in range(len(labels))))
    accuracy = correct / n_test if n_test else 0.0

    per_class = per_class_metrics(predictions, labels)
    macro_f1 = float(np.mean([c["f1"] for c in per_class])) if per_class else 0.0
    support = [c["support"] for c in per_class]
    weighted_f1 = (
        sum(c["f1"] * s for c, s in zip(per_class, support)) / sum(support)
        if sum(support) else 0.0
    )

    if os.path.exists(baseline_path):
        with open(baseline_path, encoding="utf-8") as f:
            saved = json.load(f)
        saved_cm = saved.get("confusion_matrix")
        reproduced_ok = cm.tolist() == saved_cm
    else:
        saved = None
        reproduced_ok = True

    report = {
        "n_test": n_test,
        "n_errors": n_errors,
        "error_rate": round(n_errors / n_test, 4) if n_test else 0.0,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "reproduces_saved_baseline": reproduced_ok,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
        "per_class": per_class,
        "most_common_confusions": most_common_confusions(predictions),
        "misclassified_by_label": misclassified_by_label(predictions, labels),
        "top_confidence_errors": top_confidence_errors(predictions, n=20),
        "duplicate_patterns": find_error_duplicates(predictions),
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=False)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(format_markdown(report))
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Reproducible error analysis")
    parser.add_argument("--data-dir", default=os.path.join("ml", "data"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_dir = args.data_dir
    report = analyze(
        data_dir=data_dir,
        out_json=os.path.join(data_dir, "error_analysis.json"),
        out_md=os.path.join(data_dir, "error_analysis.md"),
        seed=args.seed,
    )
    print(json.dumps({
        "n_test": report["n_test"],
        "n_errors": report["n_errors"],
        "accuracy": report["accuracy"],
        "macro_f1": report["macro_f1"],
        "weighted_f1": report["weighted_f1"],
        "reproduces_saved_baseline": report["reproduces_saved_baseline"],
        "top_confusions": report["most_common_confusions"][:10],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()