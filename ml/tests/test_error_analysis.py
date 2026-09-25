import json
import os
import tempfile
import unittest

from ml import error_analysis as ea

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class TestLoaders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_load_records_returns_list_of_dicts(self):
        path = os.path.join(self.tmp.name, "r.jsonl")
        write_jsonl(path, [{"uid": "a", "kind": "K", "content": "текст"}])
        records = ea.load_records(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["content"], "текст")

    def test_load_label_map_reads_json(self):
        path = os.path.join(self.tmp.name, "labels.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"A": 0, "B": 1}, f)
        self.assertEqual(ea.load_label_map(path), {"A": 0, "B": 1})

    def test_label_order_sorts_by_map_value(self):
        self.assertEqual(ea.label_order({"B": 1, "A": 0}), ["A", "B"])


class TestConfusionAndMetrics(unittest.TestCase):
    def test_confusion_matrix_pair_counts(self):
        predictions = [
            {"true_kind": "A", "predicted_kind": "A"},
            {"true_kind": "A", "predicted_kind": "B"},
            {"true_kind": "B", "predicted_kind": "B"},
        ]
        cm = ea.confusion_matrix_from_predictions(predictions, ["A", "B"])
        self.assertEqual(cm.tolist(), [[1, 1], [0, 1]])

    def test_per_class_metrics_computes_precision_recall_f1(self):
        # A: 2 TP, 1 FP, 1 FN ; B: 1 TP, 0 FP, 1 FN
        predictions = [
            {"true_kind": "A", "predicted_kind": "A"},
            {"true_kind": "A", "predicted_kind": "A"},
            {"true_kind": "A", "predicted_kind": "B"},
            {"true_kind": "B", "predicted_kind": "A"},
            {"true_kind": "B", "predicted_kind": "B"},
        ]
        metrics = ea.per_class_metrics(predictions, ["A", "B"])
        by_kind = {m["kind"]: m for m in metrics}
        self.assertEqual(by_kind["A"]["support"], 3)
        self.assertAlmostEqual(by_kind["A"]["precision"], 2 / 3, places=4)
        self.assertAlmostEqual(by_kind["A"]["recall"], 2 / 3, places=4)
        self.assertAlmostEqual(by_kind["A"]["f1"], 2 / 3, places=4)
        self.assertEqual(by_kind["B"]["support"], 2)
        self.assertAlmostEqual(by_kind["B"]["precision"], 0.5, places=4)
        self.assertAlmostEqual(by_kind["B"]["recall"], 0.5, places=4)
        self.assertAlmostEqual(by_kind["B"]["f1"], 0.5, places=4)

    def test_per_class_metrics_ordered_by_f1_desc(self):
        predictions = [
            {"true_kind": "A", "predicted_kind": "A"},
            {"true_kind": "B", "predicted_kind": "B"},
            {"true_kind": "B", "predicted_kind": "A"},
        ]
        metrics = ea.per_class_metrics(predictions, ["A", "B"])
        f1s = [m["f1"] for m in metrics]
        self.assertEqual(f1s, sorted(f1s, reverse=True))

    def test_most_common_confusions_counts_and_orders(self):
        predictions = [
            {"true_kind": "A", "predicted_kind": "B"},
            {"true_kind": "A", "predicted_kind": "B"},
            {"true_kind": "A", "predicted_kind": "C"},
            {"true_kind": "B", "predicted_kind": "A"},
            {"true_kind": "B", "predicted_kind": "B"},
        ]
        confusions = ea.most_common_confusions(predictions)
        self.assertEqual(confusions[0], {"true": "A", "predicted": "B", "count": 2})
        self.assertEqual(sum(c["count"] for c in confusions), 4)


class TestMisclassifiedSelection(unittest.TestCase):
    def _predictions(self):
        preds = []
        for i in range(10):
            preds.append({
                "uid": f"e{i}",
                "content": f"хибний текст {i}",
                "true_kind": "A",
                "predicted_kind": "B",
                "confidence": 0.01 * i,
            })
        preds.append({
            "uid": "ok1", "content": "правильний", "true_kind": "A",
            "predicted_kind": "A", "confidence": 0.9,
        })
        preds.append({
            "uid": "ok2", "content": "правильний 2", "true_kind": "B",
            "predicted_kind": "B", "confidence": 0.8,
        })
        return preds

    def test_misclassified_by_label_returns_examples_for_all_labels(self):
        preds = self._predictions()
        out = ea.misclassified_by_label(preds, ["A", "B"], max_examples=5)
        self.assertEqual(set(out), {"A", "B"})
        self.assertEqual(len(out["A"]), 5)
        self.assertEqual(out["B"], [])

    def test_misclassified_by_label_prefers_highest_confidence(self):
        preds = self._predictions()
        out = ea.misclassified_by_label(preds, ["A"], max_examples=3)
        confidences = [e["confidence"] for e in out["A"]]
        self.assertEqual(confidences, sorted(confidences, reverse=True))

    def test_misclassified_example_fields(self):
        preds = self._predictions()
        out = ea.misclassified_by_label(preds, ["A"], max_examples=1)
        example = out["A"][0]
        self.assertEqual(
            set(example),
            {"uid", "content", "true_kind", "predicted_kind", "confidence"},
        )

    def test_top_confidence_errors_limited_and_sorted(self):
        preds = self._predictions()
        top = ea.top_confidence_errors(preds, n=3)
        self.assertEqual(len(top), 3)
        confidences = [e["confidence"] for e in top]
        self.assertEqual(confidences, sorted(confidences, reverse=True))


class TestDuplicatePatterns(unittest.TestCase):
    def _preds(self):
        return [
            {"uid": "d1", "content": "Скарга на яму на дорозі", "true_kind": "A",
             "predicted_kind": "B", "confidence": 0.5},
            {"uid": "d2", "content": "Скарга на яму на дорозі", "true_kind": "A",
             "predicted_kind": "B", "confidence": 0.6},
            {"uid": "d3", "content": "скарга на яму  на дорозі   ", "true_kind": "A",
             "predicted_kind": "B", "confidence": 0.7},
            {"uid": "n1", "content": "Зовсім інший текст скарги про світло",
             "true_kind": "B", "predicted_kind": "A", "confidence": 0.4},
            {"uid": "n2", "content": "Зовсім інший текст скарги про воду",
             "true_kind": "B", "predicted_kind": "A", "confidence": 0.3},
        ]

    def test_finds_exact_duplicate_group(self):
        out = ea.find_error_duplicates(self._preds(), near_threshold=0.0)
        self.assertEqual(out["exact_duplicate_groups"][0]["count"], 2)
        self.assertIn("Скарга на яму на дорозі",
                      out["exact_duplicate_groups"][0]["content"])

    def test_normalized_duplicates_grouped_separately(self):
        out = ea.find_error_duplicates(self._preds(), near_threshold=0.0)
        self.assertEqual(len(out["normalized_duplicate_groups"]), 1)
        self.assertEqual(out["normalized_duplicate_groups"][0]["count"], 3)

    def test_exact_pair_is_not_double_counted_in_normalized(self):
        preds = [
            {"uid": "x", "content": "однаковий текст", "true_kind": "A",
             "predicted_kind": "B", "confidence": 0.5},
            {"uid": "y", "content": "однаковий текст", "true_kind": "A",
             "predicted_kind": "B", "confidence": 0.5},
        ]
        out = ea.find_error_duplicates(preds, near_threshold=0.9)
        self.assertEqual(len(out["exact_duplicate_groups"]), 1)
        self.assertEqual(len(out["normalized_duplicate_groups"]), 1)


class TestAnalyzeEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_dir = os.path.join(self.tmp.name, "data")
        os.makedirs(self.data_dir)
        # tiny reproducible dataset: two classes, several records each
        train = []
        for i in range(6):
            train.append({"uid": f"t{i}", "kind": "A",
                          "content": f"скарга про дорогу номер індекс {i}"})
        for i in range(6):
            train.append({"uid": f"u{i}", "kind": "B",
                          "content": f"скарга про воду номер індекс {i}"})
        test = [
            {"uid": "v1", "kind": "A", "content": "скарга про дорогу номер 1"},
            {"uid": "v2", "kind": "B", "content": "скарга про воду номер 1"},
        ]
        write_jsonl(os.path.join(self.data_dir, "train.jsonl"), train)
        write_jsonl(os.path.join(self.data_dir, "test.jsonl"), test)
        with open(os.path.join(self.data_dir, "labels.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"A": 0, "B": 1}, f)

    def test_analyze_writes_json_and_markdown_report(self):
        report = ea.analyze(
            data_dir=self.data_dir,
            out_json=os.path.join(self.data_dir, "error_analysis.json"),
            out_md=os.path.join(self.data_dir, "error_analysis.md"),
            seed=0,
        )
        self.assertTrue(os.path.exists(
            os.path.join(self.data_dir, "error_analysis.json")))
        self.assertTrue(os.path.exists(
            os.path.join(self.data_dir, "error_analysis.md")))
        self.assertIn("confusion_matrix", report)
        self.assertIn("per_class", report)
        self.assertIn("most_common_confusions", report)
        self.assertIn("misclassified_by_label", report)
        self.assertIn("top_confidence_errors", report)
        self.assertEqual(report["n_test"], 2)

    def test_fit_and_predict_deterministic(self):
        a, _ = ea.fit_and_predict(
            os.path.join(self.data_dir, "train.jsonl"),
            os.path.join(self.data_dir, "test.jsonl"),
            os.path.join(self.data_dir, "labels.json"),
            seed=0,
        )
        b, _ = ea.fit_and_predict(
            os.path.join(self.data_dir, "train.jsonl"),
            os.path.join(self.data_dir, "test.jsonl"),
            os.path.join(self.data_dir, "labels.json"),
            seed=0,
        )
        self.assertEqual([(p["uid"], p["predicted_kind"], p["confidence"])
                          for p in a],
                         [(p["uid"], p["predicted_kind"], p["confidence"])
                          for p in b])


class TestMatchesSavedBaseline(unittest.TestCase):
    def test_reproduction_matches_baseline_json_confusion(self):
        if not os.path.exists(os.path.join(DATA_DIR, "baseline.json")):
            self.skipTest("ml/data/baseline.json absent")
        with open(os.path.join(DATA_DIR, "baseline.json"), encoding="utf-8") as f:
            saved = json.load(f)
        preds, labels = ea.fit_and_predict(
            os.path.join(DATA_DIR, "train.jsonl"),
            os.path.join(DATA_DIR, "test.jsonl"),
            os.path.join(DATA_DIR, "labels.json"),
            seed=0,
        )
        self.assertEqual(labels, saved["labels"])
        cm = ea.confusion_matrix_from_predictions(preds, labels)
        self.assertEqual(cm.tolist(), saved["confusion_matrix"])
        total = int(cm.sum())
        correct = int(sum(cm[i, i] for i in range(len(labels))))
        self.assertAlmostEqual(correct / total, saved["accuracy"], places=4)


if __name__ == "__main__":
    unittest.main()