import json
import os
import unittest

from ml.embedding import labels as el

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


class TestDeterministicLabelMapping(unittest.TestCase):
    def test_label_order_sorts_by_map_value(self):
        self.assertEqual(el.label_order({"Б": 1, "А": 0}), ["А", "Б"])

    def test_label_order_matches_saved_baseline(self):
        if not os.path.exists(os.path.join(DATA_DIR, "labels.json")):
            self.skipTest("ml/data/labels.json absent")
        with open(os.path.join(DATA_DIR, "labels.json"), encoding="utf-8") as f:
            label_map = json.load(f)
        with open(os.path.join(DATA_DIR, "baseline.json"), encoding="utf-8") as f:
            baseline = json.load(f)
        self.assertEqual(el.label_order(label_map), baseline["labels"])

    def test_indices_deterministic_across_calls(self):
        label_map = {"А": 0, "Б": 1, "В": 2}
        a = el.to_indices(["Б", "А", "В", "Б"], label_map)
        b = el.to_indices(["Б", "А", "В", "Б"], label_map)
        self.assertEqual(a, b)
        self.assertEqual(a, [1, 0, 2, 1])

    def test_unknown_kind_raises(self):
        label_map = {"А": 0}
        with self.assertRaises(KeyError):
            el.to_indices(["А", "НЕВІДОМИЙ"], label_map)

    def test_all_kinds_map_within_range(self):
        if not os.path.exists(os.path.join(DATA_DIR, "labels.json")):
            self.skipTest("ml/data/labels.json absent")
        if not os.path.exists(os.path.join(DATA_DIR, "test.jsonl")):
            self.skipTest("generated ml/data/test.jsonl absent from public checkout")
        with open(os.path.join(DATA_DIR, "labels.json"), encoding="utf-8") as f:
            label_map = json.load(f)
        with open(os.path.join(DATA_DIR, "test.jsonl"), encoding="utf-8") as f:
            kinds = [json.loads(line)["kind"] for line in f]
        idx = el.to_indices(kinds, label_map)
        self.assertEqual(len(idx), len(kinds))
        self.assertTrue(all(0 <= i < len(label_map) for i in idx))


if __name__ == "__main__":
    unittest.main()