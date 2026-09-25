import unittest

from ml.labels import (
    build_label_map,
    filter_min_examples,
    label_distribution,
)


def make(content, kind):
    return {
        "uid": "X",
        "receivedDateTime": "2025-01-01T00:00",
        "kind": kind,
        "content": content,
    }


class TestLabelMap(unittest.TestCase):
    def test_builds_sorted_map(self):
        records = [
            make("a", "Теплопостачання"),
            make("b", "Водопостачання"),
            make("c", "Теплопостачання"),
        ]
        mapping = build_label_map(records)
        self.assertEqual(mapping, {
            "Водопостачання": 0,
            "Теплопостачання": 1,
        })

    def test_empty_input(self):
        self.assertEqual(build_label_map([]), {})


class TestFilterMinExamples(unittest.TestCase):
    def test_keeps_only_labels_with_min_examples(self):
        records = []
        for i in range(30):
            records.append(make(f"heat{i}", "Теплопостачання"))
        for i in range(29):
            records.append(make(f"water{i}", "Водопостачання"))
        out = filter_min_examples(records, min_examples=30)
        kinds = {r["kind"] for r in out}
        self.assertEqual(kinds, {"Теплопостачання"})

    def test_default_threshold_is_30(self):
        records = [make(f"h{i}", "Теплопостачання") for i in range(30)]
        out = filter_min_examples(records)
        self.assertEqual(len(out), 30)

    def test_threshold_is_inclusive(self):
        records = [make(f"h{i}", "Теплопостачання") for i in range(30)]
        out = filter_min_examples(records, min_examples=30)
        self.assertEqual(len(out), 30)


class TestLabelDistribution(unittest.TestCase):
    def test_counts_per_label_sorted_by_count_desc(self):
        records = [
            make("a", "Теплопостачання"),
            make("b", "Водопостачання"),
            make("c", "Водопостачання"),
        ]
        dist = label_distribution(records)
        self.assertEqual(dist, [("Водопостачання", 2), ("Теплопостачання", 1)])


if __name__ == "__main__":
    unittest.main()