import unittest

from ml.split import (
    parse_datetime,
    chronological_split,
    compute_normalized_text,
    find_cross_split_text_leakage,
    ensure_no_text_leakage,
)


def make(content, received, kind="Теплопостачання"):
    return {
        "uid": "X",
        "receivedDateTime": received,
        "kind": kind,
        "content": content,
    }


class TestParseDatetime(unittest.TestCase):
    def test_basic_format(self):
        self.assertEqual(parse_datetime("2023-02-10T08:13"),
                         (2023, 2, 10, 8, 13))

    def test_invalid_returns_none(self):
        self.assertIsNone(parse_datetime("nope"))
        self.assertIsNone(parse_datetime(""))
        self.assertIsNone(parse_datetime(None))


class TestChronologicalSplit(unittest.TestCase):
    def test_boundaries_inclusive(self):
        records = [
            make("до", "2025-12-31T23:59", "A"),
            make("рівно січень", "2026-01-01T00:00", "B"),
            make("середина", "2026-03-15T12:00", "C"),
            make("рівно липень", "2026-07-01T00:00", "D"),
            make("після", "2026-08-31T15:00", "E"),
        ]
        train, val, test = chronological_split(
            records, train_start=(2026, 1, 1), val_start=(2026, 7, 1)
        )
        self.assertEqual([r["content"] for r in train], ["до"])
        self.assertEqual([r["content"] for r in val],
                         ["рівно січень", "середина"])
        self.assertEqual([r["content"] for r in test], ["рівно липень", "після"])

    def test_partition_sizes_sum(self):
        records = [make(f"r{i}", f"2025-01-{i + 1:02d}T00:00") for i in range(30)]
        train, val, test = chronological_split(
            records, train_start=(2026, 1, 1), val_start=(2026, 7, 1)
        )
        self.assertEqual(len(train) + len(val) + len(test), 30)

    def test_unparseable_date_raises(self):
        records = [make("a", "bad-date")]
        with self.assertRaises(ValueError):
            chronological_split(
                records, train_start=(2026, 1, 1), val_start=(2026, 7, 1)
            )

    def test_deterministic_preserves_order(self):
        records = [make(f"r{i}", f"2025-06-{i + 1:02d}T00:00") for i in range(10)]
        t1, v1, s1 = chronological_split(records, (2026, 1, 1), (2026, 7, 1))
        t2, v2, s2 = chronological_split(records, (2026, 1, 1), (2026, 7, 1))
        self.assertEqual([r["content"] for r in t1], [r["content"] for r in t2])
        self.assertEqual([r["content"] for r in v1], [r["content"] for r in v2])
        self.assertEqual([r["content"] for r in s1], [r["content"] for r in s2])


class TestTextNormalization(unittest.TestCase):
    def test_lower_case_and_whitespace_collapse(self):
        self.assertEqual(compute_normalized_text("Скарга  на\nЯМУ"),
                         "скарга на яму")


class TestLeakage(unittest.TestCase):
    def test_no_leakage_when_disjoint(self):
        train = [make("текст один", "2025-01-01T00:00")]
        val = [make("текст два", "2026-02-01T00:00")]
        test = [make("текст три", "2026-08-01T00:00")]
        leaks = find_cross_split_text_leakage(train, val, test)
        self.assertEqual(leaks, {})

    def test_detects_identical_text_across_splits(self):
        train = [make("спільний текст", "2025-01-01T00:00")]
        val = [make("спільний текст", "2026-02-01T00:00")]
        test = [make("інший", "2026-08-01T00:00")]
        leaks = find_cross_split_text_leakage(train, val, test)
        self.assertIn("спільний текст", leaks)
        self.assertEqual(set(leaks["спільний текст"]), {"train", "val"})

    def test_leakage_normalization_matches_case_and_space(self):
        train = [make("Скарга  на яму", "2025-01-01T00:00")]
        test = [make("скарга на яму", "2026-08-01T00:00")]
        leaks = find_cross_split_text_leakage(train, [], test)
        self.assertIn("скарга на яму", leaks)

    def test_ensure_no_leakage_raises_on_conflict(self):
        train = [make("спільний", "2025-01-01T00:00")]
        test = [make("спільний", "2026-08-01T00:00")]
        with self.assertRaises(ValueError):
            ensure_no_text_leakage(train, [], test)

    def test_ensure_no_leakage_passes_when_clean(self):
        train = [make("а", "2025-01-01T00:00")]
        val = [make("б", "2026-02-01T00:00")]
        test = [make("в", "2026-08-01T00:00")]
        ensure_no_text_leakage(train, val, test)  # should not raise


if __name__ == "__main__":
    unittest.main()