import unittest

from ml.dedup import deduplicate_pairs


class TestDeduplicatePairs(unittest.TestCase):
    def make(self, content, kind, received="2024-01-01T08:00"):
        return {
            "uid": "X-1",
            "receivedDateTime": received,
            "kind": kind,
            "content": content,
        }

    def test_exact_duplicates_removed_keeping_first(self):
        records = [
            self.make("Скарга на яму на вул. Шевченка", "Теплопостачання"),
            self.make("Скарга на яму на вул. Шевченка", "Теплопостачання"),
            self.make("Інша скарга", "Теплопостачання"),
        ]
        out = deduplicate_pairs(records)
        self.assertEqual(len(out), 2)
        self.assertEqual([r["content"] for r in out],
                         ["Скарга на яму на вул. Шевченка", "Інша скарга"])

    def test_same_content_different_kind_not_deduped(self):
        records = [
            self.make("Текст скарги", "Теплопостачання"),
            self.make("Текст скарги", "Водопостачання"),
        ]
        out = deduplicate_pairs(records)
        self.assertEqual(len(out), 2)

    def test_whitespace_variants_not_deduped(self):
        records = [
            self.make("Скарга на  яму", "Теплопостачання"),
            self.make("Скарга на яму", "Теплопостачання"),
        ]
        out = deduplicate_pairs(records)
        self.assertEqual(len(out), 2)

    def test_empty_input(self):
        self.assertEqual(deduplicate_pairs([]), [])

    def test_preserves_orders(self):
        a = self.make("Перша", "A", received="2023-01-01T00:00")
        b = self.make("Друга", "B", received="2023-01-02T00:00")
        c = self.make("Перша", "A", received="2023-05-01T00:00")
        out = deduplicate_pairs([a, b, c])
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["content"], "Перша")
        self.assertEqual(out[1]["content"], "Друга")


if __name__ == "__main__":
    unittest.main()