import json
import os
import tempfile
import unittest

import numpy as np

from ml.embedding import cache


def sample_records(prefix="r", n=3):
    return [{"uid": f"{prefix}-{i}", "kind": "К", "content": f"текст {i}"} for i in range(n)]


class TestCacheCorrectness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache_dir = os.path.join(self.tmp.name, "embeddings")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.model = "G37A/Qwen3-Embedding-0.6B-80k-squad"

    def _save_many(self, splits):
        for split, records in splits.items():
            matrix = np.array([[1.0, 2.0]] * len(records), dtype=np.float32)
            cache.save_embeddings(
                matrix=matrix,
                model_id=self.model,
                split=split,
                records=records,
                cache_dir=self.cache_dir,
            )
        return splits

    def test_roundtrip_preserves_values(self):
        records = sample_records()
        matrix = np.array([[1.5, -2.0], [3.0, 0.5], [0.0, 1.0]], dtype=np.float32)
        cache.save_embeddings(matrix, self.model, "train", records, self.cache_dir)
        loaded = cache.load_embeddings(self.model, "train", records, self.cache_dir)
        self.assertIsNotNone(loaded)
        np.testing.assert_array_equal(loaded, matrix)

    def test_cache_key_is_model_and_split_scoped(self):
        self._save_many({
            "train": sample_records("a"),
            "validation": sample_records("b"),
            "test": sample_records("c"),
        })
        # every split reads back its own matrix with its own records
        for split, records in {
            "train": sample_records("a"),
            "validation": sample_records("b"),
            "test": sample_records("c"),
        }.items():
            m = cache.load_embeddings(self.model, split, records, self.cache_dir)
            self.assertIsNotNone(m)

    def test_different_content_invalidates(self):
        records = sample_records()
        cache.save_embeddings(
            np.ones((3, 2), dtype=np.float32), self.model, "train",
            records, self.cache_dir)
        changed = [{**r, "content": r["content"] + " змінено"} for r in records]
        self.assertIsNone(cache.load_embeddings(
            self.model, "train", changed, self.cache_dir))

    def test_different_model_invalidates(self):
        records = sample_records()
        cache.save_embeddings(
            np.ones((3, 2), dtype=np.float32), self.model, "train",
            records, self.cache_dir)
        self.assertIsNone(cache.load_embeddings(
            "ins/mother-model", "train", records, self.cache_dir))

    def test_missing_entry_returns_none(self):
        records = sample_records()
        self.assertIsNone(cache.load_embeddings(
            self.model, "test", records, self.cache_dir))

    def test_row_count_mismatch_invalidates(self):
        records = sample_records()
        cache.save_embeddings(
            np.ones((3, 2), dtype=np.float32), self.model, "train",
            records, self.cache_dir)
        truncated = records[:2]
        self.assertIsNone(cache.load_embeddings(
            self.model, "train", truncated, self.cache_dir))

    def test_fingerprint_is_stable_across_order(self):
        # fingerprint must not depend on the file order of the records
        r1 = sample_records("x")
        r2 = list(reversed(r1))
        self.assertEqual(cache.fingerprint(r1), cache.fingerprint(r2))

    def test_meta_and_matrix_written_to_cache_dir(self):
        records = sample_records()
        matrix = np.ones((3, 4), dtype=np.float32)
        cache.save_embeddings(matrix, self.model, "train", records, self.cache_dir)
        entries = sorted(os.listdir(self.cache_dir))
        self.assertIn("train.meta.json", entries)
        npy = [e for e in entries if e.endswith(".npy")]
        self.assertEqual(len(npy), 1)
        with open(os.path.join(self.cache_dir, "train.meta.json"),
                  encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["model_id"], self.model)
        self.assertEqual(meta["dim"], 4)
        self.assertEqual(meta["n"], 3)

    def test_save_is_not_hit_by_other_split(self):
        train_r = sample_records("a")
        test_r = sample_records("c")
        cache.save_embeddings(
            np.ones((3, 2), dtype=np.float32), self.model, "train",
            train_r, self.cache_dir)
        self.assertIsNone(cache.load_embeddings(
            self.model, "test", test_r, self.cache_dir))


if __name__ == "__main__":
    unittest.main()