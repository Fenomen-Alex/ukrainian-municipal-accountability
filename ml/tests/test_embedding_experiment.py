import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from ml.embedding import experiment as expmod
from ml.embedding import labels as el


class FakeEmbedder:
    """Deterministic embedder: random-ish but fixed given the same texts."""

    def __init__(self, dim=8, seed=0):
        self.dim = dim
        self.seed = seed
        self.encoded_sizes = []

    def encode(self, texts, batch_size=None, show_progress_bar=False,
               normalize_embeddings=False, convert_to_numpy=True):
        self.encoded_sizes.append(len(texts))
        rng = np.random.default_rng(self.seed)
        base = rng.normal(size=(len(texts), self.dim)).astype(np.float32)
        tweak = (np.arange(len(texts)) % 3)[:, None]
        return base + tweak


def build_dataset(tmpdir):
    data_dir = os.path.join(tmpdir, "data")
    os.makedirs(data_dir)
    labels = {"А": 0, "Б": 1}
    train = [{"uid": f"t{i}", "kind": ("А" if i % 2 == 0 else "Б"),
              "content": f"скарга про {('дорогу' if i % 2 == 0 else 'воду')} {i}"}
             for i in range(8)]
    val = [{"uid": f"v{i}", "kind": ("А" if i % 2 == 0 else "Б"),
            "content": f"скарга про {('дорогу' if i % 2 == 0 else 'воду')} val{i}"}
           for i in range(4)]
    test = [{"uid": f"s{i}", "kind": ("А" if i % 2 == 0 else "Б"),
             "content": f"скарга про {('дорогу' if i % 2 == 0 else 'воду')} test{i}"}
            for i in range(4)]
    for name, records in (("train", train), ("validation", val), ("test", test)):
        with open(os.path.join(data_dir, f"{name}.jsonl"), "w",
                  encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(data_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f)
    return data_dir


class TestExperimentEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_dir = build_dataset(self.tmp.name)
        self.cache_dir = os.path.join(self.tmp.name, "embeddings")

    def _run(self, **kw):
        return expmod.run_experiment(
            data_dir=self.data_dir,
            model_id="fake/embedder",
            cache_dir=self.cache_dir,
            device="cpu",
            seed=0,
            embedder_factory=lambda device: FakeEmbedder(seed=0),
            out_json=os.path.join(self.tmp.name, "embedding_baseline.json"),
            out_md=os.path.join(self.tmp.name, "embedding_baseline.md"),
            **kw,
        )

    def test_runs_end_to_end_and_writes_outputs(self):
        report = self._run()
        self.assertEqual(report["n_train"], 8)
        self.assertEqual(report["n_validation"], 4)
        self.assertEqual(report["n_test"], 4)
        self.assertTrue(os.path.exists(
            os.path.join(self.tmp.name, "embedding_baseline.json")))
        self.assertTrue(os.path.exists(
            os.path.join(self.tmp.name, "embedding_baseline.md")))
        self.assertIn("test_metrics", report)
        self.assertIn("hyperparameters", report)
        self.assertIn("vs_baseline", report)

    def test_cache_written_per_split(self):
        self._run()
        for split in ("train", "validation", "test"):
            self.assertTrue(os.path.exists(
                os.path.join(self.cache_dir, f"{split}.npy")),
                f"missing {split}.npy")
            self.assertTrue(os.path.exists(
                os.path.join(self.cache_dir, f"{split}.meta.json")))

    def test_second_run_deterministic(self):
        a = self._run()
        b = self._run()
        self.assertEqual(
            json.dumps(a, ensure_ascii=False, sort_keys=True),
            json.dumps(b, ensure_ascii=False, sort_keys=True),
        )

    def test_embeddings_separate_per_split(self):
        report = self._run()
        files = {}
        for split in ("train", "validation", "test"):
            with open(os.path.join(self.cache_dir, f"{split}.meta.json"),
                      encoding="utf-8") as f:
                files[split] = json.load(f)
        self.assertNotEqual(
            files["train"]["fingerprint"], files["test"]["fingerprint"])
        self.assertEqual(files["train"]["split"], "train")
        self.assertEqual(files["test"]["split"], "test")

    def test_hyperparameter_selection_uses_validation_not_test(self):
        # patch the experiment's own selection delegate to spy on the rows used
        with mock.patch(
            "ml.embedding.experiment.select_hyperparameter_impl",
            wraps=expmod.select_hyperparameter_impl,
        ) as sel:
            self._run()
            x_train, y_train, x_val, y_val = sel.call_args[0][:4]
            self.assertIsInstance(x_train, np.ndarray)
            self.assertEqual(x_train.shape[0], 8)
            self.assertEqual(len(y_train), 8)
            self.assertEqual(x_val.shape[0], 4)
            self.assertEqual(len(y_val), 4)

    def test_classifier_fit_uses_train_only(self):
        with mock.patch(
            "ml.embedding.experiment.fit_weighted_lr_impl",
            wraps=expmod.fit_weighted_lr_impl,
        ) as fit:
            self._run()
            x_fit, y_fit = fit.call_args[0][:2]
            self.assertEqual(x_fit.shape[0], 8)
            self.assertEqual(len(y_fit), 8)

    def test_report_contains_baseline_comparison(self):
        report = self._run()
        comp = report["vs_baseline"]
        self.assertIn("accuracy_delta", comp)
        self.assertIn("macro_f1_delta", comp)
        self.assertIn("weighted_f1_delta", comp)
        self.assertIn("macro_f1_improved", comp)
        for key in ("model", "device", "seed", "n_train"):
            self.assertIn(key, report)

    def test_label_order_in_report_matches_saved(self):
        report = self._run()
        self.assertEqual(report["labels"], el.label_order(
            {"А": 0, "Б": 1}))


if __name__ == "__main__":
    unittest.main()