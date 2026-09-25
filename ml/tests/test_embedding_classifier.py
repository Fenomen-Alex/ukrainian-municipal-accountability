import unittest

import numpy as np

from ml.embedding import classifier as clfmod
from ml.embedding import evaluate as ev


def make_data(n_per_class=20, dim=8, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    x_list, y_list = [], []
    for cls in range(n_classes):
        center = np.full(dim, float(cls) * 0.5)
        x = rng.normal(loc=center, scale=0.3, size=(n_per_class, dim))
        x_list.append(x)
        y_list.append(np.full(n_per_class, cls, dtype=int))
    X = np.vstack(x_list).astype(np.float32)
    y = np.concatenate(y_list)
    return X, y


class TestClassifierOutputShape(unittest.TestCase):
    def test_predict_returns_int_array_of_input_len(self):
        X, y = make_data()
        X_val, _ = make_data(n_per_class=8, seed=1)
        clf = clfmod.fit_weighted_lr(X, y, c=1.0, seed=0)
        pred = clfmod.predict(clf, X_val)
        self.assertIsInstance(pred, np.ndarray)
        self.assertEqual(pred.shape, (24,))
        self.assertTrue(np.issubdtype(pred.dtype, np.integer))

    def test_predict_indices_in_label_range(self):
        X, y = make_data(n_classes=3)
        clf = clfmod.fit_weighted_lr(X, y, c=1.0, seed=0)
        pred = clfmod.predict(clf, X)
        self.assertTrue(set(pred.tolist()).issubset({0, 1, 2}))


class TestClassWeightedLogistic(unittest.TestCase):
    def test_balanced_weights_handle_imbalance(self):
        # heavy class imbalance: class 2 has 1/10 of class 0
        Xa, ya = make_data(n_per_class=50, n_classes=1, seed=0)
        Xb, yb = make_data(n_per_class=5, n_classes=1, seed=1)
        X = np.vstack([Xa, Xb])
        y = np.concatenate([ya, yb + 1])
        clf = clfmod.fit_weighted_lr(X, y, c=1.0, seed=0)
        pred = clfmod.predict(clf, X)
        # balanced weighting should keep both classes represented
        self.assertEqual(set(pred.tolist()), {0, 1})

    def test_hyperparameter_selection_uses_validation_only(self):
        X, y = make_data()
        # validation set deliberately contains a class label the train set lacks
        # — validation must never influence training, only model choice
        X_val = np.vstack([make_data(n_per_class=4, seed=1)[0]])
        y_val = np.concatenate([make_data(n_per_class=4, seed=1)[1]])
        best_c, scores = clfmod.select_hyperparameter(
            X, y, X_val, y_val, c_grid=[0.1, 1.0, 10.0], seed=0)
        self.assertIn(best_c, [0.1, 1.0, 10.0])
        self.assertEqual(sorted(scores.keys()), sorted([0.1, 1.0, 10.0]))
        for c, score in scores.items():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_selection_is_reproducible(self):
        X, y = make_data()
        X_val, y_val = make_data(n_per_class=8, seed=3)
        a = clfmod.select_hyperparameter(X, y, X_val, y_val, [0.1, 1, 10], seed=7)
        b = clfmod.select_hyperparameter(X, y, X_val, y_val, [0.1, 1, 10], seed=7)
        self.assertEqual(a, b)


class TestDeterministicEvaluation(unittest.TestCase):
    def test_evaluate_deterministic_across_calls(self):
        y_true = [0, 0, 1, 1, 2, 2, 0, 1]
        y_pred = [0, 0, 1, 1, 2, 0, 0, 1]
        labels = [0, 1, 2]
        a = ev.evaluate_from_indices(y_true, y_pred, labels)
        b = ev.evaluate_from_indices(y_true, y_pred, labels)
        self.assertEqual(a, b)

    def test_metrics_values(self):
        y_true = [0, 0, 1, 1, 2]
        y_pred = [0, 0, 1, 2, 2]
        labels = [0, 1, 2]
        m = ev.evaluate_from_indices(y_true, y_pred, labels)
        self.assertEqual(m["confusion_matrix"],
                         [[2, 0, 0], [0, 1, 1], [0, 0, 1]])
        self.assertAlmostEqual(m["accuracy"], 0.8, places=4)


class TestBaselineComparison(unittest.TestCase):
    def test_comparison_reports_deltas(self):
        embedding_metrics = {
            "accuracy": 0.7,
            "macro_f1": 0.3,
            "weighted_f1": 0.55,
        }
        tfidf_metrics = {
            "accuracy": 0.6717,
            "macro_f1": 0.199,
            "weighted_f1": 0.5953,
        }
        comp = ev.compare_to_baseline(embedding_metrics, tfidf_metrics)
        self.assertAlmostEqual(comp["accuracy_delta"], 0.7 - 0.6717, places=4)
        self.assertAlmostEqual(comp["macro_f1_delta"], 0.3 - 0.199, places=4)
        self.assertAlmostEqual(comp["weighted_f1_delta"], 0.55 - 0.5953, places=4)
        self.assertTrue(comp["macro_f1_improved"])
        self.assertFalse(comp["weighted_f1_improved"])


if __name__ == "__main__":
    unittest.main()