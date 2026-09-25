import unittest

import numpy as np

from ml.baseline import (
    compute_metrics,
    confusion_matrix,
    normalize_rows,
    tokenize,
)


class TestTokenize(unittest.TestCase):
    def test_lowercases_and_splits_on_non_word(self):
        toks = tokenize("Скарга на ями, води. 123!")
        self.assertIn("скарга", toks)
        self.assertIn("ями", toks)
        self.assertIn("води", toks)
        self.assertNotIn(",", toks)
        self.assertNotIn(".", toks)
        self.assertNotIn("123", toks)

    def test_empty(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize(None), [])


class TestNormalizeRows(unittest.TestCase):
    def test_rows_have_unit_norm(self):
        x = np.array([[1.0, 2.0, 0.0], [0.0, 5.0, 5.0]])
        out = normalize_rows(x)
        np.testing.assert_allclose(
            np.linalg.norm(out, axis=1), np.ones(2))
        self.assertAlmostEqual(float(out[1, 0]), 0.0)

    def test_zero_row_stays_zero(self):
        x = np.array([[0.0, 0.0]])
        out = normalize_rows(x)
        np.testing.assert_array_equal(out, np.zeros((1, 2)))


class TestConfusionMatrix(unittest.TestCase):
    def test_shape_and_counts(self):
        y_true = [0, 1, 0, 1, 2]
        y_pred = [0, 0, 0, 1, 2]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        self.assertEqual(cm.shape, (3, 3))
        self.assertEqual(cm[0, 0], 2)
        self.assertEqual(cm[1, 0], 1)
        self.assertEqual(int(cm.sum()), 5)


class TestComputeMetrics(unittest.TestCase):
    def test_perfect_prediction(self):
        y_true = [0, 0, 1, 1, 2]
        y_pred = [0, 0, 1, 1, 2]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        m = compute_metrics(y_true, y_pred, cm, labels=[0, 1, 2])
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["macro_f1"], 1.0)
        self.assertEqual(m["weighted_f1"], 1.0)
        self.assertEqual(m["per_class_support"], [2, 2, 1])

    def test_macro_versus_weighted(self):
        # two classes, class 0 dominates
        y_true = [0] * 90 + [1] * 10
        y_pred = [0] * 100
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        m = compute_metrics(y_true, y_pred, cm, labels=[0, 1])
        # macro average is penalized by class 1 (F1=0), weighted is not
        self.assertLess(m["macro_f1"], m["weighted_f1"])
        self.assertEqual(m["per_class_support"], [90, 10])
        self.assertAlmostEqual(m["accuracy"], 0.9)


if __name__ == "__main__":
    unittest.main()