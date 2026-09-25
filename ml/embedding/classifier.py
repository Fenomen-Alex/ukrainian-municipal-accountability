"""Lightweight classifier on frozen embeddings.

A scikit-learn logistic regression with ``class_weight="balanced"`` so the
severely imbalanced labels are not drowned out by ``Санітарний стан``.  Only
hyperparameters are tuned, on the validation set never on the test set;
training is always deterministic.
"""

from __future__ import annotations

import numpy as np

DEFAULT_C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def fit_weighted_lr(
    x: np.ndarray,
    y: np.ndarray,
    c: float = 1.0,
    seed: int = 0,
    class_weight: str = "balanced",
    max_iter: int = 1000,
    solver: str = "lbfgs",
):
    """Fit a class-weighted multinomial logistic regression."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        C=c,
        class_weight=class_weight,
        max_iter=max_iter,
        solver=solver,
        random_state=seed,
    )
    clf.fit(np.asarray(x, dtype=np.float32), np.asarray(y, dtype=int))
    return clf


def predict(clf, x: np.ndarray) -> np.ndarray:
    """Predict integer label indices for rows of ``x`` (shape ``(n,)``)."""
    return np.asarray(clf.predict(np.asarray(x, dtype=np.float32)), dtype=int)


def cross_entropy_loss(clf, x: np.ndarray, y: np.ndarray) -> float:
    """Negative log-likelihood on a validation set — lower is better."""
    from sklearn.metrics import log_loss

    return float(log_loss(
        np.asarray(y, dtype=int),
        clf.predict_proba(np.asarray(x, dtype=np.float32)),
        labels=sorted(set(np.asarray(y, dtype=int).tolist())),
    ))


def select_hyperparameter(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    c_grid: list[float] | tuple[float, ...] = DEFAULT_C_GRID,
    seed: int = 0,
    scoring: str = "balanced",
) -> tuple[float, dict[float, float]]:
    """Pick the best ``C`` by validation macro-F1; returns ``(best_c, scores)``.

    The validation set is used solely for selection; the test set never enters
    this code path.
    """
    from sklearn.metrics import f1_score

    scores: dict[float, float] = {}
    for c in c_grid:
        clf = fit_weighted_lr(x_train, y_train, c=c, seed=seed)
        pred = predict(clf, x_val)
        if scoring == "balanced":
            score = f1_score(
                np.asarray(y_val, dtype=int), pred, average="macro", zero_division=0
            )
        else:
            score = f1_score(np.asarray(y_val, dtype=int), pred, average="weighted",
                             zero_division=0)
        scores[float(c)] = round(float(score), 6)
    best_c = max(scores, key=scores.get)  # type: ignore[arg-type]
    return float(best_c), scores


class FrozenEmbeddingPipeline:
    """Fit on train, tune on validation, evaluate on test — in that order."""

    def __init__(self, seed: int = 0, class_weight: str = "balanced"):
        self.seed = seed
        self.class_weight = class_weight
        self.clf = None
        self.best_c: float | None = None
        self.c_scores: dict[float, float] = {}

    def tune_and_fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        c_grid: list[float] | tuple[float, ...] = DEFAULT_C_GRID,
    ) -> None:
        self.best_c, self.c_scores = select_hyperparameter(
            x_train, y_train, x_val, y_val, c_grid=c_grid, seed=self.seed
        )
        self.clf = fit_weighted_lr(
            x_train, y_train, c=self.best_c, seed=self.seed,
            class_weight=self.class_weight,
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("tune_and_fit must run before predict")
        return predict(self.clf, x)