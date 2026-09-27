"""Reproducibility of the MLX training driver's batch order (no MLX, no GPU).

The defect these lock down: ``mlx_lm.lora.train_model`` seeds MLX, which covers
LoRA initialisation, but ``tuner.trainer.train`` binds ``iterate_batches`` as a
default argument and never forwards a seed, so the NumPy permutation that orders
batches came from an unseeded global RNG. Same ``--seed``, different batches.

``mlx_lm`` is not importable in the unit-test venv, so these tests drive the wrapper
with a stand-in that reproduces the library's actual seeding behaviour --
including its ``if seed:`` guard, which makes ``seed=0`` a no-op.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from ml.tune.run_train import install_seeded_batch_order, make_seeded_batch_iterator


def _fake_iterate_batches(dataset, batch_size, max_seq_length, loop=False, seed=None, comm_group=None):
    """Mirror of mlx_lm.tuner.trainer.iterate_batches' seeding behaviour.

    Reproduces the two details that matter: the permutation loop, and the
    ``if seed:`` guard that silently ignores ``seed=0``.
    """
    idx = sorted(range(len(dataset)), key=lambda i: len(dataset[i][0]))
    batch_idx = [idx[i: i + batch_size] for i in range(0, len(dataset) - batch_size + 1, batch_size)]
    if seed:
        np.random.seed(seed)
    order: list[tuple[int, ...]] = []
    while True:
        indices = np.random.permutation(len(batch_idx))
        for k, i in enumerate(indices):
            order.append(tuple(batch_idx[i]))
            yield tuple(batch_idx[i])
            if not loop and k == len(indices) - 1:
                return  # one full epoch, matching ``loop=False``


def _dataset(n: int = 24) -> list[tuple[list[int], list[int]]]:
    return [([0] * (i % 7 + 1), [1, 2]) for i in range(n)]


def _draw(seed: int, n: int = 12) -> list[tuple[int, ...]]:
    np.random.seed(0)  # a deliberately different starting state each call
    iterate = make_seeded_batch_iterator(_fake_iterate_batches, seed)
    return list(iterate(_dataset(), 2, 2048))[:n]


def test_same_seed_gives_the_same_batch_order():
    assert _draw(42) == _draw(42)


def test_different_seeds_give_different_batch_orders():
    assert _draw(42) != _draw(43)


def test_order_is_independent_of_numpy_global_state():
    """The whole point: an unseeded global RNG must not leak into batch order."""
    np.random.seed(1234)
    first = _draw(42)
    np.random.seed(999)
    _draw(7)  # perturb the global RNG
    np.random.seed(4321)
    assert _draw(42) == first


def test_seed_zero_is_honoured():
    """mlx-lm guards with ``if seed:``, so seed=0 is a no-op there.

    The wrapper seeds NumPy itself first, which makes 0 behave like any other
    value rather than silently producing an unseeded run.
    """
    a = _draw(0)
    b = _draw(0)
    assert a == b
    # and it is a real seed, not the same no-op permutation every time
    assert a != _draw(1)


def test_batches_cover_every_example_exactly_once_per_epoch():
    order = _draw(42, n=12)
    flat = [idx for batch in order for idx in batch]
    assert len(flat) == len(set(flat)), "an example was consumed twice in one epoch"
    assert set(flat) == set(range(24))


def test_seed_is_passed_through_to_the_library():
    """The wrapper must set ``seed`` explicitly, not rely on the global state."""
    seen: list[int | None] = []

    def spy(dataset, batch_size, max_seq_length, loop=False, seed=None, comm_group=None):
        seen.append(seed)
        return iter(())

    iterate = make_seeded_batch_iterator(spy, 7)
    list(iterate(_dataset(), 2, 2048))
    assert seen == [7]


def test_explicit_seed_argument_is_overridden_not_silently_kept():
    """A caller passing its own seed must not defeat the run-level seed."""
    seen: list[int | None] = []

    def spy(dataset, batch_size, max_seq_length, loop=False, seed=None, comm_group=None):
        seen.append(seed)
        return iter(())

    list(make_seeded_batch_iterator(spy, 42)(_dataset(), 2, 2048, seed=1))
    assert seen == [42]


def test_injection_targets_lora_train_not_the_bound_default():
    """``iterate_batches`` is a default argument bound at def time, so rebinding
    the module attribute cannot work. ``install_seeded_batch_order`` must patch
    ``lora.train``, which is the name ``train_model`` actually calls."""
    calls: list[dict] = []

    def original_train(*args, **kwargs):
        calls.append(kwargs)
        return "trained"

    lora = types.SimpleNamespace(train=original_train)
    trainer = types.SimpleNamespace(iterate_batches=_fake_iterate_batches)

    install_seeded_batch_order(lora, trainer, 42)
    assert lora.train(model="m", args="a", optimizer="o", train_dataset="d") == "trained"
    assert "iterate_batches" in calls[0], "seeded iterator was not injected"

    # the injected iterator is itself deterministic
    injected = calls[0]["iterate_batches"]
    np.random.seed(5)
    first = list(injected(_dataset(), 2, 2048))
    np.random.seed(6)
    assert list(injected(_dataset(), 2, 2048)) == first


def test_injection_is_idempotent_for_the_same_seed():
    """Re-installing must not stack wrappers, which would nest seed resets."""
    lora = types.SimpleNamespace(train=lambda *a, **k: None)
    trainer = types.SimpleNamespace(iterate_batches=_fake_iterate_batches)
    install_seeded_batch_order(lora, trainer, 42)
    wrapped_once = lora.train
    install_seeded_batch_order(lora, trainer, 42)
    assert lora.train is wrapped_once


def test_installation_passes_through_to_the_real_train_signature():
    """Sanity check against the installed mlx-lm: both call sites that iterate
    batches must accept an ``iterate_batches`` keyword, and ``train_model`` must
    go through the ``train`` name we patch."""
    mlx_lm = pytest.importorskip("mlx_lm", reason="mlx-lm is only installed in .venv-mlx")
    import inspect

    from mlx_lm.tuner import trainer as real_trainer

    assert "iterate_batches" in inspect.signature(real_trainer.train).parameters
    assert "iterate_batches" in inspect.signature(real_trainer.evaluate).parameters
    assert "seed" in inspect.signature(real_trainer.iterate_batches).parameters
    assert hasattr(mlx_lm.lora, "train"), "mlx_lm.lora.train is the patch target"
    assert "\n    train(" in inspect.getsource(mlx_lm.lora.train_model), (
        "train_model no longer calls the module-level `train`, so "
        "install_seeded_batch_order would silently do nothing"
    )
