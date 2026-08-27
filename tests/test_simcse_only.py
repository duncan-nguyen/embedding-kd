"""Tests for the SimCSE-only control.

The control is only worth reporting if it is exactly the contrastive term the
other objectives carry, with nothing else in it, so that is what these pin —
together with the collapse diagnostic, which is the failure this row is the most
exposed to.
"""

import math

import pytest
import torch

from src.criterions.simcse import SimCSEOnly
from src.loss import info_nce


def _views(batch: int, dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, dim, generator=generator)


def test_criterion_has_no_trainable_parameters():
    assert list(SimCSEOnly().parameters()) == []


def test_loss_is_the_shared_info_nce_term():
    view1 = _views(8, 16, seed=0)
    view2 = _views(8, 16, seed=1)

    loss, _ = SimCSEOnly(temperature=0.05)(view1, view2)
    reference, _ = info_nce(view1, view2, temperature=0.05)

    assert float(loss) == pytest.approx(float(reference), rel=1e-6)


def test_identical_views_are_already_solved():
    view = _views(6, 16, seed=2)

    loss, metrics = SimCSEOnly(temperature=0.05)(view, view.clone())

    assert float(loss) == pytest.approx(0.0, abs=1e-4)
    assert metrics["inbatch_accuracy"] == 1.0
    assert metrics["pos_cos"] == pytest.approx(1.0, abs=1e-5)


def test_a_collapsed_encoder_is_visible_in_the_metrics():
    # Every sentence mapped to the same vector: the loss is the uniform-guess
    # log(B), and the negatives sit at the positives' cosine.
    collapsed = _views(1, 16, seed=3).expand(8, 16).contiguous()

    loss, metrics = SimCSEOnly(temperature=0.05)(collapsed, collapsed.clone())

    assert float(loss) == pytest.approx(math.log(8), rel=1e-5)
    assert metrics["neg_cos"] == pytest.approx(metrics["pos_cos"], abs=1e-5)


def test_gradients_reach_both_views():
    view1 = _views(6, 16, seed=4).requires_grad_(True)
    view2 = _views(6, 16, seed=5).requires_grad_(True)

    loss, _ = SimCSEOnly()(view1, view2)
    loss.backward()

    assert view1.grad.abs().sum() > 0
    assert view2.grad.abs().sum() > 0


def test_mismatched_views_are_rejected():
    with pytest.raises(ValueError, match="same shape"):
        SimCSEOnly()(_views(6, 16, seed=6), _views(6, 8, seed=7))


def test_temperature_must_be_positive():
    with pytest.raises(ValueError, match="temperature must be positive"):
        SimCSEOnly(temperature=0.0)
