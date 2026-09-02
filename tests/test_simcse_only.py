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


def test_criterion_has_no_trainable_parameters_without_the_head():
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


# -- Gao et al.'s projection head ------------------------------------------------


def test_the_head_is_the_criterion_s_only_parameter():
    criterion = SimCSEOnly(hidden_size=16, mlp_head=True)

    shapes = {name: tuple(p.shape) for name, p in criterion.named_parameters()}

    # Linear(d, d) + Tanh: a square weight and its bias, and nothing else. Square
    # is what makes the head droppable -- the encoder's own width is what the
    # benchmarks read.
    assert shapes == {"mlp.0.weight": (16, 16), "mlp.0.bias": (16,)}


def test_the_head_sits_between_the_encoder_and_the_loss():
    criterion = SimCSEOnly(temperature=0.05, hidden_size=16, mlp_head=True)
    view1 = _views(8, 16, seed=8)
    view2 = _views(8, 16, seed=9)

    with torch.no_grad():
        loss, _ = criterion(view1, view2)
        reference, _ = info_nce(
            criterion.mlp(view1), criterion.mlp(view2), temperature=0.05
        )

    assert float(loss) == pytest.approx(float(reference), rel=1e-6)


def test_the_head_is_trained_by_the_loss():
    criterion = SimCSEOnly(hidden_size=16, mlp_head=True)

    loss, _ = criterion(_views(6, 16, seed=10), _views(6, 16, seed=11))
    loss.backward()

    assert criterion.mlp[0].weight.grad.abs().sum() > 0


def test_the_head_stays_out_of_the_encoder_weights():
    # It is saved under the criterion, which save_checkpoint writes to
    # criterion_state_dict -- never into model_state_dict, which is what the
    # benchmarks load.
    criterion = SimCSEOnly(hidden_size=16, mlp_head=True)

    assert set(criterion.state_dict()) == {"mlp.0.weight", "mlp.0.bias"}


def test_the_head_needs_a_width():
    with pytest.raises(ValueError, match="positive hidden_size"):
        SimCSEOnly(mlp_head=True)
    with pytest.raises(ValueError, match="positive hidden_size"):
        SimCSEOnly(hidden_size=0, mlp_head=True)


def test_a_width_without_the_head_builds_nothing():
    criterion = SimCSEOnly(hidden_size=16)

    assert criterion.mlp is None
    assert list(criterion.parameters()) == []
