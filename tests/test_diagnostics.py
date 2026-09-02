"""Tests for the batch-level instrumentation.

These are measurements, not terms of an objective, so what has to be pinned is
that each one reads what its name claims and that none of them can reach the
gradient. The properties below are the ones a reader of a training log will rely
on when a curve moves.
"""

import math

import pytest
import torch

from src.diagnostics import (
    alignment_uniformity,
    batch_spread,
    effective_rank,
    grad_norms,
    gram_agreement,
)


def test_spread_is_zero_on_a_collapsed_batch_and_one_on_an_orthogonal_one():
    collapsed = torch.randn(1, 8).expand(6, 8)
    orthogonal = torch.eye(6, 8)

    assert float(batch_spread(collapsed)) == pytest.approx(0.0, abs=1e-5)
    assert float(batch_spread(orthogonal)) == pytest.approx(1.0, abs=1e-5)


def test_spread_ignores_the_norms_it_is_given():
    """It is a statement about angles: rescaling any row must not move it."""
    batch = torch.randn(8, 16)
    scaled = batch * torch.linspace(0.1, 10.0, 8).unsqueeze(1)

    assert float(batch_spread(batch)) == pytest.approx(float(batch_spread(scaled)), abs=1e-5)


def test_effective_rank_counts_the_directions_the_batch_actually_spends():
    orthogonal = torch.eye(6, 16)
    collapsed = torch.randn(1, 16).expand(6, 16)

    assert float(effective_rank(orthogonal)) == pytest.approx(6.0, rel=1e-4)
    assert float(effective_rank(collapsed)) == pytest.approx(1.0, rel=1e-4)


def test_uniformity_reaches_its_floor_on_a_uniform_sphere():
    """-2t is the value of log E exp(-t||x - y||^2) as the cloud fills the sphere."""
    generator = torch.Generator().manual_seed(3)
    spread = torch.randn(512, 64, generator=generator)
    collapsed = torch.randn(1, 64, generator=generator).expand(512, 64)

    _, uniform_spread = alignment_uniformity(spread, spread)
    _, uniform_collapsed = alignment_uniformity(collapsed, collapsed)

    assert float(uniform_spread) < float(uniform_collapsed)
    assert float(uniform_spread) == pytest.approx(-4.0, abs=0.3)
    # A batch that has folded to a point is at zero distance from itself.
    assert float(uniform_collapsed) == pytest.approx(0.0, abs=1e-4)


def test_alignment_is_the_squared_distance_between_the_two_views():
    a = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    b = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)

    alignment, _ = alignment_uniformity(a, b)

    assert float(alignment) == pytest.approx(float(((a - b) ** 2).sum(dim=-1).mean()), rel=1e-5)
    identical, _ = alignment_uniformity(a, a)
    assert float(identical) == pytest.approx(0.0, abs=1e-6)


def test_gram_agreement_survives_a_rotation_and_a_change_of_width():
    """Rung 2 is why the student can be compared against the teacher's own d_T."""
    generator = torch.Generator().manual_seed(11)
    student = torch.randn(32, 8, generator=generator)
    rotation, _ = torch.linalg.qr(torch.randn(8, 8, generator=generator))
    teacher = torch.randn(32, 64, generator=generator)

    rmse, corr = gram_agreement(student, teacher)
    rotated_rmse, rotated_corr = gram_agreement(student @ rotation, teacher)

    assert float(rotated_rmse) == pytest.approx(float(rmse), abs=1e-5)
    assert float(rotated_corr) == pytest.approx(float(corr), abs=1e-5)
    # Against itself the two Grams coincide, whatever the width.
    exact_rmse, exact_corr = gram_agreement(student, student @ rotation)
    assert float(exact_rmse) == pytest.approx(0.0, abs=1e-5)
    assert float(exact_corr) == pytest.approx(1.0, abs=1e-5)


def test_gram_correlation_divides_out_a_uniform_contraction_that_the_rmse_keeps():
    """The gap between the two readings is what makes both worth logging."""
    generator = torch.Generator().manual_seed(12)
    teacher = torch.nn.functional.normalize(
        torch.randn(64, 16, generator=generator), dim=-1
    )
    # A student whose cloud is the teacher's, pulled towards a single direction:
    # the neighbourhood order survives, the absolute angles do not.
    contracted = torch.nn.functional.normalize(
        teacher + 1.5 * torch.ones(1, 16), dim=-1
    )

    rmse, corr = gram_agreement(contracted, teacher)

    assert float(rmse) > 0.1
    assert float(corr) > 0.9


def test_grad_norms_scale_with_the_weight_and_never_touch_the_gradient():
    node = torch.randn(4, 6, requires_grad=True)
    term = (node**2).sum()

    reported = grad_norms(node, {"term": term}, {"term": 0.25})

    assert node.grad is None
    expected = 0.25 * float((2 * node).detach().norm())
    assert float(reported["g_term"]) == pytest.approx(expected, rel=1e-5)


def test_grad_norms_report_zero_rather_than_dropping_a_term():
    """A series with holes in it is a series a plot cannot read."""
    node = torch.randn(4, 6, requires_grad=True)
    constant = torch.zeros(())

    reported = grad_norms(
        node, {"live": (node**2).sum(), "off": (node**2).sum(), "dead": constant}
    )

    assert set(reported) == {"g_live", "g_off", "g_dead"}
    assert float(reported["g_dead"]) == 0.0
    assert float(reported["g_live"]) > 0.0


def test_grad_norms_leave_the_graph_intact_for_the_real_backward():
    node = torch.randn(4, 6, requires_grad=True)
    total = (node**2).sum()

    grad_norms(node, {"total": total})
    total.backward()

    assert torch.allclose(node.grad, 2 * node)


def test_small_batches_degrade_to_zero_instead_of_raising():
    """Tail batches are ordinary; a diagnostic must not be what kills a run."""
    single = torch.randn(1, 8)

    assert float(batch_spread(single)) == 0.0
    assert float(effective_rank(single)) == 0.0
    rmse, corr = gram_agreement(single, torch.randn(1, 32))
    assert float(rmse) == 0.0 and float(corr) == 0.0
    alignment, uniformity = alignment_uniformity(single, single)
    assert math.isfinite(float(alignment)) and float(uniformity) == 0.0
