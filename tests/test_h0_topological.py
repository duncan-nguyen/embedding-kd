"""Tests for the H0 persistence term.

The claim the term rests on is that the sorted finite H0 death times of a batch
are its minimum spanning tree's edge weights, read in a space the two models do
not share: no correspondence between axes, no equality of dimension, only the
shape of the cloud. These tests pin that -- the MST against a reference solver,
the invariances that make the term comparable across d_T and d_S, and the fact
that the gradient reaches the student and stops at the teacher.
"""

import pytest
import torch
from scipy.sparse.csgraph import minimum_spanning_tree

from src.criterions.geoode_kd import GeoODEKD
from src.criterions.h0_topological_loss import (
    H0TopologicalLoss,
    h0_death_times,
    h0_topological_loss,
    mst_edge_weights,
    pairwise_distance,
)


def _cloud(batch: int, dim: int, seed: int) -> torch.Tensor:
    return torch.randn(batch, dim, generator=torch.Generator().manual_seed(seed))


def test_mst_matches_a_reference_solver():
    # Prim's is hand-rolled here to stay on-device and differentiable; scipy's
    # Kruskal is the check that it is still an MST.
    dist = pairwise_distance(_cloud(12, 5, 0))
    ours = torch.sort(mst_edge_weights(dist)).values
    reference = minimum_spanning_tree(dist.numpy()).toarray()
    reference = torch.tensor(sorted(reference[reference > 0]))
    assert torch.allclose(ours, reference.to(ours.dtype), atol=1e-6)


def test_mst_has_b_minus_one_edges():
    assert h0_death_times(_cloud(9, 4, 1)).shape == (8,)


@pytest.mark.parametrize("metric", ["chord", "angular", "cosine"])
def test_identical_clouds_have_zero_loss(metric):
    x = _cloud(16, 7, 2)
    assert h0_topological_loss(x, x.clone(), metric=metric).item() == pytest.approx(0.0, abs=1e-10)


def test_loss_is_invariant_to_rotation_and_dimension():
    # The two spaces share no basis, so this invariance is the whole point: a
    # student in d_S can match a teacher in d_T without a map between them.
    teacher = _cloud(16, 12, 3)
    rotation, _ = torch.linalg.qr(_cloud(12, 12, 4))
    padded = torch.cat([teacher @ rotation, torch.zeros(16, 5)], dim=-1)
    assert h0_topological_loss(padded, teacher).item() == pytest.approx(0.0, abs=1e-8)


def test_loss_is_positive_for_differently_shaped_clouds():
    # A cloud squeezed onto one axis has shorter MST edges than an isotropic one.
    teacher = _cloud(16, 8, 5)
    student = teacher.clone()
    student[:, 1:] *= 0.05
    assert h0_topological_loss(student, teacher).item() > 1e-4


def test_gradient_reaches_the_student_only():
    student = _cloud(16, 8, 6).requires_grad_(True)
    teacher = _cloud(16, 32, 7).requires_grad_(True)
    h0_topological_loss(student, teacher).backward()
    assert student.grad is not None and student.grad.norm() > 0
    assert teacher.grad is None


def test_rejects_mismatched_batch_and_singleton():
    with pytest.raises(ValueError):
        h0_topological_loss(_cloud(8, 4, 8), _cloud(9, 4, 9))
    with pytest.raises(ValueError):
        h0_death_times(_cloud(1, 4, 10))


def test_module_wrapper_matches_the_function():
    student, teacher = _cloud(10, 6, 11), _cloud(10, 9, 12)
    module = H0TopologicalLoss(metric="angular", squared=False)
    expected = h0_topological_loss(student, teacher, metric="angular", squared=False)
    assert module(student, teacher).item() == pytest.approx(expected.item())


# ------------------------------------------------------------------ integration


def _geoode_batch(batch=6, dim=8, tokens=5, layers=3, teacher_dim=8, seed=13):
    generator = torch.Generator().manual_seed(seed)
    hidden_states = [
        torch.randn(batch, tokens, dim, generator=generator) for _ in range(layers + 1)
    ]
    teacher = torch.nn.functional.normalize(
        torch.randn(batch, dim, generator=generator), dim=-1
    )
    teacher_topo = torch.randn(batch, teacher_dim, generator=generator)
    return hidden_states, teacher, teacher_topo


def test_lambda_topo_zero_leaves_the_objective_untouched():
    hidden_states, teacher, teacher_topo = _geoode_batch()
    baseline, _ = GeoODEKD(lambda_ctr=0.0)(hidden_states=hidden_states, teacher=teacher)
    total, metrics = GeoODEKD(lambda_ctr=0.0, lambda_topo=0.0)(
        hidden_states=hidden_states, teacher=teacher, teacher_topo=teacher_topo
    )
    assert total.item() == pytest.approx(baseline.item())
    assert metrics["loss_topo"] == 0.0


def test_lambda_topo_adds_the_weighted_h0_term():
    hidden_states, teacher, teacher_topo = _geoode_batch()
    criterion = GeoODEKD(lambda_ctr=0.0, lambda_topo=0.25)
    total, metrics = criterion(
        hidden_states=hidden_states, teacher=teacher, teacher_topo=teacher_topo
    )
    assert metrics["loss_topo"] > 0.0
    assert total.item() == pytest.approx(
        metrics["loss_end"] + 0.25 * metrics["loss_topo"], rel=1e-5
    )


def test_h0_term_reads_the_unprojected_teacher_when_given_one():
    # Passing d_T-dimensional teacher embeddings must change the term -- otherwise
    # the run is silently supervised by the projected target twice over.
    hidden_states, teacher, teacher_topo = _geoode_batch(teacher_dim=16)
    criterion = GeoODEKD(lambda_ctr=0.0, lambda_topo=1.0)
    _, with_topo = criterion(
        hidden_states=hidden_states, teacher=teacher, teacher_topo=teacher_topo
    )
    _, fallback = criterion(hidden_states=hidden_states, teacher=teacher)
    assert with_topo["loss_topo"] != pytest.approx(fallback["loss_topo"])
    assert fallback["loss_topo"] > 0.0


def test_h0_term_is_skipped_on_a_singleton_batch():
    hidden_states, teacher, teacher_topo = _geoode_batch(batch=1)
    _, metrics = GeoODEKD(lambda_ctr=0.0, lambda_topo=1.0)(
        hidden_states=hidden_states, teacher=teacher, teacher_topo=teacher_topo
    )
    assert metrics["loss_topo"] == 0.0


def test_rejects_a_negative_weight_and_an_unknown_metric():
    with pytest.raises(ValueError):
        GeoODEKD(lambda_topo=-1.0)
    with pytest.raises(ValueError):
        GeoODEKD(topo_metric="euclidean")
