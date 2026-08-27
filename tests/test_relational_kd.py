"""Tests for RKD (Park et al., 2019).

What makes RKD usable as the relational baseline here is that its two potentials
are invariant to everything the student's space is free to choose — its width,
its scale and its orientation — so the tests pin exactly that, then check the
pieces the invariance is built from.
"""

import pytest
import torch
import torch.nn.functional as F

from src.criterions.relational_kd import RelationalKD


def _points(batch: int, dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(
        torch.randn(batch, dim, generator=generator, dtype=torch.float64), dim=-1
    )


def _rotation(dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=generator, dtype=torch.float64))
    return q


def _criterion(**kwargs) -> RelationalKD:
    defaults = {"normalize_student": False}
    defaults.update(kwargs)
    return RelationalKD(**defaults)


def test_criterion_has_no_trainable_parameters():
    # Nothing is fitted between the teacher and the student, so the deployed model
    # is the plain encoder and the row is comparable with the parameter-free ones.
    assert list(RelationalKD().parameters()) == []


def test_pairwise_distance_matches_cdist_with_a_zero_diagonal():
    x = _points(7, 5, seed=0)

    distance = RelationalKD.pairwise_distance(x)

    assert torch.allclose(distance, torch.cdist(x, x), atol=1e-9)
    assert torch.equal(distance.diagonal(), torch.zeros(7, dtype=torch.float64))


def test_angles_are_the_cosines_at_the_middle_point():
    # A right angle at the origin: the entry for (x, origin, y) must be exactly 0,
    # and the one for (x, origin, x) exactly 1.
    x = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64
    )

    angles = _criterion()._angles(x).view(3, 3, 3)

    assert angles[0, 1, 2] == pytest.approx(0.0, abs=1e-12)
    assert angles[0, 1, 1] == pytest.approx(1.0, abs=1e-12)
    assert angles.abs().max() <= 1.0 + 1e-9


def test_potentials_vanish_on_a_scaled_and_rotated_copy():
    # The student is free to place the embeddings at any scale and in any basis;
    # only the relations between them are supervised.
    teacher = _points(8, 16, seed=1)
    student = 3.7 * (teacher @ _rotation(16, seed=2))
    criterion = _criterion()

    assert criterion.distance_loss(student, teacher) == pytest.approx(0.0, abs=1e-12)
    assert criterion.angle_loss(student, teacher) == pytest.approx(0.0, abs=1e-12)


def test_scrambled_relations_cost_more_than_matched_ones():
    teacher = _points(8, 16, seed=3)
    permutation = torch.randperm(8, generator=torch.Generator().manual_seed(4))
    criterion = _criterion()

    assert criterion.distance_loss(teacher[permutation], teacher) > 1e-3
    assert criterion.angle_loss(teacher[permutation], teacher) > 1e-3


def test_the_teacher_may_be_wider_than_the_student():
    # The point of a relational objective: a 16-d teacher supervises an 8-d student
    # with no projection fitted between them, and the gradient reaches the student.
    teacher = _points(6, 16, seed=5)
    student = _points(6, 8, seed=6).float().requires_grad_(True)

    loss, metrics = _criterion()(student, teacher.float())
    loss.backward()

    assert torch.isfinite(loss)
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert metrics["loss_total"] == pytest.approx(
        25.0 * metrics["loss_dist"] + 50.0 * metrics["loss_angle"], rel=1e-5
    )


def test_normalize_student_measures_the_relations_on_the_sphere():
    # Cosine geometry is what the benchmarks score, so with normalisation on, a
    # student that differs from the teacher only by per-row scaling is already
    # right; with it off, the varying norms show up as a mismatch.
    teacher = _points(6, 12, seed=7).float()
    scales = torch.linspace(0.5, 4.0, 6, dtype=torch.float32).unsqueeze(-1)
    student = scales * teacher

    on, _ = RelationalKD(w_task=0.0, normalize_student=True)(student, teacher)
    off, _ = RelationalKD(w_task=0.0, normalize_student=False)(student, teacher)

    assert on == pytest.approx(0.0, abs=1e-8)
    assert off > 1e-3


def test_task_loss_enters_with_its_own_weight():
    teacher = _points(6, 12, seed=8).float()
    student = (teacher @ _rotation(12, seed=9).float())
    task_loss = torch.tensor(0.75)

    loss, metrics = RelationalKD(
        w_task=0.4, w_dist=25.0, w_angle=50.0, normalize_student=False
    )(student, teacher, task_loss)

    assert metrics["loss_task"] == pytest.approx(0.75)
    assert float(loss) == pytest.approx(0.4 * 0.75, abs=1e-5)


def test_a_zero_weight_drops_its_term():
    teacher = _points(6, 12, seed=10).float()
    student = _points(6, 12, seed=11).float()

    only_distance, metrics = RelationalKD(
        w_task=0.0, w_angle=0.0, normalize_student=False
    )(student, teacher)

    assert metrics["loss_angle"] == 0.0
    assert float(only_distance) == pytest.approx(25.0 * metrics["loss_dist"], rel=1e-5)


def test_an_all_identical_batch_has_no_relations_to_match():
    # Every pairwise distance is zero, so the batch mean the potential divides by is
    # zero too: the term has to vanish rather than produce a NaN.
    teacher = _points(1, 12, seed=12).float().expand(5, 12).contiguous()

    loss, metrics = RelationalKD(w_task=0.0, normalize_student=False)(teacher, teacher)

    assert torch.isfinite(loss)
    assert metrics["loss_dist"] == pytest.approx(0.0, abs=1e-9)


def test_batch_mismatch_is_rejected():
    with pytest.raises(ValueError, match="Batch mismatch"):
        RelationalKD()(_points(4, 8, seed=13).float(), _points(5, 16, seed=14).float())
