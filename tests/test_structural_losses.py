"""Matched structural controls for the H0 reviewer experiment."""

import pytest
import torch

from src.criterions.geoode_kd import GeoODEKD
from src.criterions.h0_topological_loss import h0_topological_loss
from src.criterions.structural_losses import (
    structural_distribution_loss,
    structural_loss_against_target,
    structural_target,
)
from src.data_utils.dataset_cache import DualTokenizerCollateWithTeacher

KINDS = ("sorted_pairwise", "teacher_mst", "knn_distribution")


class _Tokenizer:
    def __call__(self, texts, **_kwargs):
        rows = len(texts)
        return {
            "input_ids": torch.ones(rows, 3, dtype=torch.long),
            "attention_mask": torch.ones(rows, 3, dtype=torch.long),
        }


@pytest.mark.parametrize("kind", KINDS)
def test_structural_controls_are_zero_under_a_shared_rotation(kind):
    generator = torch.Generator().manual_seed(1)
    teacher = torch.randn(12, 6, generator=generator)
    q, _ = torch.linalg.qr(torch.randn(6, 6, generator=generator))
    student = teacher @ q

    loss = structural_distribution_loss(student, teacher, kind)

    assert float(loss) == pytest.approx(0.0, abs=2e-10)


@pytest.mark.parametrize("kind", KINDS)
def test_precomputed_teacher_target_matches_the_direct_path_with_chunking(kind):
    generator = torch.Generator().manual_seed(2)
    teacher = torch.randn(16, 9, generator=generator)
    student = torch.randn(16, 5, generator=generator)
    values, edges = structural_target(teacher, kind, chunk_size=4, knn_k=1)

    cached = structural_loss_against_target(
        student,
        values,
        kind,
        teacher_edges=edges,
        chunk_size=4,
        knn_k=1,
    )
    direct = structural_distribution_loss(student, teacher, kind, chunk_size=4, knn_k=1)

    assert torch.allclose(cached, direct)


@pytest.mark.parametrize("kind", KINDS)
def test_every_structural_control_has_a_finite_student_gradient(kind):
    generator = torch.Generator().manual_seed(3)
    teacher = torch.randn(10, 7, generator=generator)
    student = torch.randn(10, 5, generator=generator, requires_grad=True)

    loss = structural_distribution_loss(student, teacher, kind)
    loss.backward()

    assert torch.isfinite(loss)
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert float(student.grad.norm()) > 0


def test_teacher_mst_keeps_correspondence_that_h0_discards():
    generator = torch.Generator().manual_seed(4)
    teacher = torch.randn(14, 6, generator=generator)
    student = teacher[torch.randperm(14, generator=generator)]

    h0 = h0_topological_loss(student, teacher)
    mst = structural_distribution_loss(student, teacher, "teacher_mst")

    assert float(h0) == pytest.approx(0.0, abs=2e-10)
    assert float(mst) > 1e-4


@pytest.mark.parametrize("kind", KINDS)
def test_geoode_reports_the_selected_structural_term(kind):
    generator = torch.Generator().manual_seed(5)
    hidden = [torch.randn(8, 3, 5, generator=generator, requires_grad=True)]
    teacher = torch.randn(8, 5, generator=generator)
    teacher_topo = torch.randn(8, 9, generator=generator)
    criterion = GeoODEKD(
        lambda_end=0.0,
        lambda_ctr=0.0,
        lambda_topo=0.5,
        structural_loss=kind,
    )

    total, metrics = criterion(hidden, teacher, teacher_topo=teacher_topo)

    key = f"loss_{kind}"
    assert metrics[key] > 0
    assert metrics["loss_structural"] == pytest.approx(metrics[key])
    assert metrics["loss_h0"] == 0.0
    assert float(total.detach()) == pytest.approx(0.5 * metrics[key])


def test_h1_cannot_be_combined_with_a_non_h0_control():
    with pytest.raises(ValueError, match="only defined"):
        GeoODEKD(structural_loss="teacher_mst", lambda_h1=0.1)


@pytest.mark.parametrize("kind", KINDS)
def test_collate_precomputes_the_selected_teacher_structural_target(kind):
    generator = torch.Generator().manual_seed(6)
    samples = [
        (
            (f"a-{i}", f"b-{i}"),
            torch.randn(5, generator=generator),
            torch.randn(8, generator=generator),
        )
        for i in range(8)
    ]
    collate = DualTokenizerCollateWithTeacher(
        _Tokenizer(),
        "pair_cls",
        16,
        topo_metric="chord",
        structural_loss=kind,
        structural_knn_k=1,
    )

    batch = collate(samples)

    assert "teacher_structural_values" in batch
    assert "teacher_deaths" not in batch
    assert batch["teacher_structural_values"].shape[0] in (7, 8)
    assert ("teacher_structural_edges" in batch) is (kind == "teacher_mst")


@pytest.mark.parametrize("bad", ["all_pairs", "mst", "knn"])
def test_unknown_structural_loss_is_rejected(bad):
    with pytest.raises(ValueError, match="structural_loss"):
        GeoODEKD(structural_loss=bad)
