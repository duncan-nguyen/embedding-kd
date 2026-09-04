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
    chunk_count,
    h0_death_times,
    h0_loss_against_deaths,
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


# --------------------------------------------------------------------------- #
# The MST that reads the diagram: same tree, taken off the kernel-launch path
# --------------------------------------------------------------------------- #


def _prim_reference(dist: torch.Tensor) -> torch.Tensor:
    """The B-1 step torch loop the scipy selection replaced, kept as the oracle."""
    B = dist.shape[0]
    d_select = dist.detach()
    in_tree = torch.zeros(B, dtype=torch.bool, device=dist.device)
    in_tree[0] = True
    min_cost = d_select[0].clone()
    parent = torch.zeros(B, dtype=torch.long, device=dist.device)
    min_cost[0] = float("inf")

    edge_i, edge_j = [], []
    for _ in range(B - 1):
        candidate = min_cost.masked_fill(in_tree, float("inf"))
        j = torch.argmin(candidate)
        edge_i.append(parent[j])
        edge_j.append(j)
        in_tree[j] = True
        better = (d_select[j] < min_cost) & (~in_tree)
        min_cost = torch.where(better, d_select[j], min_cost)
        parent = torch.where(better, j.expand_as(parent), parent)

    return dist[torch.stack(edge_i), torch.stack(edge_j)]


@pytest.mark.parametrize("metric", ["chord", "angular", "cosine"])
@pytest.mark.parametrize("batch", [2, 3, 5, 17, 32])
def test_the_scipy_mst_is_the_tree_the_prim_loop_found(metric, batch):
    generator = torch.Generator().manual_seed(batch)
    for _ in range(10):
        dist = pairwise_distance(
            torch.randn(batch, 8, generator=generator), metric=metric
        )
        assert torch.allclose(
            torch.sort(mst_edge_weights(dist)).values,
            torch.sort(_prim_reference(dist)).values,
            atol=1e-6,
        )


@pytest.mark.parametrize("metric", ["chord", "angular", "cosine"])
def test_duplicate_rows_still_agree(metric):
    """A corpus repeats sentences, and equal texts cache to equal vectors, so a
    batch really does contain rows at distance zero from each other -- the case
    scipy would read as "no edge" if the weights were not shifted off zero first."""
    generator = torch.Generator().manual_seed(7)
    base = torch.randn(5, 8, generator=generator)
    for _ in range(25):
        rows = torch.randint(0, 5, (24,), generator=generator)
        dist = pairwise_distance(base[rows], metric=metric)
        assert torch.allclose(
            torch.sort(mst_edge_weights(dist)).values,
            torch.sort(_prim_reference(dist)).values,
            atol=1e-6,
        )


@pytest.mark.parametrize("metric", ["chord", "angular", "cosine"])
def test_the_distance_matrix_is_exactly_symmetric(metric):
    """``sim[i, j]`` and ``sim[j, i]`` are one dot product summed in two orders, so
    the matmul disagrees with itself in the last bit -- and at the clamp floor, where
    near-identical rows land, that bit decides whether a distance is clamped. An
    asymmetric matrix is not a metric graph: the tree it defines then depends on
    which triangle the algorithm reads, which is exactly how the two MST routines
    came to disagree on duplicate rows."""
    generator = torch.Generator().manual_seed(11)
    for rows in (2, 9, 33):
        dist = pairwise_distance(torch.randn(rows, 8, generator=generator), metric=metric)
        assert torch.equal(dist, dist.T)


def test_the_mst_weights_stay_differentiable():
    """Selecting the tree is detached; the weights it selects are not."""
    embeddings = torch.randn(12, 8, requires_grad=True)
    mst_edge_weights(pairwise_distance(embeddings, metric="chord")).sum().backward()

    assert torch.isfinite(embeddings.grad).all()
    assert embeddings.grad.abs().sum() > 0


def test_a_precomputed_teacher_diagram_gives_the_same_loss():
    """Where the teacher's diagram is built is a scheduling choice: it reads a
    frozen cache under no_grad and depends on nothing the step computes."""
    generator = torch.Generator().manual_seed(3)
    student = torch.randn(16, 8, generator=generator, requires_grad=True)
    teacher = torch.randn(16, 32, generator=generator)

    joint = h0_topological_loss(student, teacher, metric="chord")
    with torch.no_grad():
        deaths = h0_death_times(teacher, metric="chord", sort=True)
    split = h0_loss_against_deaths(student, deaths, metric="chord")

    assert torch.allclose(joint, split, atol=1e-7)
    joint.backward()
    reference = student.grad.clone()
    student.grad = None
    split.backward()
    assert torch.allclose(reference, student.grad, atol=1e-7)


def test_a_diagram_from_a_different_batch_is_refused():
    """B - 1 death times belong to the batch of B they were built from; a shape that
    does not match means the collate and the step disagree about the batch."""
    student = torch.randn(8, 4)
    with pytest.raises(ValueError, match="death times"):
        h0_loss_against_deaths(student, torch.zeros(5))


# --------------------------------------------------------------------- chunking
# The cloud a diagram describes and the batch the optimizer steps on are two
# independent choices; --topo_batch_size separates them. What has to hold is that
# the chunked term is exactly the per-chunk term averaged, that both sides of the
# loss cut the batch the same way, and that leaving the knob alone reproduces the
# whole-batch term bit for bit.


@pytest.mark.parametrize("chunk_size", [None, 0, 16, 17, 100])
def test_a_chunk_that_does_not_fit_twice_is_the_whole_batch(chunk_size):
    assert chunk_count(16, chunk_size) == 1


def test_chunk_count_counts_only_whole_chunks():
    assert chunk_count(16, 8) == 2
    assert chunk_count(16, 5) == 3  # the 16th row sits the term out
    assert chunk_count(16, 2) == 8


def test_a_chunk_of_one_point_is_refused():
    # A one-point cloud has no MST at all, so this is a typo rather than a request.
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_count(16, 1)


def test_chunked_deaths_are_the_chunks_own_diagrams():
    x = _cloud(20, 6, 10)
    chunked = h0_death_times(x, chunk_size=8)

    assert chunked.shape == (2, 7)  # 20 // 8 = 2 clouds; four rows dropped
    for index in range(2):
        expected = h0_death_times(x[index * 8 : (index + 1) * 8])
        assert torch.allclose(chunked[index], expected, atol=1e-7)


def test_the_chunked_loss_is_the_mean_of_the_chunk_losses():
    student, teacher = _cloud(24, 5, 11), _cloud(24, 9, 12)

    chunked = h0_topological_loss(student, teacher, chunk_size=8)
    by_hand = torch.stack(
        [
            h0_topological_loss(student[k * 8 : (k + 1) * 8], teacher[k * 8 : (k + 1) * 8])
            for k in range(3)
        ]
    ).mean()

    assert chunked.item() == pytest.approx(by_hand.item(), rel=1e-6)


def test_chunking_off_is_the_term_that_was_there_before():
    student, teacher = _cloud(12, 5, 13), _cloud(12, 9, 14)
    assert h0_topological_loss(student, teacher, chunk_size=0).item() == pytest.approx(
        h0_topological_loss(student, teacher).item()
    )


def test_a_chunked_loss_sees_a_shorter_scale_than_the_whole_batch():
    """The point of the knob: b sets how far up the merge tree the term can see, so
    the two readings of the same pair of clouds are genuinely different numbers."""
    student, teacher = _cloud(32, 5, 15), _cloud(32, 9, 16)
    whole = h0_topological_loss(student, teacher)
    chunked = h0_topological_loss(student, teacher, chunk_size=8)
    assert chunked.item() != pytest.approx(whole.item(), rel=1e-3)


def test_a_precomputed_chunked_teacher_diagram_gives_the_same_loss():
    """The collate builds the [n, b-1] teacher side; the step must cut the student's
    rows by exactly the same rule or the two diagrams are of different clouds."""
    generator = torch.Generator().manual_seed(17)
    student = torch.randn(24, 8, generator=generator, requires_grad=True)
    teacher = torch.randn(24, 32, generator=generator)

    joint = h0_topological_loss(student, teacher, chunk_size=8)
    with torch.no_grad():
        deaths = h0_death_times(teacher, sort=True, chunk_size=8)
    split = h0_loss_against_deaths(student, deaths, chunk_size=8)

    assert torch.allclose(joint, split, atol=1e-7)
    joint.backward()
    reference = student.grad.clone()
    student.grad = None
    split.backward()
    assert torch.allclose(reference, student.grad, atol=1e-7)


def test_a_diagram_built_with_a_different_chunk_size_is_refused():
    student, teacher = _cloud(24, 5, 18), _cloud(24, 9, 19)
    deaths = h0_death_times(teacher, chunk_size=8)
    with pytest.raises(ValueError, match="death times"):
        h0_loss_against_deaths(student, deaths, chunk_size=12)


def test_the_dropped_tail_rows_do_not_reach_the_term():
    """B mod b rows sit L_topo out, so moving them cannot move the loss."""
    student, teacher = _cloud(20, 5, 20), _cloud(20, 9, 21)
    moved = student.clone()
    moved[16:] = _cloud(4, 5, 22)  # only the dropped tail changes

    assert h0_topological_loss(moved, teacher, chunk_size=8).item() == pytest.approx(
        h0_topological_loss(student, teacher, chunk_size=8).item()
    )


def test_topo_batch_size_makes_the_criterion_read_chunks():
    hidden_states, teacher, teacher_topo = _geoode_batch(batch=24, teacher_dim=16)
    whole = GeoODEKD(lambda_ctr=0.0, lambda_topo=1.0)
    chunked = GeoODEKD(lambda_ctr=0.0, lambda_topo=1.0, topo_batch_size=8)

    _, whole_metrics = whole(
        hidden_states=hidden_states, teacher=teacher, teacher_topo=teacher_topo
    )
    _, chunk_metrics = chunked(
        hidden_states=hidden_states, teacher=teacher, teacher_topo=teacher_topo
    )
    student = chunked.endpoint_states(hidden_states, None)[-1]
    expected = h0_topological_loss(student, teacher_topo.float(), chunk_size=8)

    assert chunk_metrics["loss_h0"] == pytest.approx(expected.item(), rel=1e-5)
    assert chunk_metrics["loss_h0"] != pytest.approx(whole_metrics["loss_h0"], rel=1e-3)


def test_the_criterion_takes_a_chunked_teacher_diagram_from_the_collate():
    hidden_states, teacher, teacher_topo = _geoode_batch(batch=24, teacher_dim=16)
    criterion = GeoODEKD(lambda_ctr=0.0, lambda_topo=1.0, topo_batch_size=8)

    _, from_cache = criterion(
        hidden_states=hidden_states, teacher=teacher, teacher_topo=teacher_topo
    )
    _, from_collate = criterion(
        hidden_states=hidden_states,
        teacher=teacher,
        teacher_deaths=h0_death_times(teacher_topo.float(), sort=True, chunk_size=8),
    )
    assert from_collate["loss_h0"] == pytest.approx(from_cache["loss_h0"], rel=1e-6)


def test_a_batch_smaller_than_the_chunk_is_still_one_diagram():
    """An epoch's tail batch is not an error: it is read whole, as it always was."""
    hidden_states, teacher, teacher_topo = _geoode_batch(batch=6, teacher_dim=16)
    criterion = GeoODEKD(lambda_ctr=0.0, lambda_topo=1.0, topo_batch_size=8)
    _, metrics = criterion(
        hidden_states=hidden_states, teacher=teacher, teacher_topo=teacher_topo
    )
    _, whole = GeoODEKD(lambda_ctr=0.0, lambda_topo=1.0)(
        hidden_states=hidden_states, teacher=teacher, teacher_topo=teacher_topo
    )
    assert metrics["loss_h0"] == pytest.approx(whole["loss_h0"], rel=1e-6)


def test_the_criterion_refuses_a_chunk_of_one():
    with pytest.raises(ValueError, match="topo_batch_size"):
        GeoODEKD(topo_batch_size=1)
    with pytest.raises(ValueError, match="topo_batch_size"):
        GeoODEKD(topo_batch_size=-4)
