"""Tests for the H1 persistence term.

Where H0 reads how the batch merges, H1 reads what it encloses: every 1-cycle of
the Vietoris-Rips filtration, born when the edge that closes it arrives and dead
when the triangle that fills it does. These tests pin the three things the term
rests on -- that the differentiable diagram is the diagram gudhi computes, that
``W_2^2`` is the true optimal partial matching (diagonal included) and not a
sorted-vector surrogate, and that the invariances which make the term comparable
across d_T and d_S survive.
"""

import numpy as np
import pytest
import torch

gudhi = pytest.importorskip("gudhi")

from src.criterions.geoode_kd import GeoODEKD
from src.criterions.h0_topological_loss import pairwise_distance
from src.criterions.h1_topological_loss import (
    H1TopologicalLoss,
    h1_diagram,
    h1_loss_against_diagram,
    h1_persistence_pairs,
    h1_topological_loss,
    wasserstein2_squared,
)


def _cloud(batch: int, dim: int, seed: int) -> torch.Tensor:
    return torch.randn(batch, dim, generator=torch.Generator().manual_seed(seed))


def _circle(points: int, dim: int = 3, noise: float = 0.0, seed: int = 0):
    """A cloud with one large 1-cycle: the thing the term is supposed to see."""
    angle = torch.linspace(0, 2 * torch.pi, points + 1)[:-1]
    ring = torch.zeros(points, dim)
    ring[:, 0], ring[:, 1] = torch.cos(angle), torch.sin(angle)
    if noise:
        ring = ring + noise * _cloud(points, dim, seed)
    return ring


def _reference_diagram(embeddings: torch.Tensor, metric: str = "chord"):
    """gudhi's own H1 diagram, read straight off ``persistence_intervals_in_dimension``."""
    dist = pairwise_distance(embeddings, metric=metric).numpy().astype(np.float64)
    tree = gudhi.RipsComplex(distance_matrix=dist).create_simplex_tree(max_dimension=2)
    tree.compute_persistence(persistence_dim_max=False)
    return tree.persistence_intervals_in_dimension(1)


def _sorted_rows(diagram) -> np.ndarray:
    array = np.asarray(diagram, dtype=np.float64).reshape(-1, 2)
    return array[np.lexsort((array[:, 1], array[:, 0]))]


# ------------------------------------------------------------------- the diagram


@pytest.mark.parametrize("metric", ["chord", "angular", "cosine"])
def test_the_differentiable_diagram_is_gudhis_diagram(metric):
    """The critical-edge indices are the whole trick: reading the live distance
    matrix at them has to reproduce the filtration values gudhi paired."""
    cloud = _cloud(14, 6, 0)
    ours = h1_diagram(cloud, metric=metric).detach().numpy()
    assert np.allclose(
        _sorted_rows(ours), _sorted_rows(_reference_diagram(cloud, metric)), atol=1e-9
    )


def test_the_pairs_are_edges_that_bracket_each_other():
    """A birth edge arrives before the death edge that fills its cycle."""
    dist = pairwise_distance(_cloud(16, 5, 1))
    pairs = h1_persistence_pairs(dist)
    assert pairs.shape[1] == 4 and pairs.shape[0] > 0
    births = dist[pairs[:, 0], pairs[:, 1]]
    deaths = dist[pairs[:, 2], pairs[:, 3]]
    assert (deaths > births).all()


def test_a_ring_has_one_dominant_cycle():
    diagram = h1_diagram(_circle(20)).detach()
    persistence = diagram[:, 1] - diagram[:, 0]
    assert diagram.shape[0] >= 1
    # The ring's own loop outlives every sampling artefact by a wide margin.
    ranked = torch.sort(persistence, descending=True).values
    assert ranked[0] > 1.0
    if ranked.shape[0] > 1:
        assert ranked[0] > 5 * ranked[1]


def test_too_few_points_for_a_cycle_gives_an_empty_diagram():
    for batch in (2, 1):
        assert h1_diagram(_cloud(batch, 4, 2)).shape == (0, 2)


# --------------------------------------------------------------- the W2 matching


def test_wasserstein_is_zero_between_a_diagram_and_itself():
    diagram = h1_diagram(_cloud(15, 5, 3)).detach()
    assert wasserstein2_squared(diagram, diagram.clone()).item() == pytest.approx(0.0)


def test_wasserstein_matches_a_brute_force_optimum():
    """``W_2^2`` is a minimum over partial matchings; on diagrams small enough to
    enumerate, the scipy assignment has to find that same minimum."""
    import itertools

    generator = torch.Generator().manual_seed(4)
    for _ in range(20):
        n, m = int(torch.randint(1, 4, (1,), generator=generator)), 3
        a = torch.rand(n, 2, generator=generator)
        b = torch.rand(m, 2, generator=generator)
        a[:, 1] += a[:, 0]  # death > birth
        b[:, 1] += b[:, 0]

        def diag(point):
            return ((point[1] - point[0]) ** 2 / 2).item()

        best = float("inf")
        # Every partial matching: choose k pairs, send the rest to the diagonal.
        for k in range(min(n, m) + 1):
            for left in itertools.combinations(range(n), k):
                for right in itertools.permutations(range(m), k):
                    cost = sum(
                        ((a[i] - b[j]) ** 2).sum().item()
                        for i, j in zip(left, right)
                    )
                    cost += sum(diag(a[i]) for i in range(n) if i not in left)
                    cost += sum(diag(b[j]) for j in range(m) if j not in right)
                    best = min(best, cost)
        assert wasserstein2_squared(a, b).item() == pytest.approx(best, rel=1e-6)


def test_an_empty_diagram_costs_the_other_sides_persistence():
    """No cycles at all is a legitimate diagram, not a missing one: everything on
    the other side is then matched to the diagonal."""
    diagram = torch.tensor([[0.1, 0.5], [0.2, 0.9]])
    empty = torch.zeros(0, 2)
    expected = (0.4**2 + 0.7**2) / 2
    assert wasserstein2_squared(empty, diagram).item() == pytest.approx(expected)
    assert wasserstein2_squared(diagram, empty).item() == pytest.approx(expected)
    assert wasserstein2_squared(empty, empty.clone()).item() == pytest.approx(0.0)


def test_a_near_diagonal_point_is_cheap_not_forced_onto_a_partner():
    """The diagonal is what lets the two sides disagree about how many cycles there
    are; a low-persistence extra cycle must cost about its own persistence."""
    teacher = torch.tensor([[0.1, 0.9]], dtype=torch.float64)
    student = torch.tensor([[0.1, 0.9], [0.4, 0.4001]], dtype=torch.float64)
    assert wasserstein2_squared(student, teacher).item() == pytest.approx(
        (0.0001**2) / 2, rel=1e-6
    )


# ------------------------------------------------------------------- the loss


@pytest.mark.parametrize("metric", ["chord", "angular", "cosine"])
def test_identical_clouds_have_zero_loss(metric):
    cloud = _cloud(14, 7, 5)
    assert h1_topological_loss(cloud, cloud.clone(), metric=metric).item() == pytest.approx(
        0.0, abs=1e-12
    )


def test_loss_is_invariant_to_rotation_and_dimension():
    # The two spaces share no basis: this is what lets a d_S student be scored
    # against a d_T teacher with no map in between.
    teacher = _cloud(14, 10, 6)
    rotation, _ = torch.linalg.qr(_cloud(10, 10, 7))
    padded = torch.cat([teacher @ rotation, torch.zeros(14, 6)], dim=-1)
    assert h1_topological_loss(padded, teacher).item() == pytest.approx(0.0, abs=1e-8)


def test_a_ring_is_not_free_to_flatten():
    """The term's reason to exist: a cloud that keeps the teacher's merge tree can
    still lose the teacher's loop, and only H1 charges for that."""
    teacher = _circle(20)
    student = teacher.clone()
    student[:, 1] *= 0.02  # the ring collapses to a segment; the cycle is gone
    assert h1_topological_loss(student, teacher).item() > 1.0


def test_gradient_reaches_the_student_only():
    student = _circle(16, noise=0.15, seed=8).requires_grad_(True)
    teacher = _cloud(16, 32, 9).requires_grad_(True)
    h1_topological_loss(student, teacher).backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert student.grad.norm() > 0
    assert teacher.grad is None


def test_the_gradient_shrinks_the_gap_it_is_asked_to_shrink():
    """One step downhill has to lower the loss -- the point of re-reading the chosen
    pairs and the chosen matching off the live tensors."""
    # The ground metric lives on the unit sphere, so the student has to differ from
    # the teacher in *shape*, not scale: a scaled ring normalises back onto the ring.
    teacher = _circle(18)
    student = _circle(18, noise=0.3, seed=18).requires_grad_(True)
    before = h1_topological_loss(student, teacher)
    assert before.item() > 0.0
    before.backward()
    with torch.no_grad():
        stepped = student - 0.05 * student.grad
    assert h1_topological_loss(stepped, teacher).item() < before.item()


def test_rejects_mismatched_batches():
    with pytest.raises(ValueError):
        h1_topological_loss(_cloud(8, 4, 10), _cloud(9, 4, 11))


def test_module_wrapper_matches_the_function():
    student, teacher = _cloud(12, 6, 12), _cloud(12, 9, 13)
    module = H1TopologicalLoss(metric="angular")
    expected = h1_topological_loss(student, teacher, metric="angular")
    assert module(student, teacher).item() == pytest.approx(expected.item())


def test_a_precomputed_teacher_diagram_gives_the_same_loss():
    """Where the teacher's diagram is built is a scheduling choice, exactly as for
    H0: it reads a frozen cache under no_grad and depends on nothing the step does."""
    student = _circle(16, noise=0.2, seed=14).requires_grad_(True)
    teacher = _cloud(16, 32, 15)

    joint = h1_topological_loss(student, teacher)
    with torch.no_grad():
        teacher_diagram = h1_diagram(teacher)
    split = h1_loss_against_diagram(student, teacher_diagram)

    assert torch.allclose(joint, split, atol=1e-7)
    joint.backward()
    reference = student.grad.clone()
    student.grad = None
    split.backward()
    assert torch.allclose(reference, student.grad, atol=1e-7)


# ------------------------------------------------------------------ integration


def _geoode_batch(batch=12, dim=8, tokens=5, layers=3, teacher_dim=16, seed=16):
    generator = torch.Generator().manual_seed(seed)
    hidden_states = [
        torch.randn(batch, tokens, dim, generator=generator) for _ in range(layers + 1)
    ]
    teacher = torch.nn.functional.normalize(
        torch.randn(batch, dim, generator=generator), dim=-1
    )
    teacher_topo = torch.randn(batch, teacher_dim, generator=generator)
    return hidden_states, teacher, teacher_topo


def test_lambda_h1_zero_leaves_l_topo_the_h0_term():
    hidden_states, teacher, teacher_topo = _geoode_batch()
    _, metrics = GeoODEKD(lambda_ctr=0.0, lambda_topo=1.0)(
        hidden_states=hidden_states, teacher=teacher, teacher_topo=teacher_topo
    )
    assert metrics["loss_h1"] == 0.0
    assert metrics["loss_topo"] == pytest.approx(metrics["loss_h0"])


def test_lambda_h1_adds_the_weighted_h1_term():
    hidden_states, teacher, teacher_topo = _geoode_batch()
    criterion = GeoODEKD(lambda_ctr=0.0, lambda_topo=0.5, lambda_h1=0.25)
    total, metrics = criterion(
        hidden_states=hidden_states, teacher=teacher, teacher_topo=teacher_topo
    )
    assert metrics["loss_h1"] > 0.0
    assert metrics["loss_topo"] == pytest.approx(
        metrics["loss_h0"] + 0.25 * metrics["loss_h1"], rel=1e-5
    )
    assert total.item() == pytest.approx(
        metrics["loss_end"] + 0.5 * metrics["loss_topo"], rel=1e-5
    )


def test_lambda_topo_zero_switches_off_h1_as_well():
    """L_topo carries both halves, so its weight gates both -- and with it the whole
    gudhi call, which is the expensive part."""
    hidden_states, teacher, teacher_topo = _geoode_batch()
    _, metrics = GeoODEKD(lambda_ctr=0.0, lambda_topo=0.0, lambda_h1=1.0)(
        hidden_states=hidden_states, teacher=teacher, teacher_topo=teacher_topo
    )
    assert metrics["loss_h1"] == 0.0 and metrics["loss_topo"] == 0.0


def test_the_h1_term_reads_the_unprojected_teacher_when_given_one():
    hidden_states, teacher, teacher_topo = _geoode_batch(teacher_dim=32)
    criterion = GeoODEKD(lambda_ctr=0.0, lambda_topo=1.0, lambda_h1=1.0)
    _, with_topo = criterion(
        hidden_states=hidden_states, teacher=teacher, teacher_topo=teacher_topo
    )
    _, fallback = criterion(hidden_states=hidden_states, teacher=teacher)
    assert with_topo["loss_h1"] != pytest.approx(fallback["loss_h1"])


def test_a_precomputed_teacher_h1_diagram_supersedes_the_cache():
    hidden_states, teacher, teacher_topo = _geoode_batch()
    criterion = GeoODEKD(lambda_ctr=0.0, lambda_topo=1.0, lambda_h1=1.0)
    _, joint = criterion(
        hidden_states=hidden_states, teacher=teacher, teacher_topo=teacher_topo
    )
    with torch.no_grad():
        diagram = h1_diagram(teacher_topo.float())
    _, split = criterion(
        hidden_states=hidden_states,
        teacher=teacher,
        teacher_topo=teacher_topo,
        teacher_h1=diagram,
    )
    assert split["loss_h1"] == pytest.approx(joint["loss_h1"], rel=1e-6)


def test_the_h1_term_is_skipped_on_a_batch_too_small_for_a_cycle():
    for batch in (1, 2):
        hidden_states, teacher, teacher_topo = _geoode_batch(batch=batch)
        _, metrics = GeoODEKD(lambda_ctr=0.0, lambda_topo=1.0, lambda_h1=1.0)(
            hidden_states=hidden_states, teacher=teacher, teacher_topo=teacher_topo
        )
        assert metrics["loss_h1"] == 0.0


def test_rejects_a_negative_h1_weight():
    with pytest.raises(ValueError):
        GeoODEKD(lambda_h1=-1.0)


def test_the_collate_builds_the_teacher_h1_diagram_only_when_asked():
    from src.data_utils.dataset_cache import DualTokenizerCollateWithTeacher

    class _Tokenizer:
        def __call__(self, texts, **kwargs):
            count = len(texts)
            return {
                "input_ids": torch.zeros(count, 3, dtype=torch.long),
                "attention_mask": torch.ones(count, 3, dtype=torch.long),
            }

    topo = _circle(8, dim=5, noise=0.2, seed=17)
    batch = [(("a", "b"), torch.zeros(4), topo[i]) for i in range(8)]

    without = DualTokenizerCollateWithTeacher(
        _Tokenizer(), "pair_reg", 8, topo_metric="chord"
    )(batch)
    assert "teacher_h1" not in without and "teacher_deaths" in without

    with_h1 = DualTokenizerCollateWithTeacher(
        _Tokenizer(), "pair_reg", 8, topo_metric="chord", need_h1=True
    )(batch)
    assert with_h1["teacher_h1"].shape[-1] == 2
    assert torch.allclose(with_h1["teacher_h1"], h1_diagram(topo), atol=1e-6)


# --------------------------------------------------------------------- chunking


def test_the_chunked_h1_term_is_the_mean_of_the_chunk_terms():
    student, teacher = _cloud(18, 5, 30), _circle(18, dim=7, noise=0.1, seed=31)

    chunked = h1_topological_loss(student, teacher, chunk_size=6)
    by_hand = torch.stack(
        [
            h1_topological_loss(student[k * 6 : (k + 1) * 6], teacher[k * 6 : (k + 1) * 6])
            for k in range(3)
        ]
    ).mean()

    assert chunked.item() == pytest.approx(by_hand.item(), rel=1e-6)


def test_chunking_off_is_the_h1_term_that_was_there_before():
    student, teacher = _cloud(9, 5, 32), _circle(9, dim=7, noise=0.1, seed=33)
    assert h1_topological_loss(student, teacher, chunk_size=0).item() == pytest.approx(
        h1_topological_loss(student, teacher).item()
    )


def test_the_collate_hands_the_raw_cache_over_when_h1_is_chunked():
    """Per-chunk H1 diagrams have different numbers of cycles, so they cannot be
    stacked into the batch dict; the step builds them from the cache instead."""
    from src.data_utils.dataset_cache import DualTokenizerCollateWithTeacher

    class _Tokenizer:
        def __call__(self, texts, **kwargs):
            count = len(texts)
            return {
                "input_ids": torch.zeros(count, 3, dtype=torch.long),
                "attention_mask": torch.ones(count, 3, dtype=torch.long),
            }

    topo = _circle(12, dim=5, noise=0.2, seed=34)
    batch = [(("a", "b"), torch.zeros(4), topo[i]) for i in range(12)]

    chunked = DualTokenizerCollateWithTeacher(
        _Tokenizer(), "pair_reg", 8, topo_metric="chord", need_h1=True, topo_batch_size=6
    )(batch)

    assert "teacher_h1" not in chunked
    assert torch.allclose(chunked["teacher_topo"], topo)
    assert chunked["teacher_deaths"].shape == (2, 5)


def test_the_criterion_chunks_the_h1_term_too():
    hidden_states, teacher, teacher_topo = _geoode_batch(batch=18, teacher_dim=16)
    criterion = GeoODEKD(
        lambda_ctr=0.0, lambda_topo=1.0, lambda_h1=1.0, topo_batch_size=6
    )
    _, metrics = criterion(
        hidden_states=hidden_states, teacher=teacher, teacher_topo=teacher_topo
    )
    student = criterion.endpoint_states(hidden_states, None)[-1]
    expected = h1_topological_loss(student, teacher_topo.float(), chunk_size=6)
    assert metrics["loss_h1"] == pytest.approx(expected.item(), rel=1e-5)


def test_a_chunk_too_small_to_close_a_cycle_drops_the_h1_term():
    hidden_states, teacher, teacher_topo = _geoode_batch(batch=18, teacher_dim=16)
    _, metrics = GeoODEKD(
        lambda_ctr=0.0, lambda_topo=1.0, lambda_h1=1.0, topo_batch_size=2
    )(hidden_states=hidden_states, teacher=teacher, teacher_topo=teacher_topo)
    assert metrics["h1_active"] == 0.0
    assert metrics["loss_h1"] == 0.0
