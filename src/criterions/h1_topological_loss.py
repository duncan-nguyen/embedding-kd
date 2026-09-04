"""Differentiable H1 topological loss for teacher-student embedding distillation.

The H0 term next door (:mod:`src.criterions.h0_topological_loss`) reads the batch's
connectivity: the sorted MST edge weights are the finite H0 death times, and nothing
about the arrangement of the points beyond how they merge. A cloud can have exactly
the teacher's merge tree and still have closed off a loop the teacher left open, or
opened one the teacher never had -- those are H1 features, and this file matches them.

Two things make the term differentiable, both standard in the differentiable-TDA
line (Carriere et al., "Optimization for persistence diagram descriptors"):

1. **The persistence pairs are a discrete choice, the filtration values are not.**
   In a Vietoris-Rips flag filtration every H1 birth and every H1 death is the
   filtration value of a single *edge* -- the edge that closes the cycle and the one
   that completes the triangle filling it. So a diagram point is a pair of entries of
   the distance matrix, and the pair of index-pairs is what gudhi computes (on the
   detached matrix) while the values themselves are re-read from the live matrix and
   carry gradient back to the embeddings.

2. **The optimal matching is a discrete choice, its cost is not.** ``W_2^2`` is a
   minimum over partial matchings; the argmin is taken once with scipy on detached
   costs and the cost of *that* matching is then recomputed in torch.

Both are the same trick the MST selection already uses in the H0 file: choose on
``detach()``, evaluate on the live tensor.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from src.criterions.h0_topological_loss import Metric, pairwise_distance, split_chunks

# A 1-cycle needs three vertices to exist at all, so a batch of two has an empty
# H1 diagram by definition rather than an undefined one.
MIN_BATCH = 3


def _require_gudhi():
    """gudhi is imported here rather than at module scope.

    Everything else in the repo runs without it: it is only the H1 term that needs a
    persistence backend, and that term is off in the recipe. Importing it lazily
    keeps ``--lambda_h1 0`` runs (i.e. all of them by default) working on an
    environment that never installed it, and turns a missing install into a message
    that says which flag asked for it.
    """
    try:
        import gudhi
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "The H1 persistence term needs the 'gudhi' package "
            "(pip install gudhi). It is only required when --lambda_h1 > 0."
        ) from exc
    return gudhi


def h1_persistence_pairs(dist: torch.Tensor) -> torch.Tensor:
    """Critical-edge indices of the finite H1 Vietoris-Rips pairs.

    Returns a ``[K, 4]`` long tensor whose rows are ``(b_i, b_j, d_i, d_j)``: the two
    endpoints of the edge whose arrival creates the cycle, then of the edge whose
    arrival completes the triangle that kills it. Reading ``dist`` at those two index
    pairs gives back exactly ``Dgm_1``.

    The 2-skeleton is built to its full extent (no ``max_edge_length``), so every
    1-cycle of a finite cloud is eventually filled and the diagram has no essential
    class -- there is nothing here that a diagonal-aware matching would have to pair
    against infinity.

    The complex is ``O(B^3)`` simplices, and that is the term's real cost. Measured
    forward+backward on this box, against ~3 ms for the H0 term at every size:
    B=32 ~7 ms, B=64 ~30 ms, B=128 ~430 ms, B=256 ~6 s. The default GeoODE batch of
    32 is comfortable; a batch of 128 or more makes this the step, so the H1 arm is
    a small-batch arm unless someone subsamples the cloud first.
    """
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError("dist must be a square [B, B] matrix.")

    if dist.shape[0] < MIN_BATCH:
        return torch.zeros((0, 4), dtype=torch.long, device=dist.device)

    gudhi = _require_gudhi()
    detached = dist.detach().to("cpu", torch.float64).numpy()

    simplex_tree = gudhi.RipsComplex(
        distance_matrix=detached
    ).create_simplex_tree(max_dimension=2)
    # persistence_dim_max=False: H2 of a 2-complex is not asked for, and skipping it
    # is the difference between reducing the triangles and reducing nothing above them.
    simplex_tree.compute_persistence(persistence_dim_max=False)

    # (regular H0, regular H1+, essential H0, essential H1+); [1][0] is regular H1.
    regular_higher = simplex_tree.flag_persistence_generators()[1]
    if not regular_higher or len(regular_higher[0]) == 0:
        return torch.zeros((0, 4), dtype=torch.long, device=dist.device)

    return torch.as_tensor(
        np.asarray(regular_higher[0], dtype=np.int64), dtype=torch.long
    ).to(dist.device)


def h1_diagram(embeddings: torch.Tensor, metric: Metric = "chord") -> torch.Tensor:
    """``Dgm_1`` of a batch as a differentiable ``[K, 2]`` tensor of ``(birth, death)``.

    A batch too small to close a cycle returns an empty diagram rather than raising:
    unlike H0, whose ``B - 1`` death times are a fixed-size vector that a one-row tail
    batch simply cannot provide, "no cycles" is a perfectly ordinary diagram and the
    matching against the teacher's is still defined -- every teacher cycle just falls
    onto the diagonal.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be [B, D], got {tuple(embeddings.shape)}")
    if embeddings.shape[0] < MIN_BATCH:
        return embeddings.new_zeros((0, 2))
    dist = pairwise_distance(embeddings, metric=metric)
    pairs = h1_persistence_pairs(dist)
    if pairs.shape[0] == 0:
        return dist.new_zeros((0, 2))
    births = dist[pairs[:, 0], pairs[:, 1]]
    deaths = dist[pairs[:, 2], pairs[:, 3]]
    return torch.stack([births, deaths], dim=-1)


def _diagonal_costs(diagram: torch.Tensor) -> torch.Tensor:
    """Squared L2 distance from each point to the diagonal: ``(d - b)^2 / 2``."""
    if diagram.shape[0] == 0:
        return diagram.new_zeros((0,))
    return (diagram[:, 1] - diagram[:, 0]) ** 2 / 2.0


def _matching(diagram_s: torch.Tensor, diagram_t: torch.Tensor) -> tuple:
    """The optimal partial matching, as index arrays into the augmented problem.

    The standard square lift: rows are the ``n`` student points followed by ``m``
    diagonal stand-ins for the teacher's, columns the ``m`` teacher points followed by
    ``n`` diagonal stand-ins for the student's. Student point ``i`` may only reach
    *its own* stand-in, which is what makes "matched to the diagonal" mean
    "projected onto the diagonal" rather than "matched to some other point's shadow";
    the forbidden entries carry a finite penalty rather than ``inf`` because
    ``linear_sum_assignment`` refuses a matrix it cannot see a feasible path through.
    The stand-in/stand-in block is free, so unmatched shadows cost nothing.
    """
    n, m = diagram_s.shape[0], diagram_t.shape[0]
    s = diagram_s.detach().to("cpu", torch.float64).numpy()
    t = diagram_t.detach().to("cpu", torch.float64).numpy()

    diag_s = ((s[:, 1] - s[:, 0]) ** 2) / 2.0
    diag_t = ((t[:, 1] - t[:, 0]) ** 2) / 2.0
    # Every point can always take its own diagonal, so no matching costs more than
    # this; one unit above it is therefore never worth taking.
    forbidden = float(diag_s.sum() + diag_t.sum()) + 1.0

    cost = np.zeros((n + m, m + n), dtype=np.float64)
    if n and m:
        cost[:n, :m] = ((s[:, None, :] - t[None, :, :]) ** 2).sum(-1)
    cost[:n, m:] = forbidden
    np.fill_diagonal(cost[:n, m:], diag_s)
    cost[n:, :m] = forbidden
    np.fill_diagonal(cost[n:, :m], diag_t)
    # cost[n:, m:] stays 0: two diagonal stand-ins matched to each other.

    return linear_sum_assignment(cost)


def wasserstein2_squared(
    diagram_s: torch.Tensor, diagram_t: torch.Tensor
) -> torch.Tensor:
    """``W_2^2`` between two persistence diagrams, differentiable in ``diagram_s``.

    The ground metric between diagram points is the Euclidean one and low-persistence
    points may be matched to the diagonal, so the two diagrams need not have the same
    number of features -- which they generally do not: the student and the teacher
    are two different clouds and their cycle counts are their own.

    The returned value is the sum over matched pairs, i.e. ``W_2`` squared as written,
    not an average over features. It therefore sits on a different scale from the
    H0 term, which is a mean over the ``B - 1`` death times; ``lambda_h1`` is what
    reconciles the two.
    """
    if diagram_s.ndim != 2 or diagram_s.shape[-1] != 2:
        raise ValueError(f"diagram_s must be [K, 2], got {tuple(diagram_s.shape)}")
    if diagram_t.ndim != 2 or diagram_t.shape[-1] != 2:
        raise ValueError(f"diagram_t must be [K, 2], got {tuple(diagram_t.shape)}")

    diagram_t = diagram_t.to(device=diagram_s.device, dtype=diagram_s.dtype)
    n, m = diagram_s.shape[0], diagram_t.shape[0]
    if n == 0 and m == 0:
        return diagram_s.new_zeros(())
    if n == 0:
        return _diagonal_costs(diagram_t).sum()
    if m == 0:
        return _diagonal_costs(diagram_s).sum()

    rows, cols = _matching(diagram_s, diagram_t)
    rows = torch.as_tensor(rows, dtype=torch.long, device=diagram_s.device)
    cols = torch.as_tensor(cols, dtype=torch.long, device=diagram_s.device)

    # Re-read the chosen matching off the live tensors. Three kinds of matched pair
    # survive with a non-zero cost; stand-in against stand-in contributes nothing.
    paired = (rows < n) & (cols < m)
    s_to_diag = (rows < n) & (cols >= m)
    t_to_diag = (rows >= n) & (cols < m)

    total = diagram_s.new_zeros(())
    if paired.any():
        difference = diagram_s[rows[paired]] - diagram_t[cols[paired]]
        total = total + (difference**2).sum()
    if s_to_diag.any():
        total = total + _diagonal_costs(diagram_s[rows[s_to_diag]]).sum()
    if t_to_diag.any():
        total = total + _diagonal_costs(diagram_t[cols[t_to_diag]]).sum()
    return total


def h1_loss_against_diagram(
    student_embeddings: torch.Tensor,
    teacher_diagram: torch.Tensor,
    metric: Metric = "chord",
) -> torch.Tensor:
    """The H1 term against a teacher diagram that was computed elsewhere.

    As with H0, the teacher side is a constant read from a frozen cache, so building
    it belongs in the collate -- in a DataLoader worker, overlapped with the previous
    step -- rather than on the training step's critical path. This is the entry point
    that takes the result.
    """
    if student_embeddings.ndim != 2:
        raise ValueError("student_embeddings must be [B, D].")
    return wasserstein2_squared(
        h1_diagram(student_embeddings, metric=metric), teacher_diagram
    )


def h1_topological_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    metric: Metric = "chord",
    chunk_size: int | None = None,
) -> torch.Tensor:
    """``W_2^2(Dgm_1(T), Dgm_1(Z))`` between a teacher and a student batch.

    Teacher and student may live in different ambient dimensions -- a diagram is a
    multiset of scalar pairs, and the only thing the two sides have to share is the
    B corresponding samples of the batch.

    ``chunk_size`` reads the batch as several smaller clouds instead of one (see
    :func:`src.criterions.h0_topological_loss.chunk_count`) and *averages* their
    ``W_2^2``. The average, rather than the sum, is what keeps the term's scale a
    property of the geometry instead of the number of chunks -- the same choice L_H0
    makes by being a mean over death times. Unlike H0 this also makes the term
    cheaper by a factor of ``n^2``: the 2-skeleton is ``O(B^3)``.
    """
    if student_embeddings.ndim != 2 or teacher_embeddings.ndim != 2:
        raise ValueError("student_embeddings and teacher_embeddings must be [B, D].")
    if student_embeddings.shape[0] != teacher_embeddings.shape[0]:
        raise ValueError("Teacher and student batch sizes must match.")

    student_chunks = split_chunks(student_embeddings, chunk_size)
    teacher_chunks = split_chunks(teacher_embeddings, chunk_size)
    losses = []
    for student_chunk, teacher_chunk in zip(student_chunks, teacher_chunks):
        with torch.no_grad():
            teacher_diagram = h1_diagram(teacher_chunk, metric=metric)
        losses.append(
            h1_loss_against_diagram(student_chunk, teacher_diagram, metric=metric)
        )
    return torch.stack(losses).mean()


class H1TopologicalLoss(nn.Module):
    """nn.Module wrapper."""

    def __init__(self, metric: Metric = "chord", chunk_size: int | None = None) -> None:
        super().__init__()
        self.metric = metric
        self.chunk_size = chunk_size

    def forward(
        self, student_embeddings: torch.Tensor, teacher_embeddings: torch.Tensor
    ) -> torch.Tensor:
        return h1_topological_loss(
            student_embeddings=student_embeddings,
            teacher_embeddings=teacher_embeddings,
            metric=self.metric,
            chunk_size=self.chunk_size,
        )
