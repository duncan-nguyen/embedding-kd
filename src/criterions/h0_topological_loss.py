"""Differentiable H0 topological loss for teacher-student embedding distillation."""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.csgraph import minimum_spanning_tree

Metric = Literal["chord", "angular", "cosine"]


def pairwise_distance(x: torch.Tensor, metric: Metric = "chord", eps: float = 1e-7) -> torch.Tensor:
    """Pairwise distances for row-wise embeddings x with shape [B, D]."""
    if x.ndim != 2:
        raise ValueError(f"x must have shape [B, D], got {tuple(x.shape)}")
    if x.shape[0] < 2:
        raise ValueError("H0 persistence requires batch size B >= 2.")

    # The death times are small differences of similarities near 1, so the Gram
    # matrix is built in fp32 even inside the training loop's autocast region:
    # in fp16 the resolution near 1 is ~5e-4 and 2 - 2 cos loses most of its
    # significant digits before the sqrt.
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = F.normalize(x.float(), p=2, dim=-1)
        sim = x @ x.T
        # sim[i, j] and sim[j, i] are the same dot product summed in different
        # orders, so the matmul returns them differing in the last bit -- and near
        # the clamp floor below (two near-identical rows, which a corpus with
        # repeated sentences really produces) that last bit is the difference
        # between a clamped and an unclamped distance. An asymmetric distance
        # matrix is not a metric graph at all: the MST it defines depends on which
        # triangle the algorithm happens to read. Symmetrising costs one [B, B] add
        # and makes the diagram a property of the point cloud again.
        sim = 0.5 * (sim + sim.T)

    if metric == "angular":
        sim_safe = sim.clamp(-1.0 + eps, 1.0 - eps)
        dist = torch.acos(sim_safe)
    elif metric == "chord":
        sim_safe = sim.clamp(-1.0, 1.0)
        dist = torch.sqrt((2.0 - 2.0 * sim_safe).clamp_min(eps))
    elif metric == "cosine":
        sim_safe = sim.clamp(-1.0, 1.0)
        dist = 1.0 - sim_safe
    else:
        raise ValueError("metric must be 'chord', 'angular', or 'cosine'.")

    eye = torch.eye(x.shape[0], device=x.device, dtype=torch.bool)
    return dist.masked_fill(eye, 0.0)


def chunk_count(n_rows: int, chunk_size: int | None) -> int:
    """How many disjoint chunks of ``chunk_size`` rows a batch of ``n_rows`` gives.

    A persistence diagram is a statement about a *point cloud*, and which cloud that
    is has so far been decided by the optimiser's batch size: one diagram per batch,
    ``B - 1`` death times. The two are independent choices, though -- the batch size
    sets the gradient's variance, the cloud size sets the scale at which the
    filtration reads the geometry -- so the topological terms can be given their own
    ``chunk_size`` and the batch split into ``n = B // chunk_size`` clouds whose
    losses are averaged.

    ``None`` or 0 keeps the old behaviour exactly: one diagram per batch. So does a
    ``chunk_size`` the batch cannot hold (including the tail batch of an epoch),
    which is the whole batch rather than an error.
    """
    if not chunk_size:
        return 1
    if chunk_size < 2:
        raise ValueError(
            f"chunk_size must be 0/None (whole batch) or >= 2, got {chunk_size}"
        )
    if chunk_size >= n_rows:
        return 1
    return n_rows // chunk_size


def split_chunks(x: torch.Tensor, chunk_size: int | None) -> list[torch.Tensor]:
    """The batch's rows as equal-sized clouds, in order.

    The rows arrive from a shuffled loader, so contiguous blocks are already random
    subsets and no further permutation is needed -- what matters is only that the
    teacher and the student split *identically*, which they do because both sides
    apply this same rule to the same batch.

    The ``B mod chunk_size`` trailing rows are dropped from the topological term (and
    from that term only). Every chunk therefore has the same number of death times,
    which is what lets the teacher's diagrams stack into one ``[n, chunk_size - 1]``
    tensor; folding the remainder into the last chunk would instead compare clouds of
    two different sizes, and the filtration scale is exactly what the size sets.
    """
    n = chunk_count(x.shape[0], chunk_size)
    if n == 1:
        return [x]
    return [x[k * chunk_size : (k + 1) * chunk_size] for k in range(n)]


def _mst_edge_indices(dist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Endpoints of the MST edges of a dense distance matrix, as index tensors.

    The selection is a discrete, non-differentiable decision, which is why the old
    Prim loop already took it on ``dist.detach()``. Taking it in scipy instead of in
    ``B - 1`` sequential torch steps is the same decision at a fraction of the cost:
    the loop issued about eight tiny kernels per iteration -- roughly 250 launches
    for a batch of 32, all of them dependent, so neither the GPU nor the CPU could
    do anything else while they drained.

    scipy reads 0 as "no edge", and ``1 - cos`` is exactly 0 for two identical rows
    (a real case here: a corpus repeats sentences, and the cache maps equal texts to
    equal vectors). Every weight is therefore shifted above zero first. Adding a
    constant to all edges is strictly monotone, so Kruskal selects the same edges;
    the shift is undone by reading the weights from the unshifted matrix.

    Ties can leave more than one valid MST, and which one is returned may differ
    from the old loop's. The loss is unaffected: every MST of a graph has the same
    sorted multiset of edge weights, which is all the H0 diagram is.
    """
    detached = dist.detach().to("cpu", torch.float64).numpy()
    shifted = detached + 1.0
    # Upper triangle only: the graph is undirected and this is the half scipy reads.
    upper = torch.triu(torch.ones(dist.shape, dtype=torch.bool), diagonal=1).numpy()
    tree = minimum_spanning_tree(shifted * upper).tocoo()
    rows = torch.as_tensor(tree.row, dtype=torch.long, device=dist.device)
    cols = torch.as_tensor(tree.col, dtype=torch.long, device=dist.device)
    return rows, cols


def mst_edge_weights(dist: torch.Tensor) -> torch.Tensor:
    """MST edge weights. Selection is detached; the weights stay differentiable."""
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError("dist must be a square [B, B] matrix.")
    B = dist.shape[0]
    if B < 2:
        raise ValueError("MST requires at least two points.")

    edge_i, edge_j = _mst_edge_indices(dist)
    return dist[edge_i, edge_j]


def h0_death_times(
    embeddings: torch.Tensor,
    metric: Metric = "chord",
    sort: bool = True,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Finite H0 Vietoris-Rips death times = MST edge weights.

    ``[B - 1]`` for the batch as one cloud, or ``[n, chunk_size - 1]`` when
    ``chunk_size`` splits it into ``n`` clouds (see :func:`chunk_count`).
    """
    def deaths_of(x: torch.Tensor) -> torch.Tensor:
        deaths = mst_edge_weights(pairwise_distance(x, metric=metric))
        return torch.sort(deaths).values if sort else deaths

    chunks = split_chunks(embeddings, chunk_size)
    if len(chunks) == 1:
        return deaths_of(chunks[0])
    return torch.stack([deaths_of(chunk) for chunk in chunks], dim=0)


def h0_topological_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    metric: Metric = "chord",
    squared: bool = True,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """
    Differentiable H0 Wasserstein surrogate based on sorted finite death times.

    Teacher and student may have different ambient dimensions, but must contain
    the same B corresponding samples in the batch. ``chunk_size`` splits that batch
    into equal clouds and averages their losses; ``None`` reads the batch as one.
    """
    if student_embeddings.ndim != 2 or teacher_embeddings.ndim != 2:
        raise ValueError("student_embeddings and teacher_embeddings must be [B, D].")
    if student_embeddings.shape[0] != teacher_embeddings.shape[0]:
        raise ValueError("Teacher and student batch sizes must match.")

    with torch.no_grad():
        teacher_deaths = h0_death_times(
            teacher_embeddings, metric=metric, sort=True, chunk_size=chunk_size
        )

    return h0_loss_against_deaths(
        student_embeddings,
        teacher_deaths,
        metric=metric,
        squared=squared,
        chunk_size=chunk_size,
    )


def h0_loss_against_deaths(
    student_embeddings: torch.Tensor,
    teacher_deaths: torch.Tensor,
    metric: Metric = "chord",
    squared: bool = True,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """The same loss against a teacher diagram that was computed elsewhere.

    The teacher side is a constant: it reads a frozen cache under ``no_grad`` and
    depends on nothing the step computes. It therefore does not have to be built on
    the GPU inside the training step at all -- the batch's rows are already known
    when the batch is collated, so the diagram can be built there, in a DataLoader
    worker, overlapped with the previous step. This is the entry point that takes
    the result.
    """
    if student_embeddings.ndim != 2:
        raise ValueError("student_embeddings must be [B, D].")
    student_deaths = h0_death_times(
        student_embeddings, metric=metric, sort=True, chunk_size=chunk_size
    )
    teacher_deaths = teacher_deaths.to(
        device=student_deaths.device, dtype=student_deaths.dtype
    )
    if teacher_deaths.shape != student_deaths.shape:
        raise ValueError(
            f"teacher diagram has {tuple(teacher_deaths.shape)} death times but the "
            f"student batch produced {tuple(student_deaths.shape)}; a diagram of "
            "B - 1 finite H0 death times belongs to the batch it was built from, "
            "and to the chunk size that batch was split with"
        )
    mse = torch.mean((student_deaths - teacher_deaths) ** 2)
    return mse if squared else torch.sqrt(mse + 1e-12)


class H0TopologicalLoss(nn.Module):
    """nn.Module wrapper."""
    def __init__(
        self,
        metric: Metric = "chord",
        squared: bool = True,
        chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.metric = metric
        self.squared = squared
        self.chunk_size = chunk_size

    def forward(self, student_embeddings: torch.Tensor, teacher_embeddings: torch.Tensor) -> torch.Tensor:
        return h0_topological_loss(
            student_embeddings=student_embeddings,
            teacher_embeddings=teacher_embeddings,
            metric=self.metric,
            squared=self.squared,
            chunk_size=self.chunk_size,
        )
