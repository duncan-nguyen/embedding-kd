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


def h0_death_times(embeddings: torch.Tensor, metric: Metric = "chord", sort: bool = True) -> torch.Tensor:
    """Finite H0 Vietoris-Rips death times = MST edge weights."""
    deaths = mst_edge_weights(pairwise_distance(embeddings, metric=metric))
    return torch.sort(deaths).values if sort else deaths


def h0_topological_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    metric: Metric = "chord",
    squared: bool = True,
) -> torch.Tensor:
    """
    Differentiable H0 Wasserstein surrogate based on sorted finite death times.

    Teacher and student may have different ambient dimensions, but must contain
    the same B corresponding samples in the batch.
    """
    if student_embeddings.ndim != 2 or teacher_embeddings.ndim != 2:
        raise ValueError("student_embeddings and teacher_embeddings must be [B, D].")
    if student_embeddings.shape[0] != teacher_embeddings.shape[0]:
        raise ValueError("Teacher and student batch sizes must match.")

    with torch.no_grad():
        teacher_deaths = h0_death_times(teacher_embeddings, metric=metric, sort=True)

    return h0_loss_against_deaths(
        student_embeddings, teacher_deaths, metric=metric, squared=squared
    )


def h0_loss_against_deaths(
    student_embeddings: torch.Tensor,
    teacher_deaths: torch.Tensor,
    metric: Metric = "chord",
    squared: bool = True,
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
    student_deaths = h0_death_times(student_embeddings, metric=metric, sort=True)
    teacher_deaths = teacher_deaths.to(
        device=student_deaths.device, dtype=student_deaths.dtype
    )
    if teacher_deaths.shape != student_deaths.shape:
        raise ValueError(
            f"teacher diagram has {tuple(teacher_deaths.shape)} death times but the "
            f"student batch produced {tuple(student_deaths.shape)}; a diagram of "
            "B - 1 finite H0 death times belongs to the batch it was built from"
        )
    mse = torch.mean((student_deaths - teacher_deaths) ** 2)
    return mse if squared else torch.sqrt(mse + 1e-12)


class H0TopologicalLoss(nn.Module):
    """nn.Module wrapper."""
    def __init__(self, metric: Metric = "chord", squared: bool = True) -> None:
        super().__init__()
        self.metric = metric
        self.squared = squared

    def forward(self, student_embeddings: torch.Tensor, teacher_embeddings: torch.Tensor) -> torch.Tensor:
        return h0_topological_loss(
            student_embeddings=student_embeddings,
            teacher_embeddings=teacher_embeddings,
            metric=self.metric,
            squared=self.squared,
        )
