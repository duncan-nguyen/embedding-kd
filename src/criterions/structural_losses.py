"""Matched-compute structural controls for the H0 persistence loss.

Each control reduces a point cloud to a small collection of scalar distances and
matches that collection with mean squared error.  The default sample counts are
chosen to stay close to H0's ``B - 1`` finite death times:

* ``sorted_pairwise`` samples ``B - 1`` fixed pairwise distances and compares
  their independently sorted distributions;
* ``teacher_mst`` selects the teacher's ``B - 1`` MST edges and compares the
  student's distances on those same sample pairs;
* ``knn_distribution`` compares independently sorted directed kNN distances
  (``B`` values at the default ``k=1``).

Teacher statistics can be built in a DataLoader worker and passed to
``structural_loss_against_target``.  This mirrors the existing H0 fast path and
keeps frozen-teacher work off the optimisation step's critical path.
"""

from __future__ import annotations

from typing import Literal

import torch

from src.criterions.h0_topological_loss import (
    Metric,
    mst_edge_indices,
    pairwise_distance,
    split_chunks,
)

StructuralLoss = Literal["sorted_pairwise", "teacher_mst", "knn_distribution"]
STRUCTURAL_LOSSES = ("h0", "sorted_pairwise", "teacher_mst", "knn_distribution")


def _sampled_pair_indices(rows: int, device: torch.device) -> torch.Tensor:
    """Deterministic, well-spread ``B - 1`` entries of the upper triangle."""
    if rows < 2:
        raise ValueError("structural distances require at least two points")
    pairs = torch.triu_indices(rows, rows, offset=1, device=device).transpose(0, 1)
    count = rows - 1
    positions = torch.linspace(
        0, pairs.shape[0] - 1, count, device=device
    ).round().long()
    return pairs[positions]


def _knn_values(dist: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        raise ValueError(f"structural_knn_k must be positive, got {k}")
    effective_k = min(int(k), dist.shape[0] - 1)
    masked = dist.masked_fill(
        torch.eye(dist.shape[0], dtype=torch.bool, device=dist.device), float("inf")
    )
    neighbours = masked.topk(effective_k, largest=False, dim=1).values.flatten()
    return torch.sort(neighbours).values


def _target_for_chunk(
    embeddings: torch.Tensor,
    kind: StructuralLoss,
    metric: Metric,
    knn_k: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    dist = pairwise_distance(embeddings, metric=metric)
    if kind == "sorted_pairwise":
        pairs = _sampled_pair_indices(dist.shape[0], dist.device)
        values = torch.sort(dist[pairs[:, 0], pairs[:, 1]]).values
        return values, None
    if kind == "teacher_mst":
        edge_i, edge_j = mst_edge_indices(dist)
        edges = torch.stack([edge_i, edge_j], dim=-1)
        return dist[edge_i, edge_j], edges
    if kind == "knn_distribution":
        return _knn_values(dist, knn_k), None
    raise ValueError(f"unknown structural loss {kind!r}")


def structural_target(
    teacher_embeddings: torch.Tensor,
    kind: StructuralLoss,
    *,
    metric: Metric = "chord",
    chunk_size: int | None = None,
    knn_k: int = 1,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reduce a frozen teacher batch to values and optional teacher-MST edges."""
    if teacher_embeddings.ndim != 2:
        raise ValueError("teacher_embeddings must have shape [B, D]")
    chunks = split_chunks(teacher_embeddings, chunk_size)
    if any(chunk.shape[0] < 2 for chunk in chunks):
        raise ValueError("structural distances require at least two points")
    targets = [
        _target_for_chunk(chunk, kind=kind, metric=metric, knn_k=knn_k)
        for chunk in chunks
    ]
    values = [target[0] for target in targets]
    edges = [target[1] for target in targets]
    if len(targets) == 1:
        return values[0], edges[0]
    stacked_edges = (
        None
        if edges[0] is None
        else torch.stack([edge for edge in edges if edge is not None])
    )
    return torch.stack(values), stacked_edges


def _student_values_for_chunk(
    embeddings: torch.Tensor,
    kind: StructuralLoss,
    edges: torch.Tensor | None,
    metric: Metric,
    knn_k: int,
) -> torch.Tensor:
    dist = pairwise_distance(embeddings, metric=metric)
    if kind == "sorted_pairwise":
        pairs = _sampled_pair_indices(dist.shape[0], dist.device)
        return torch.sort(dist[pairs[:, 0], pairs[:, 1]]).values
    if kind == "teacher_mst":
        if edges is None:
            raise ValueError("teacher_mst requires the teacher edge indices")
        return dist[edges[:, 0], edges[:, 1]]
    if kind == "knn_distribution":
        return _knn_values(dist, knn_k)
    raise ValueError(f"unknown structural loss {kind!r}")


def structural_loss_against_target(
    student_embeddings: torch.Tensor,
    teacher_values: torch.Tensor,
    kind: StructuralLoss,
    *,
    teacher_edges: torch.Tensor | None = None,
    metric: Metric = "chord",
    chunk_size: int | None = None,
    knn_k: int = 1,
) -> torch.Tensor:
    """MSE against teacher statistics precomputed for this exact batch."""
    if student_embeddings.ndim != 2:
        raise ValueError("student_embeddings must have shape [B, D]")
    chunks = split_chunks(student_embeddings, chunk_size)
    student_values = []
    for index, chunk in enumerate(chunks):
        edges = teacher_edges if len(chunks) == 1 else (
            None if teacher_edges is None else teacher_edges[index]
        )
        student_values.append(
            _student_values_for_chunk(chunk, kind, edges, metric, knn_k)
        )
    values = student_values[0] if len(chunks) == 1 else torch.stack(student_values)
    target = teacher_values.to(device=values.device, dtype=values.dtype)
    if target.shape != values.shape:
        raise ValueError(
            f"teacher structural target has shape {tuple(target.shape)} but the "
            f"student produced {tuple(values.shape)}"
        )
    return torch.mean((values - target) ** 2)


def structural_distribution_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    kind: StructuralLoss,
    *,
    metric: Metric = "chord",
    chunk_size: int | None = None,
    knn_k: int = 1,
) -> torch.Tensor:
    """Direct path used when no precomputed teacher target is available."""
    if student_embeddings.shape[0] != teacher_embeddings.shape[0]:
        raise ValueError("teacher and student batch sizes must match")
    with torch.no_grad():
        values, edges = structural_target(
            teacher_embeddings,
            kind,
            metric=metric,
            chunk_size=chunk_size,
            knn_k=knn_k,
        )
    return structural_loss_against_target(
        student_embeddings,
        values,
        kind,
        teacher_edges=edges,
        metric=metric,
        chunk_size=chunk_size,
        knn_k=knn_k,
    )
