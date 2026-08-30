"""Differentiable H0 topological loss for teacher-student embedding distillation."""
from __future__ import annotations
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def mst_edge_weights(dist: torch.Tensor) -> torch.Tensor:
    """Prim MST. Selection is detached; selected edge weights remain differentiable."""
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError("dist must be a square [B, B] matrix.")
    B = dist.shape[0]
    if B < 2:
        raise ValueError("MST requires at least two points.")

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
        i = parent[j]
        edge_i.append(i)
        edge_j.append(j)
        in_tree[j] = True

        better = (d_select[j] < min_cost) & (~in_tree)
        min_cost = torch.where(better, d_select[j], min_cost)
        parent = torch.where(better, j.expand_as(parent), parent)

    edge_i = torch.stack(edge_i)
    edge_j = torch.stack(edge_j)
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

    student_deaths = h0_death_times(student_embeddings, metric=metric, sort=True)
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
