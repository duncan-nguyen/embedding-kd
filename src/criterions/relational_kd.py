"""RKD: Relational Knowledge Distillation (Park, Kim, Lu & Cho, CVPR 2019).

The teacher supervises *relations between examples* instead of the examples
themselves, which is what lets it cross a dimensionality gap untouched: both
potentials are invariant to the width of the space they are measured in, so
nothing is fitted between the teacher's embedding and the student's.

    L_RKD-D = sum_{i,j}   l_delta( psi_D(t_i, t_j),      psi_D(s_i, s_j) )     (Eq. 4)
    L_RKD-A = sum_{i,j,k} l_delta( psi_A(t_i, t_j, t_k), psi_A(s_i, s_j, s_k) ) (Eq. 7)

psi_D is the pairwise Euclidean distance divided by the batch mean distance
(Eq. 3) — that normalisation is what makes the term scale-free — psi_A is the
cosine of the angle subtended at the middle point (Eq. 6), and l_delta is the
Huber loss, which keeps a single badly placed pair from dominating the batch.

This is the relational baseline the paper is stated against: it constrains the
geometry of the final layer only, and says nothing about the trajectory that
reaches it. Like GeoODE-KD it holds no parameters, so the two rows differ by
their objective rather than by what was added to the student.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RelationalKD(nn.Module):
    """Distance-wise plus angle-wise relational distillation (Eq. 8).

    Args:
        w_task: weight of the student's own task loss.
        w_dist: lambda_RKD-D, the weight of the distance-wise potential.
        w_angle: lambda_RKD-A, the weight of the angle-wise potential.
        huber_delta: delta of the Huber loss l_delta (the paper uses 1).
        normalize_student: L2-normalise the student embeddings before measuring
            relations. The teacher cache is already normalised and every
            benchmark scores cosine similarity, so this puts both sides of the
            comparison on the same sphere; off is the raw-Euclidean ablation.
        eps: floor applied under the square root of the pairwise distances, whose
            gradient is unbounded at zero.
    """

    def __init__(
        self,
        w_task: float = 1.0,
        w_dist: float = 25.0,
        w_angle: float = 50.0,
        huber_delta: float = 1.0,
        normalize_student: bool = True,
        eps: float = 1e-12,
    ):
        super().__init__()
        if huber_delta <= 0:
            raise ValueError(f"huber_delta must be positive, got {huber_delta}")
        self.w_task = float(w_task)
        self.w_dist = float(w_dist)
        self.w_angle = float(w_angle)
        self.huber_delta = float(huber_delta)
        self.normalize_student = bool(normalize_student)
        self.eps = float(eps)

    @staticmethod
    def pairwise_distance(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """||x_i - x_j||_2 for every pair in the batch.

        Expanded through the Gram matrix rather than a [B, B, d] difference, so
        the memory stays quadratic in the batch and independent of the width.
        """
        gram = x @ x.transpose(0, 1)
        squared = gram.diagonal().unsqueeze(0) + gram.diagonal().unsqueeze(1) - 2 * gram
        squared = squared.clamp(min=0.0)
        distance = squared.clamp(min=eps).sqrt()
        # The clamp leaves sqrt(eps) on the diagonal; the distance of a point to
        # itself is exactly zero, and the clone keeps the write out of autograd's
        # way.
        distance = distance.clone()
        distance.fill_diagonal_(0.0)
        return distance

    def _huber(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(
            prediction, target, reduction="mean", delta=self.huber_delta
        )

    def _normalized_distance(self, x: torch.Tensor) -> torch.Tensor:
        """psi_D of Eq. (3): pairwise distances over their batch mean."""
        distance = self.pairwise_distance(x, eps=self.eps)
        positive = distance[distance > 0]
        # A batch of identical sentences has no relations to match; returning the
        # zero matrix makes both potentials vanish instead of dividing by zero.
        if positive.numel() == 0:
            return distance
        return distance / (positive.mean() + self.eps)

    def _angles(self, x: torch.Tensor) -> torch.Tensor:
        """psi_A of Eq. (6): cos of the angle subtended at the middle point."""
        # differences[i, j] = x_j - x_i, so row i holds every edge leaving x_i
        # and the batched product below is the cos at the vertex x_i.
        differences = x.unsqueeze(0) - x.unsqueeze(1)  # [B, B, d]
        directions = F.normalize(differences, p=2, dim=-1, eps=self.eps)
        return torch.bmm(directions, directions.transpose(1, 2)).view(-1)

    def distance_loss(
        self, student: torch.Tensor, teacher: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            target = self._normalized_distance(teacher)
        return self._huber(self._normalized_distance(student), target)

    def angle_loss(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target = self._angles(teacher)
        return self._huber(self._angles(student), target)

    def forward(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        task_loss: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Args:
        student: pooled student embeddings, [B, d_S].
        teacher: cached teacher embeddings for the same rows, [B, d_T].
        task_loss: the student's own objective, added with weight ``w_task``.
        """
        if student.shape[0] != teacher.shape[0]:
            raise ValueError(
                f"Batch mismatch: student has {student.shape[0]} rows, "
                f"teacher has {teacher.shape[0]}"
            )

        # Both potentials are quadratic in inner products, and half precision
        # loses the small distances that carry the relations.
        student = student.float()
        teacher = teacher.float()
        if self.normalize_student:
            student = F.normalize(student, p=2, dim=-1, eps=self.eps)

        loss_dist = (
            self.distance_loss(student, teacher)
            if self.w_dist != 0
            else student.new_zeros(())
        )
        loss_angle = (
            self.angle_loss(student, teacher)
            if self.w_angle != 0
            else student.new_zeros(())
        )

        total = self.w_dist * loss_dist + self.w_angle * loss_angle
        metrics = {
            "loss_dist": float(loss_dist.detach()),
            "loss_angle": float(loss_angle.detach()),
        }
        if task_loss is not None:
            total = total + self.w_task * task_loss
            metrics["loss_task"] = float(task_loss.detach())
        metrics["loss_total"] = float(total.detach())
        return total, metrics
