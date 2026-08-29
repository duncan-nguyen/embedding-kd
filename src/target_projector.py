"""Trainable target maps: the adaptive counterpart of the frozen ``P_T``.

``teacher_projection.py`` fits the map once, before training, and freezes it. The
prevailing practice is the opposite: a linear map learned jointly with the student,
either mapping the teacher down (EMO, sentence-transformers v5.5) or the student up
(TALAS, LEAF, jina-v5). This module is that practice, implemented as a baseline so
the comparison is a controlled one -- same corpus, same schedule, same objective
``L_end + L_ctr``, same student init, differing only in whether the map adapts.

The two directions are not interchangeable:

* ``t2s`` maps the teacher into ``d_S`` and compares there. The map has to discard
  ``d_T - d_S`` dimensions, and it is free to choose *which* ones by whatever makes
  the loss small -- which is exactly the shortcut the frozen recipe argues against
  (Bhattarai 2509.25253 Thm 2: a projected loss of zero does not imply the student's
  Gram matrix matches the teacher's, unless the map is right-orthogonal).
* ``s2t`` maps the student up into ``d_T`` and compares in the teacher's own space,
  so no teacher information is discarded. It can still cheat, but differently: the
  extra ``d_T - d_S`` dimensions give the map slack to absorb error the student
  never has to learn.

Neither map survives into inference: the deployed model is the student encoder, and
these parameters exist only during training. That is what makes them a fair control
rather than a bigger model -- they change the *supervision*, not the artefact.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

DIRECTIONS = ("t2s", "s2t")


class LearnedTargetProjector(nn.Module):
    """A single linear layer standing where ``P_T = P_PCA R`` would be.

    Args:
        teacher_dim: ``d_T`` of the cached teacher embeddings.
        student_dim: ``d_S`` of the student.
        direction: ``"t2s"`` maps ``d_T -> d_S``, ``"s2t"`` maps ``d_S -> d_T``.
        eps: normalisation epsilon, shared with the criterion.

    No bias: a shift would move the targets off the sphere every downstream metric
    is measured on, and none of the maps being compared has one. Initialised the way
    a projector is initialised in practice -- not from the teacher's spectrum, since
    a spectral init would hand the baseline the very thing under test.
    """

    def __init__(
        self,
        teacher_dim: int,
        student_dim: int,
        direction: str = "t2s",
        eps: float = 1e-12,
    ):
        super().__init__()
        if direction not in DIRECTIONS:
            raise ValueError(
                f"unknown direction {direction!r}; expected one of {', '.join(DIRECTIONS)}"
            )
        if teacher_dim <= 0 or student_dim <= 0:
            raise ValueError("teacher_dim and student_dim must be positive")
        self.direction = direction
        self.teacher_dim = int(teacher_dim)
        self.student_dim = int(student_dim)
        self.eps = float(eps)
        self.linear = (
            nn.Linear(teacher_dim, student_dim, bias=False)
            if direction == "t2s"
            else nn.Linear(student_dim, teacher_dim, bias=False)
        )

    @property
    def comparison_dim(self) -> int:
        """The dimension the endpoint loss is measured in."""
        return self.student_dim if self.direction == "t2s" else self.teacher_dim

    def align(
        self, states: Sequence[torch.Tensor], teacher: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Move the student trajectory and the teacher into one shared space.

        Returns ``(states, teacher)`` already normalised and of equal width, so every
        teacher-facing term of the objective is computed exactly as it is for a
        frozen map -- the only difference being that this map carries gradients.

        ``t2s`` leaves the student alone and brings the teacher down. ``s2t`` maps
        every layer state up: pooling is linear, so mapping the pooled vector is the
        same as pooling the mapped tokens, and doing it per layer keeps the depth
        diagnostics and the (opt-in) per-transition terms defined in the space the
        comparison actually happens in.
        """
        if self.direction == "t2s":
            return list(states), self._normalize(self.linear(teacher))
        return [self._normalize(self.linear(state)) for state in states], teacher

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x.float(), p=2, dim=-1, eps=self.eps).to(x.dtype)

    def extra_repr(self) -> str:
        arrow = (
            f"{self.teacher_dim} -> {self.student_dim}"
            if self.direction == "t2s"
            else f"{self.student_dim} -> {self.teacher_dim}"
        )
        return f"direction={self.direction}, {arrow}"
