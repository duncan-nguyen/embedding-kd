"""GeoODE-KD: endpoint distillation of sentence embeddings against a frozen teacher map.

The objective is L_end + L_ctr (Eq. 38): the student's final layer is anchored on
the teacher endpoint (Eq. 36) and regularised by InfoNCE over two dropout views
(Eq. 37). Nothing else is in it.

    L_total = lambda_end * L_end + lambda_ctr * L_ctr

The teacher targets arrive already mapped into the student's dimension by the
frozen ``P_T = P_PCA R`` of Eq. (8) and are read under a stop-gradient, so the
supervision is a fixed point on the unit hypersphere rather than a moving one.

Nothing here is a module with weights: training adds no parameters and inference
is exactly the unmodified student encoder. The single exception is the opt-in
``target_projector`` baseline, which puts a *trainable* map where the frozen P_T
would be; its parameters exist during training only and are what that baseline is
meant to test.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from src.loss import info_nce
from src.pooling import mean_pooling


def _pool(
    hidden_state: torch.Tensor, attention_mask: torch.Tensor | None, method: str
) -> torch.Tensor:
    """Pool(.) of Eq. (7)."""
    if method == "cls":
        return hidden_state[:, 0, :]
    if method == "mean":
        if attention_mask is None:
            return hidden_state.mean(dim=1)
        return mean_pooling(hidden_state, attention_mask)
    raise ValueError(
        f"Unsupported student pooling method: {method!r}; expected 'cls' or 'mean'"
    )


class GeoODEKD(nn.Module):
    """Endpoint distillation with a contrastive regulariser (Eq. 38).

    Args:
        lambda_end: weight of the endpoint distillation loss L_end (Eq. 36).
        lambda_ctr: weight of the contrastive regulariser L_ctr (Eq. 37).
        contrastive_temperature: tau_c of Eq. (37).
        target_projector: optional trainable map standing where the frozen ``P_T``
            would be (:class:`src.target_projector.LearnedTargetProjector`). It is
            the learned-projector *baseline*, not part of the recipe: given one, the
            teacher and the student's final state are brought into a shared space by
            it, and its parameters are trained with the student. ``None`` (the
            default) means the targets arrive already mapped and frozen.
        pooling: pooling used to turn each layer's token states into a sentence vector.
        include_embedding_layer: treat the embedding output as depth 0 state as well.
            Off by default: the paper's L states are the L Transformer layers. Only
            the final state carries loss; this decides which state ``cos_first``
            reports.
    """

    def __init__(
        self,
        lambda_end: float = 1.0,
        lambda_ctr: float = 0.5,
        contrastive_temperature: float = 0.05,
        pooling: str = "cls",
        include_embedding_layer: bool = False,
        eps_norm: float = 1e-12,
        target_projector: nn.Module | None = None,
    ):
        super().__init__()
        if lambda_end < 0 or lambda_ctr < 0:
            raise ValueError("lambda_end and lambda_ctr must be non-negative (Eq. 38)")

        self.lambda_end = float(lambda_end)
        self.lambda_ctr = float(lambda_ctr)
        self.contrastive_temperature = float(contrastive_temperature)
        self.target_projector = target_projector
        self.pooling = pooling
        self.include_embedding_layer = bool(include_embedding_layer)
        self.eps_norm = float(eps_norm)

    # ------------------------------------------------------------------ geometry

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """norm(.) of Eq. (9), applied row-wise."""
        return F.normalize(x, p=2, dim=-1, eps=self.eps_norm)

    def layer_states(
        self,
        hidden_states: Sequence[torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        """Pool and normalise every supervised depth into Z^(l) (Eqs. 7, 10)."""
        states = list(hidden_states)
        if not self.include_embedding_layer and len(states) > 1:
            # hidden_states[0] is the embedding output, not a Transformer layer.
            states = states[1:]
        return [
            self.normalize(_pool(state, attention_mask, self.pooling).float())
            for state in states
        ]

    # ------------------------------------------------------------------ losses

    def endpoint_loss(
        self, final_state: torch.Tensor, teacher: torch.Tensor
    ) -> torch.Tensor:
        """Endpoint semantic distillation, Eq. (36)."""
        return (1.0 - (final_state * teacher).sum(dim=-1)).mean()

    def contrastive_loss(
        self, view_a: torch.Tensor, view_b: torch.Tensor
    ) -> torch.Tensor:
        """Unsupervised contrastive regulariser at the final layer, Eq. (37)."""
        loss, _ = info_nce(view_a, view_b, temperature=self.contrastive_temperature)
        return loss

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        hidden_states: Iterable[torch.Tensor],
        teacher: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        second_view: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute L_total of Eq. (38).

        Args:
            hidden_states: the student's per-layer token states, as returned by
                ``output_hidden_states=True`` (embedding output first).
            teacher: cached teacher targets already mapped by ``P_T`` and normalised,
                ``[B, d_S]``.
            attention_mask: student attention mask, needed for mean pooling.
            second_view: pooled (unnormalised) final representation of a second
                dropout view, used only for the contrastive term.
        """
        states = self.layer_states(hidden_states, attention_mask)
        if not states:
            raise ValueError("hidden_states contained no supervised layer")

        teacher = self.normalize(teacher.to(states[-1].dtype))
        # The contrastive term is a statement about the student's *own* space, so it
        # keeps reading the unmapped final state even when a learned projector moves
        # the endpoint comparison into a shared space.
        student_view = states[-1]
        if self.target_projector is not None:
            states, teacher = self.target_projector.align(states, teacher)
        if teacher.shape != states[-1].shape:
            raise ValueError(
                f"teacher targets have shape {tuple(teacher.shape)} but the student's "
                f"final state has shape {tuple(states[-1].shape)}; teacher embeddings "
                "must be projected into the student dimension before training"
            )

        loss_end = self.endpoint_loss(states[-1], teacher)

        if self.lambda_ctr > 0.0 and second_view is not None:
            loss_ctr = self.contrastive_loss(
                student_view, self.normalize(second_view.float())
            )
        else:
            loss_ctr = torch.zeros((), device=teacher.device, dtype=teacher.dtype)

        total = self.lambda_end * loss_end + self.lambda_ctr * loss_ctr

        with torch.no_grad():
            metrics = {
                "loss_total": float(total.detach()),
                "loss_end": float(loss_end.detach()),
                "loss_ctr": float(loss_ctr.detach()),
                # Only the final layer is supervised; the shallow cosine is logged
                # next to it as a free sanity check on the untouched lower stack.
                "cos_first": float((states[0] * teacher).sum(dim=-1).mean()),
                "cos_final": float((states[-1] * teacher).sum(dim=-1).mean()),
            }

        return total, metrics
