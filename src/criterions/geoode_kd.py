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
        endpoint_loss: ``"cosine"`` is Eq. (36). ``"mse"`` is the
            sentence-transformers (<= v5.4) distillation baseline: squared error
            between the *unnormalised* pooled final state and the projected teacher
            target exactly as delivered (the distiller then skips the final norm(.)
            of Eq. 8), summed over dimensions and averaged over the batch so that it
            sits on the scale of the cosine term. It is a baseline, not part of the
            recipe.
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
        endpoint_loss: str = "cosine",
        lambda_gram: float = 0.0,
    ):
        super().__init__()
        if lambda_end < 0 or lambda_ctr < 0 or lambda_gram < 0:
            raise ValueError("lambda_end, lambda_ctr and lambda_gram must be non-negative")
        if endpoint_loss not in ("cosine", "mse", "procrustes"):
            raise ValueError(
                "endpoint_loss must be 'cosine', 'mse' or 'procrustes', "
                f"got {endpoint_loss!r}"
            )
        if endpoint_loss != "cosine" and target_projector is not None:
            raise ValueError(
                f"endpoint_loss={endpoint_loss!r} is a frozen-target control; it is not "
                "defined together with a learned target projector"
            )
        self.endpoint_loss_form = endpoint_loss
        self.lambda_gram = float(lambda_gram)

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

    @staticmethod
    def endpoint_mse(raw_final_state: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """Squared-error endpoint of the sentence-transformers baseline.

        Both arguments are read as they come: no normalisation on either side, so
        the student is also asked to reproduce the target's norm. The reduction is
        the per-sample squared distance ``||z_i - tau_i||^2`` averaged over the
        batch (``d_S x nn.MSELoss``), not the mean over all elements: on the sphere
        ``||z - tau||^2 = 2 (1 - cos)``, so this keeps the term on the same scale as
        the cosine endpoint and ``lambda_ctr`` means the same thing in both rows of
        the recipe ablation. With ``nn.MSELoss``'s element mean the term is ``d_S``
        times smaller and the InfoNCE regulariser silently takes over the run.
        """
        return ((raw_final_state - teacher) ** 2).sum(dim=-1).mean()

    @staticmethod
    def batch_procrustes(final_state: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """Per-step orthogonal re-alignment (the EdgePoint2 / Bhattarai family).

        Solves ``R_b = argmin_{R in O(d)} ||Z R - T||_F`` for the current batch in
        closed form and applies it to the student states. ``R_b`` is treated as a
        constant: at the optimum its contribution to the gradient vanishes (envelope
        theorem), so the student is trained on ``1 - <z R_b, tau>``. With a batch
        smaller than ``d`` the solution is only determined on the batch's span; the
        rest of ``R_b`` is an arbitrary orthogonal completion, which is exactly the
        arm's weakness and the reason it is a control rather than the recipe.
        """
        with torch.no_grad():
            cross = final_state.transpose(0, 1) @ teacher
            u, _, vh = torch.linalg.svd(cross.float(), full_matrices=True)
            rotation = (u @ vh).to(final_state.dtype)
        return final_state @ rotation

    @staticmethod
    def gram_loss(final_state: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """Pairwise-similarity matching (SP-KD / RKD family) on the batch: squared
        error between the student's and the target's off-diagonal Gram entries.
        The control for Prop. 3 -- with a fixed orthonormal interface it should be
        redundant with the endpoint term."""
        gram_s = final_state @ final_state.transpose(0, 1)
        gram_t = teacher @ teacher.transpose(0, 1)
        mask = ~torch.eye(gram_s.shape[0], dtype=torch.bool, device=gram_s.device)
        return ((gram_s - gram_t)[mask] ** 2).mean()

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
        hidden_states = list(hidden_states)
        states = self.layer_states(hidden_states, attention_mask)
        if not states:
            raise ValueError("hidden_states contained no supervised layer")

        teacher = teacher.to(states[-1].dtype)
        if self.endpoint_loss_form == "mse":
            # The baseline regresses the raw pooled state onto the raw target; the
            # normalised copies below still feed the cosine diagnostics.
            raw_final = _pool(hidden_states[-1], attention_mask, self.pooling).float()
            loss_end_mse = self.endpoint_mse(raw_final, teacher)
        teacher = self.normalize(teacher)
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

        if self.endpoint_loss_form == "mse":
            loss_end = loss_end_mse
        elif self.endpoint_loss_form == "procrustes":
            loss_end = self.endpoint_loss(self.batch_procrustes(states[-1], teacher), teacher)
        else:
            loss_end = self.endpoint_loss(states[-1], teacher)

        if self.lambda_gram > 0.0:
            loss_gram = self.gram_loss(states[-1], teacher)
        else:
            loss_gram = torch.zeros((), device=teacher.device, dtype=teacher.dtype)

        if self.lambda_ctr > 0.0 and second_view is not None:
            loss_ctr = self.contrastive_loss(
                student_view, self.normalize(second_view.float())
            )
        else:
            loss_ctr = torch.zeros((), device=teacher.device, dtype=teacher.dtype)

        total = (
            self.lambda_end * loss_end
            + self.lambda_ctr * loss_ctr
            + self.lambda_gram * loss_gram
        )

        with torch.no_grad():
            metrics = {
                "loss_total": float(total.detach()),
                "loss_end": float(loss_end.detach()),
                "loss_ctr": float(loss_ctr.detach()),
                "loss_gram": float(loss_gram.detach()),
                # Only the final layer is supervised; the shallow cosine is logged
                # next to it as a free sanity check on the untouched lower stack.
                "cos_first": float((states[0] * teacher).sum(dim=-1).mean()),
                "cos_final": float((states[-1] * teacher).sum(dim=-1).mean()),
            }

        return total, metrics
