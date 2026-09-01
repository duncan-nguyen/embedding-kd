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
import torch.nn.functional as F
from torch import nn

from src.criterions.h0_topological_loss import (
    Metric,
    h0_loss_against_deaths,
    h0_topological_loss,
)
from src.criterions.h1_topological_loss import (
    MIN_BATCH as H1_MIN_BATCH,
    h1_loss_against_diagram,
    h1_topological_loss,
)
from src.loss import info_nce
from src.metrics import scalar_metrics
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
        lambda_topo: weight of the topological term
            ``L_topo = L_H0 + lambda_h1 * L_H1``. It matches the *shape* of the batch
            rather than any individual point, so it is invariant to the width of the
            space and reads the teacher's own geometry when ``teacher_topo`` is
            given. 0 is the recipe.
        lambda_h1: weight of the H1 (cycle) half of ``L_topo``: the squared
            2-Wasserstein distance between the teacher's and the student's
            1-dimensional persistence diagrams, low-persistence features matched to
            the diagonal. 0 -- the default -- leaves ``L_topo`` the pure H0 term it
            was, and the H1 machinery (and its gudhi dependency) is never touched.
            The two halves are on different scales by construction -- ``L_H0`` is a
            mean over ``B - 1`` death times, ``W_2^2`` is a sum over matched cycles --
            so this weight carries that ratio as well as the relative importance.
        topo_metric: ground metric of both diagrams on the unit sphere: ``"chord"``
            (Euclidean), ``"angular"`` (geodesic) or ``"cosine"``.
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
        lambda_topo: float = 0.0,
        lambda_h1: float = 0.0,
        topo_metric: Metric = "chord",
    ):
        super().__init__()
        if (
            lambda_end < 0
            or lambda_ctr < 0
            or lambda_gram < 0
            or lambda_topo < 0
            or lambda_h1 < 0
        ):
            raise ValueError(
                "lambda_end, lambda_ctr, lambda_gram, lambda_topo and lambda_h1 "
                "must be non-negative"
            )
        if topo_metric not in ("chord", "angular", "cosine"):
            raise ValueError(
                f"topo_metric must be 'chord', 'angular' or 'cosine', got {topo_metric!r}"
            )
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
        self.lambda_topo = float(lambda_topo)
        self.lambda_h1 = float(lambda_h1)
        self.topo_metric = topo_metric

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

    def endpoint_states(
        self,
        hidden_states: Sequence[torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        """``[Z^(1), Z^(L)]`` -- the only two depths anything downstream reads.

        The objective is an endpoint one: every loss term reads ``Z^(L)`` and
        nothing else, and ``Z^(1)`` exists only for the ``cos_first`` diagnostic.
        :meth:`layer_states` pools and normalises all L depths, which on a 12-layer
        student is ten sentence-poolings per step whose results are then dropped --
        and under mean pooling each one is a full ``[B, L, d]`` reduction.

        A single supervised depth returns one state, so ``states[0] is states[-1]``
        exactly as it does for :meth:`layer_states`.
        """
        states = list(hidden_states)
        if not self.include_embedding_layer and len(states) > 1:
            states = states[1:]
        if not states:
            return []
        endpoints = states if len(states) == 1 else [states[0], states[-1]]
        return [
            self.normalize(_pool(state, attention_mask, self.pooling).float())
            for state in endpoints
        ]

    # ------------------------------------------------------------------ losses

    def endpoint_loss(
        self, final_state: torch.Tensor, teacher: torch.Tensor
    ) -> torch.Tensor:
        """Endpoint semantic distillation, Eq. (36)."""
        return (1.0 - (final_state * teacher).sum(dim=-1)).mean()

    @staticmethod
    def endpoint_mse(
        raw_final_state: torch.Tensor, teacher: torch.Tensor
    ) -> torch.Tensor:
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
    def batch_procrustes(
        final_state: torch.Tensor, teacher: torch.Tensor
    ) -> torch.Tensor:
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

    def topological_loss_against_deaths(
        self, final_state: torch.Tensor, teacher_deaths: torch.Tensor
    ) -> torch.Tensor:
        """The H0 term against a teacher diagram built outside the training step.

        The teacher side of this loss is a constant, so where it is computed is a
        scheduling question, not a modelling one; see
        :func:`src.criterions.h0_topological_loss.h0_loss_against_deaths`.
        """
        return h0_loss_against_deaths(
            final_state, teacher_deaths, metric=self.topo_metric
        )

    def topological_loss(
        self, final_state: torch.Tensor, teacher: torch.Tensor
    ) -> torch.Tensor:
        """H0 persistence matching between the two batches.

        Both diagrams are read off the batch's minimum spanning tree, so this is a
        statement about the connectivity of the point cloud and nothing else: it
        needs no correspondence between the two spaces' axes and no equality of
        their dimensions, which is why ``teacher`` here may be the *unprojected*
        teacher cache. The teacher side is a constant (no_grad inside).
        """
        return h0_topological_loss(final_state, teacher, metric=self.topo_metric)

    def h1_loss_against_diagram(
        self, final_state: torch.Tensor, teacher_diagram: torch.Tensor
    ) -> torch.Tensor:
        """The H1 term against a teacher diagram built outside the training step."""
        return h1_loss_against_diagram(
            final_state, teacher_diagram, metric=self.topo_metric
        )

    def h1_loss(
        self, final_state: torch.Tensor, teacher: torch.Tensor
    ) -> torch.Tensor:
        """``W_2^2`` between the two batches' 1-dimensional persistence diagrams.

        Where the H0 term reads how the cloud merges, this reads what it encloses:
        the birth and death of every 1-cycle of the Vietoris-Rips filtration, matched
        against the teacher's with low-persistence cycles allowed to fall onto the
        diagonal. Like H0 it is a statement about scalar filtration values only, so it
        needs no correspondence between the two spaces' axes and ``teacher`` here may
        be the *unprojected* teacher cache. The teacher side is a constant.
        """
        return h1_topological_loss(final_state, teacher, metric=self.topo_metric)

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
        teacher_topo: torch.Tensor | None = None,
        teacher_deaths: torch.Tensor | None = None,
        teacher_h1: torch.Tensor | None = None,
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
            teacher_topo: optional cached teacher embeddings in the teacher's *own*
                dimension ``[B, d_T]``, read only by the topological terms. ``None``
                falls back to the projected targets, which makes them a statement
                about the shape P_T left behind rather than the teacher's own.
            teacher_deaths: the same H0 term's teacher side, already reduced to its
                ``[B - 1]`` sorted death times by the collate. It supersedes
                ``teacher_topo`` when given -- it is what ``teacher_topo`` would have
                been turned into here, computed off the training step's critical path.
            teacher_h1: the H1 term's teacher side, already reduced by the collate to
                its ``[K, 2]`` diagram of ``(birth, death)`` pairs, for the same
                reason. It supersedes ``teacher_topo`` for that term when given.
        """
        hidden_states = list(hidden_states)
        states = self.endpoint_states(hidden_states, attention_mask)
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
            loss_end = self.endpoint_loss(
                self.batch_procrustes(states[-1], teacher), teacher
            )
        else:
            loss_end = self.endpoint_loss(states[-1], teacher)

        if self.lambda_gram > 0.0:
            loss_gram = self.gram_loss(states[-1], teacher)
        else:
            loss_gram = torch.zeros((), device=teacher.device, dtype=teacher.dtype)

        zero = torch.zeros((), device=teacher.device, dtype=teacher.dtype)
        # ``teacher_topo`` is the teacher cache in its *own* d_T; falling back to the
        # projected target would supervise the run through P_T twice over. Both
        # halves of L_topo read the same cloud.
        topo_target = teacher if teacher_topo is None else teacher_topo.float()

        # A single-sample tail batch has no MST, so the term is simply absent there.
        if self.lambda_topo > 0.0 and states[-1].shape[0] >= 2:
            if teacher_deaths is not None:
                loss_h0 = self.topological_loss_against_deaths(
                    states[-1], teacher_deaths
                )
            else:
                loss_h0 = self.topological_loss(states[-1], topo_target)
        else:
            loss_h0 = zero

        # A 1-cycle needs three points, so H1 sits out one batch more than H0 does.
        if (
            self.lambda_topo > 0.0
            and self.lambda_h1 > 0.0
            and states[-1].shape[0] >= H1_MIN_BATCH
        ):
            if teacher_h1 is not None:
                loss_h1 = self.h1_loss_against_diagram(states[-1], teacher_h1)
            else:
                loss_h1 = self.h1_loss(states[-1], topo_target)
        else:
            loss_h1 = zero

        # L_topo = L_H0 + lambda_1 L_H1; lambda_topo below weights the pair.
        loss_topo = loss_h0 + self.lambda_h1 * loss_h1

        if self.lambda_ctr > 0.0 and second_view is not None:
            loss_ctr = self.contrastive_loss(
                student_view, self.normalize(second_view.float())
            )
        else:
            loss_ctr = zero

        total = (
            self.lambda_end * loss_end
            + self.lambda_ctr * loss_ctr
            + self.lambda_gram * loss_gram
            + self.lambda_topo * loss_topo
        )

        with torch.no_grad():
            metrics = scalar_metrics(
                loss_total=total,
                loss_end=loss_end,
                loss_ctr=loss_ctr,
                loss_gram=loss_gram,
                loss_topo=loss_topo,
                loss_h0=loss_h0,
                loss_h1=loss_h1,
                # Only the final layer is supervised; the shallow cosine is logged
                # next to it as a free sanity check on the untouched lower stack.
                cos_first=(states[0] * teacher).sum(dim=-1).mean(),
                cos_final=(states[-1] * teacher).sum(dim=-1).mean(),
            )

        return total, metrics
