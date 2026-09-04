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
    chunk_count,
    split_chunks,
    Metric,
    h0_death_times,
    h0_loss_against_deaths,
    h0_topological_loss,
)
from src.criterions.h1_topological_loss import (
    MIN_BATCH as H1_MIN_BATCH,
    h1_diagram,
    h1_loss_against_diagram,
    h1_topological_loss,
)
from src.diagnostics import (
    alignment_uniformity,
    batch_spread,
    effective_rank,
    grad_norms,
    gram_agreement,
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
        topo_batch_size: rows in the point cloud the topological terms read. 0 -- the
            default -- makes that cloud the optimiser's batch, one diagram per step.
            Any other value ``b >= 2`` splits the batch into ``B // b`` disjoint
            clouds of ``b`` rows, averages their losses, and drops the ``B mod b``
            trailing rows from this term. It decouples the filtration scale from the
            batch size: ``L_H0`` compares ``b - 1`` death times, so ``b`` alone
            decides how far up the merge tree the term can see. A batch smaller than
            ``b`` (an epoch's tail) is read whole.
        topo_metric: ground metric of both diagrams on the unit sphere: ``"chord"``
            (Euclidean), ``"angular"`` (geodesic) or ``"cosine"``.
        pooling: pooling used to turn each layer's token states into a sentence vector.
        include_embedding_layer: treat the embedding output as depth 0 state as well.
            Off by default: the paper's L states are the L Transformer layers. Only
            the final state carries loss; this decides which state ``cos_first``
            reports.
        diagnostics: initial value of the :attr:`diagnostics` switch below.

    Attributes:
        diagnostics: when true, :meth:`forward` also reports the *tier 1*
            measurements -- per-term gradient norms, the batch's effective ranks, the
            H0 death-time bias and the student's own H1 diagram. Each of those costs
            a backward through the loss head, a small SVD, a second minimum spanning
            tree or a gudhi call respectively, which is a real fraction of a step on
            a 22M-parameter student, so the training loop flips this on a stride
            instead of leaving it on. Nothing it computes is differentiated through:
            switching it on does not move a seeded trajectory. The *tier 0*
            measurements next to it (spread, alignment/uniformity, batch Gram
            agreement, the weighted per-term contributions and the flags saying which
            terms were defined at all) are elementwise on ``[B, d]`` and are always
            reported.
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
        topo_batch_size: int = 0,
        diagnostics: bool = False,
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
        topo_batch_size = int(topo_batch_size or 0)
        if topo_batch_size < 0 or topo_batch_size == 1:
            raise ValueError(
                "topo_batch_size must be 0 (one diagram per training batch) or >= 2, "
                f"got {topo_batch_size}"
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
        # The cloud the persistence terms read, in rows. 0 keeps the cloud the
        # optimiser's batch, which is what it was before this knob existed.
        self.topo_batch_size = topo_batch_size

        self.lambda_end = float(lambda_end)
        self.lambda_ctr = float(lambda_ctr)
        self.contrastive_temperature = float(contrastive_temperature)
        self.target_projector = target_projector
        self.pooling = pooling
        self.include_embedding_layer = bool(include_embedding_layer)
        self.eps_norm = float(eps_norm)
        # Plain attribute rather than a buffer: it is a logging switch the training
        # loop flips on a stride, and it must never travel in a state dict.
        self.diagnostics = bool(diagnostics)

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
            final_state,
            teacher_deaths,
            metric=self.topo_metric,
            chunk_size=self.topo_batch_size,
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
        return h0_topological_loss(
            final_state,
            teacher,
            metric=self.topo_metric,
            chunk_size=self.topo_batch_size,
        )

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
        return h1_topological_loss(
            final_state,
            teacher,
            metric=self.topo_metric,
            chunk_size=self.topo_batch_size,
        )

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
        # The node every term of the objective passes through on its way into the
        # encoder, including the unnormalised ``raw_final`` of the MSE baseline.
        # Reading the per-term gradients here puts them on one scale.
        grad_node = hidden_states[-1]

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
        # A zero in the log is otherwise three different things -- the weight is 0,
        # the batch was too small, or the two diagrams already agree -- so each term
        # reports whether it was defined at all next to its value.
        # How many rows each diagram is actually built on: the whole batch, or the
        # chunk ``topo_batch_size`` asked for. ``chunk_count`` falls back to the batch
        # when the batch is the smaller of the two (an epoch's tail batch), so this is
        # the size that decides whether a diagram exists at all.
        topo_rows = states[-1].shape[0]
        chunks = chunk_count(topo_rows, self.topo_batch_size)
        if chunks > 1:
            topo_rows = self.topo_batch_size
        topo_active = self.lambda_topo > 0.0 and topo_rows >= 2
        if topo_active:
            if teacher_deaths is not None:
                loss_h0 = self.topological_loss_against_deaths(
                    states[-1], teacher_deaths
                )
            else:
                loss_h0 = self.topological_loss(states[-1], topo_target)
        else:
            loss_h0 = zero

        # A 1-cycle needs three points, so H1 sits out one batch more than H0 does.
        h1_active = (
            self.lambda_topo > 0.0
            and self.lambda_h1 > 0.0
            and topo_rows >= H1_MIN_BATCH
        )
        if h1_active:
            # A pre-built ``teacher_h1`` is one diagram of the whole batch, so it is
            # only the teacher side when the batch *is* the cloud. Under chunking the
            # collate hands over the raw cache instead and the per-chunk diagrams are
            # built here -- they are ragged ``[K_k, 2]`` tensors that no collate could
            # have stacked into the batch dict anyway.
            if teacher_h1 is not None and chunks == 1:
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

        reported: dict[str, torch.Tensor | float] = {
            "loss_total": total,
            "loss_end": loss_end,
            "loss_ctr": loss_ctr,
            "loss_gram": loss_gram,
            "loss_topo": loss_topo,
            "loss_h0": loss_h0,
            "loss_h1": loss_h1,
        }
        # Tier 1 first: the gradient readings need the graph the backward has not
        # consumed yet, and they are the one group that must run outside no_grad.
        if self.diagnostics:
            reported.update(
                self._gradient_diagnostics(
                    grad_node,
                    total=total,
                    loss_end=loss_end,
                    loss_ctr=loss_ctr,
                    loss_gram=loss_gram,
                    loss_topo=loss_topo,
                )
            )

        with torch.no_grad():
            reported.update(
                {
                    # Only the final layer is supervised; the shallow cosine is logged
                    # next to it as a free sanity check on the untouched lower stack.
                    "cos_first": (states[0] * teacher).sum(dim=-1).mean(),
                    "cos_final": (states[-1] * teacher).sum(dim=-1).mean(),
                    # What each term actually contributed to the number that was
                    # differentiated, as opposed to the term's own magnitude.
                    "w_end": self.lambda_end * loss_end,
                    "w_ctr": self.lambda_ctr * loss_ctr,
                    "w_gram": self.lambda_gram * loss_gram,
                    "w_topo": self.lambda_topo * loss_topo,
                    "topo_active": float(topo_active),
                    "h1_active": float(h1_active),
                }
            )
            reported.update(self._shape_diagnostics(states[-1], teacher, topo_target))
            if self.lambda_ctr > 0.0 and second_view is not None:
                alignment, uniformity = alignment_uniformity(
                    student_view, second_view.float()
                )
                reported["ctr_alignment"] = alignment
                reported["ctr_uniformity"] = uniformity
            reported.update(
                self._topology_diagnostics(
                    states[-1],
                    topo_target,
                    teacher_deaths=teacher_deaths,
                    teacher_h1=teacher_h1,
                    topo_active=topo_active,
                    h1_active=h1_active,
                )
            )
            if self.diagnostics:
                mid = self._middle_state(hidden_states, attention_mask)
                if mid is not None and mid.shape == teacher.shape:
                    reported["cos_mid"] = (mid * teacher).sum(dim=-1).mean()

            metrics = scalar_metrics(**reported)

        return total, metrics

    # -------------------------------------------------------------- diagnostics

    def _gradient_diagnostics(
        self, node: torch.Tensor, *, total: torch.Tensor, **terms: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Weighted per-term gradient norms at the student's final hidden state.

        The loss *values* in the record say how large each term is; these say how
        hard each one is pulling, which is the only one of the two the optimizer
        reads. ``g_total`` is the norm of their vector sum, so the gap between it
        and the sum of the parts is how much the terms are cancelling.
        """
        if not (torch.is_grad_enabled() and node.requires_grad):
            return {}
        weights = {
            "end": self.lambda_end,
            "ctr": self.lambda_ctr,
            "gram": self.lambda_gram,
            "topo": self.lambda_topo,
            "total": 1.0,
        }
        named = {name.removeprefix("loss_"): term for name, term in terms.items()}
        named["total"] = total
        return grad_norms(node, named, weights)

    @staticmethod
    def _shape_diagnostics(
        final_state: torch.Tensor, teacher: torch.Tensor, topo_target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Is the batch following the teacher's geometry, or folding in on itself?

        ``cos_final`` can rise either way. ``spread_*`` and ``erank_*`` say which,
        and ``gram_*`` is rung 2 of the structural ladder read on the batch --
        against ``topo_target``, i.e. the teacher in its own width when the run has
        it, since the comparison is rotation- and dimension-invariant and the
        teacher's own geometry is the thing the audit is about.
        """
        rmse, corr = gram_agreement(final_state, topo_target)
        return {
            "spread_student": batch_spread(final_state),
            "spread_teacher": batch_spread(teacher),
            "gram_rmse_batch": rmse,
            "gram_corr_batch": corr,
        }

    def _topology_diagnostics(
        self,
        final_state: torch.Tensor,
        topo_target: torch.Tensor,
        *,
        teacher_deaths: torch.Tensor | None,
        teacher_h1: torch.Tensor | None,
        topo_active: bool,
        h1_active: bool,
    ) -> dict[str, torch.Tensor]:
        """The signed residual of L_H0, and how much of a diagram H1 even sees.

        ``L_H0`` is a squared error, so it throws away the one directional thing it
        knows: ``death_bias = mean(d_student - d_teacher)`` is negative while the
        student's cloud is more tightly connected than the teacher's and positive
        while it is looser. ``h1_n_*`` and ``h1_pers_max_*`` say whether the H1 term
        has anything to match at this batch size at all -- a ``loss_h1`` of zero next
        to two empty diagrams is not the same result as one next to two full ones.
        """
        out: dict[str, torch.Tensor] = {}
        if teacher_deaths is not None:
            out["death_mean_t"] = teacher_deaths.float().mean()
        if teacher_h1 is not None:
            out["h1_n_t"] = torch.tensor(
                float(teacher_h1.shape[0]), device=final_state.device
            )
            if teacher_h1.shape[0]:
                out["h1_pers_max_t"] = (
                    (teacher_h1[:, 1] - teacher_h1[:, 0]).float().max()
                )

        if not self.diagnostics:
            return out

        if topo_active:
            student_deaths = h0_death_times(
                final_state,
                metric=self.topo_metric,
                sort=True,
                chunk_size=self.topo_batch_size,
            )
            reference = (
                teacher_deaths
                if teacher_deaths is not None
                else h0_death_times(
                    topo_target,
                    metric=self.topo_metric,
                    sort=True,
                    chunk_size=self.topo_batch_size,
                )
            )
            reference = reference.to(student_deaths.device, student_deaths.dtype)
            out["death_mean_s"] = student_deaths.mean()
            out.setdefault("death_mean_t", reference.mean())
            if reference.shape == student_deaths.shape:
                out["death_bias"] = (student_deaths - reference).mean()

        if h1_active:
            # Under chunking the loss never sees a batch-sized diagram, so report the
            # first chunk's -- the same cloud size the term was computed on.
            student_diagram = h1_diagram(
                split_chunks(final_state, self.topo_batch_size)[0],
                metric=self.topo_metric,
            )
            out["h1_n_s"] = torch.tensor(
                float(student_diagram.shape[0]), device=final_state.device
            )
            if student_diagram.shape[0]:
                out["h1_pers_max_s"] = (
                    (student_diagram[:, 1] - student_diagram[:, 0]).float().max()
                )

        out["erank_student"] = effective_rank(final_state)
        out["erank_teacher"] = effective_rank(topo_target)
        return out

    def _middle_state(
        self,
        hidden_states: Sequence[torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """``Z^(L/2)``, pooled and normalised, for the depth profile.

        :meth:`endpoint_states` deliberately skips every interior depth, because the
        objective reads none of them. This pools one of them back, under no_grad and
        on the diagnostic stride only: with the endpoint the only supervised state,
        where the middle of the stack sits relative to the target is the cheapest
        available reading of how far down the trajectory the endpoint term reaches.
        """
        states = list(hidden_states)
        if not self.include_embedding_layer and len(states) > 1:
            states = states[1:]
        if len(states) < 3:
            return None
        middle = states[len(states) // 2]
        return self.normalize(_pool(middle, attention_mask, self.pooling).float())
