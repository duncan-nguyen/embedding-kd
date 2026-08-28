"""GeoODE-KD: distilling sentence embeddings as teacher-guided geometric dynamics.

The student's own Transformer layers are read as discrete integration steps of a
continuous flow on the unit hypersphere. The teacher supplies a potential

    E(Z, T) = alpha * E_sem(Z, T) + beta * E_geo(Z, T)                     (Eq. 20)

with E_sem the mean *squared geodesic distance* to the teacher and E_geo the batch
Gram gap. Cosine structure is dimension-free, so E_geo is measured against the
teacher's *native* Gram matrix (``teacher_gram``) whenever the caller supplies it;
the projection P_T then enters only the point-wise term. Its negative Riemannian gradient, run in the finite-horizon time warp
s(t) / R(t) with R(t) = int_t^1 s, is a semantic vector field (Eq. 26) that reaches
the teacher exactly at t = 1 (Corollary 1). One exact step of that flow from layer
``l`` predicts where layer ``l+1`` should land (Eq. 30). The velocity loss
(Eq. 32) compares the *direction* of the realized layer update
U^(l) = Log_{Z^(l)}(Z^(l+1)) with the field V^(l) = -B grad_S E(Z^(l), T) in the
tangent space of Z^(l), over the intermediate transitions l = 1..L-2. Only the
final layer is anchored on the teacher endpoint (Eq. 36), which is the boundary
condition of the same flow and supervises the last transition.

Nothing here is a module with weights: the criterion is teacher-conditioned geometry
plus a stop-gradient target, so training adds no parameters and inference is exactly
the unmodified student encoder.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

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
    """Teacher-guided geometric dynamics objective (Eq. 38).

    Args:
        alpha: weight of the instance-level semantic energy inside the potential.
        beta: weight of the relational (Gram) energy inside the potential.
        lambda_end: weight of the endpoint distillation loss.
        lambda_vel: weight of the velocity-matching loss.
        lambda_ctr: weight of the contrastive regulariser at the final layer.
        contrastive_temperature: tau_c of Eq. (37).
        guidance_schedule: ``"linear"`` for s(t) = t (the paper default),
            ``"power"`` for s(t) = t^p, ``"constant"`` for s(t) = 1.
        guidance_power: p of the ``"power"`` schedule.
        pooling: pooling used to turn each layer's token states into a sentence vector.
        include_embedding_layer: treat the embedding output as depth 0 state as well.
            Off by default: the paper's L states are the L Transformer layers.
        stop_grad_target: apply sg[.] to the field V^(l) (Eq. 32). Turning it off
            is the full-gradient-dynamics ablation named in Section 3.5.

    Per-layer diagnostics are not part of the returned training metrics; they are
    produced on demand by :meth:`depth_report`, which the distiller samples on its own
    cadence and writes to ``depth_metrics.jsonl``.
    """

    # arccos has an unbounded derivative at +-1; the clamp keeps Log_z well defined
    # and its gradient finite when a state coincides with (or opposes) its target.
    _COS_CLAMP = 1.0 - 1e-6

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        lambda_end: float = 1.0,
        lambda_vel: float = 1.0,
        lambda_ctr: float = 0.1,
        contrastive_temperature: float = 0.05,
        guidance_schedule: str = "linear",
        guidance_power: float = 1.0,
        pooling: str = "cls",
        include_embedding_layer: bool = False,
        stop_grad_target: bool = True,
        eps_norm: float = 1e-12,
    ):
        super().__init__()
        if guidance_schedule not in {"linear", "power", "constant"}:
            raise ValueError(
                f"Unsupported guidance_schedule={guidance_schedule!r}; "
                "expected 'linear', 'power' or 'constant'"
            )
        if alpha < 0 or beta < 0:
            raise ValueError("alpha and beta must be non-negative (Eq. 20)")
        if guidance_schedule == "power" and guidance_power <= -1.0:
            raise ValueError("guidance_power must exceed -1 so that int_0^1 s is finite")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lambda_end = float(lambda_end)
        self.lambda_vel = float(lambda_vel)
        self.lambda_ctr = float(lambda_ctr)
        self.contrastive_temperature = float(contrastive_temperature)
        self.guidance_schedule = guidance_schedule
        self.guidance_power = float(guidance_power)
        self.pooling = pooling
        self.include_embedding_layer = bool(include_embedding_layer)
        self.stop_grad_target = bool(stop_grad_target)
        self.eps_norm = float(eps_norm)

    # ------------------------------------------------------------------ geometry

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """norm(.) of Eq. (9), applied row-wise."""
        return F.normalize(x, p=2, dim=-1, eps=self.eps_norm)

    @staticmethod
    def tangent_project(Z: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """Pi_Z of Eq. (22), applied row-wise: U - (z^T u) z."""
        return U - (Z * U).sum(dim=-1, keepdim=True) * Z

    @staticmethod
    def geodesic_distance(Z: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """d_g(z_i, tau_i) = arccos(z_i^T tau_i), the great-circle distance (Eq. 17)."""
        return torch.arccos((Z * T).sum(dim=-1).clamp(-1.0, 1.0))

    def log_map(self, Z: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Log_Z(T) of Eq. (23), row-wise: the tangent vector at z pointing along the
        geodesic to tau, with length d_g(z, tau).

        Log_z(tau) = d_g / sin(d_g) * Pi_z(tau). It is the negative Riemannian
        gradient of the squared geodesic distance 1/2 d_g^2, and d_g / sin(d_g) -> 1
        as tau -> z.
        """
        cosine = (Z * T).sum(dim=-1).clamp(-self._COS_CLAMP, self._COS_CLAMP)
        theta = torch.arccos(cosine)
        scale = theta / torch.sin(theta)
        return scale.unsqueeze(-1) * self.tangent_project(Z, T)

    @staticmethod
    def exp_map(Z: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Exp_Z(V) of Eq. (31), row-wise: follow the geodesic from z with initial
        velocity v for unit time. Closed form on the sphere:

        Exp_z(v) = cos(|v|) z + sin(|v|) v / |v|,

        which is exactly on the sphere for every tangent v (Proposition 1) and
        satisfies Exp_z(Log_z(tau)) = tau.
        """
        norm = V.norm(dim=-1, keepdim=True)
        # sin(n)/n written through torch.sinc, so n = 0 is handled exactly.
        return torch.cos(norm) * Z + torch.sinc(norm / math.pi) * V

    def retract(self, Z: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """First-order retraction RowNorm(Z + V): the cheap approximation to
        :meth:`exp_map` that the earlier draft used. Kept for the discretisation
        ablation; the flow itself uses the exact exponential map."""
        return self.normalize(Z + V)

    def guidance(self, t: float) -> float:
        """s(t) of Eq. (28) and its ablations."""
        if self.guidance_schedule == "constant":
            return 1.0
        if self.guidance_schedule == "linear":
            return float(t)
        return float(t) ** self.guidance_power

    def guidance_mass(self, t: float) -> float:
        """R(t) = int_t^1 s(u) du of Eq. (27): the guidance still to be spent after
        depth t. R(1) = 0, and R(0) is the total guidance of the whole trajectory."""
        if self.guidance_schedule == "constant":
            return 1.0 - float(t)
        p = 1.0 if self.guidance_schedule == "linear" else self.guidance_power
        return (1.0 - float(t) ** (p + 1.0)) / (p + 1.0)

    def step_fraction(self, t: float, t_next: float) -> float:
        """rho of Eq. (30): the fraction of the remaining geodesic to the target that
        the flow covers between depths t and t_next,

            rho(t, t_next) = 1 - R(t_next) / R(t),

        i.e. the exact integral of the time warp s / R over [t, t_next]. It equals 1
        on the last interval (t_next = 1), so the final prediction is the target.
        """
        if not 0.0 <= t < t_next <= 1.0:
            raise ValueError(f"expected 0 <= t < t_next <= 1, got t={t}, t_next={t_next}")
        remaining = self.guidance_mass(t)
        if remaining <= 0.0:
            return 1.0
        return 1.0 - self.guidance_mass(t_next) / remaining

    @staticmethod
    def teacher_gram(T: torch.Tensor, teacher_gram: torch.Tensor | None) -> torch.Tensor:
        """G_T of Eq. (18): the native teacher Gram when given, else that of the
        projected targets."""
        if teacher_gram is not None:
            return teacher_gram.to(T.dtype)
        return T @ T.transpose(0, 1)

    def energy(
        self,
        Z: torch.Tensor,
        T: torch.Tensor,
        teacher_gram: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-conditioned potential of Eq. (20).

        Returns ``(E, E_sem, E_geo)`` with the two components before their weights.
        ``teacher_gram`` is the teacher's native ``[B, B]`` cosine matrix; without it
        the Gram of the projected targets is used.

        Reported in the paper's batch-mean form so the numbers stay comparable across
        batch sizes; :meth:`vector_field` differentiates ``B`` times this, which is
        the same flow at a batch-size-independent speed.
        """
        batch = Z.shape[0]
        e_sem = 0.5 * self.geodesic_distance(Z, T).pow(2).mean()  # Eq. 17
        gram_gap = Z @ Z.transpose(0, 1) - self.teacher_gram(T, teacher_gram)
        e_geo = gram_gap.pow(2).sum() / (batch * batch)  # Eq. 19
        return self.alpha * e_sem + self.beta * e_geo, e_sem, e_geo

    def vector_field(
        self,
        Z: torch.Tensor,
        T: torch.Tensor,
        teacher_gram: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """V(Z, T) of Eq. (25): the tangent negative gradient of the potential, before
        the depth-dependent time warp.

            V = alpha * Log_Z(T) - (4 beta / B) * Pi_Z[(Z Z^T - T T^T) Z]

        Taken from the *per-sample* energy, i.e. ``B`` times Eq. (20): the
        paper's batch-mean energy would give a field that slows down with batch
        size, while the direction is identical. The full field of Eq. (26) is
        s(t) / R(t) * V; the warp is integrated exactly by :meth:`step_fraction`
        rather than evaluated pointwise, so it never appears here.
        """
        batch = Z.shape[0]
        gram_gap = Z @ Z.transpose(0, 1) - self.teacher_gram(T, teacher_gram)
        relational = self.tangent_project(Z, gram_gap @ Z)
        return self.alpha * self.log_map(Z, T) - (4.0 * self.beta / batch) * relational

    def flow_step(
        self,
        Z: torch.Tensor,
        T: torch.Tensor,
        rho: float,
        teacher_gram: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Move a fraction ``rho`` of the prescribed tangent displacement along the
        geodesic: Exp_Z(rho * V(Z, T)). With beta = 0 this is exactly the spherical
        interpolation slerp(z, tau; rho), so rho = 1 lands on the teacher."""
        return self.exp_map(Z, rho * self.vector_field(Z, T, teacher_gram))

    def euler_step(
        self,
        Z: torch.Tensor,
        T: torch.Tensor,
        t: float,
        t_next: float,
        teacher_gram: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One step of the discretised flow from depth t to t_next, Eq. (30)."""
        return self.flow_step(Z, T, self.step_fraction(t, t_next), teacher_gram)

    # ------------------------------------------------------------------ losses

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

    @staticmethod
    def _depth(index: int, num_layers: int) -> float:
        """t_l = l / L of Eq. (14) for the 0-based state ``index`` (l = index + 1)."""
        return (index + 1) / num_layers

    def velocity_loss(
        self,
        states: Sequence[torch.Tensor],
        teacher: torch.Tensor,
        teacher_gram: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Velocity matching, Eq. (32):

            L_vel = 1/(L-2) sum_{l=1}^{L-2} 1/B sum_i [1 - cos(U_i^(l), sg[V_i^(l)])]

        with U^(l) = Log_{Z^(l)}(Z^(l+1)) the realized tangent update and
        V^(l) = V(Z^(l), T) the teacher-conditioned field. The cosine is scale-free,
        so the depth warp s(t)/R(t) (a positive scalar) drops out: only the
        direction of each intermediate transition is trained. The final transition
        l = L-1 is left to :meth:`endpoint_loss`.
        """
        num_layers = len(states)
        if num_layers < 3:
            return torch.zeros((), device=teacher.device, dtype=teacher.dtype)

        terms = []
        for index in range(num_layers - 2):
            current, actual = states[index], states[index + 1]
            if self.stop_grad_target:
                # sg[.] of Eq. (32): the field is a fixed local target read off the
                # current state; gradients reach Z^(l) only through U^(l).
                with torch.no_grad():
                    field = self.vector_field(current.detach(), teacher, teacher_gram)
            else:
                field = self.vector_field(current, teacher, teacher_gram)
            update = self.log_map(current, actual)
            cosine = F.cosine_similarity(update, field, dim=-1, eps=self.eps_norm)
            terms.append((1.0 - cosine).mean())

        return torch.stack(terms).mean()

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

    # ------------------------------------------------------------------ analysis

    @staticmethod
    def _mean_offdiagonal_cosine(Z: torch.Tensor) -> float:
        """Mean pairwise cosine excluding the diagonal: a batch anisotropy proxy.

        The paper keeps a contrastive term because pure imitation can inherit the
        teacher's concentration; this is the number that shows whether it did.
        """
        batch = Z.shape[0]
        if batch < 2:
            return 0.0
        gram = Z @ Z.transpose(0, 1)
        total = gram.sum() - gram.diagonal().sum()
        return float(total / (batch * (batch - 1)))

    @torch.no_grad()
    def depth_report(
        self,
        states: Sequence[torch.Tensor],
        teacher: torch.Tensor,
        teacher_gram: torch.Tensor | None = None,
    ) -> dict[str, object]:
        """Per-depth diagnostics for post-hoc analysis of one batch.

        The paper's research hypothesis is stated over quantities that the training
        loss never reports: how the teacher discrepancy and the relational gap evolve
        *across depth*, and whether the student's actual layer transition follows the
        prescribed field. All of it is measured here, on the realized trajectory:

        - ``cos_teacher`` / ``geodesic_distance`` / ``gram_gap`` / ``energy``: the
          profile of Eqs. (17), (19) and (20) at every depth. Hypotheses 1 and 2 are
          claims about these curves.
        - ``predicted_geodesic_distance``: Corollary 1's closed form
          d(t_l) = d(t_1) R(t_l) / R(t_1), the profile the instance-only flow would
          trace from the same first layer. It reaches zero at the last layer, and the
          gap to ``geodesic_distance`` is how far the student is from the flow.
        - ``energy_violations``: depths where the realized trajectory *raises* the
          energy. Proposition 2 guarantees descent for the ideal continuous flow, so
          this counts how far the discrete student is from realizing it.
        - ``vel_residual``: the per-transition term of Eq. (32),
          1 - cos(U, V) in the tangent space, i.e. where along depth the velocity
          loss is actually being paid. Reported for every transition, including
          the last one that the training loss leaves to the endpoint term.
        - ``field_norm`` vs ``step_norm``: the geodesic length the flow prescribes
          for the transition next to the geodesic length the layer actually moved.
          A tiny ``vel_residual`` means little if the prescribed step is negligible,
          and this pair is what separates the two.
        - ``direction_alignment``: cosine, in the tangent space of the current
          layer, between Log of the actual update and the prescribed tangent
          displacement. Scale-free, so unlike the loss it says whether the student
          moves *where* the teacher points, not just *how far*.
        """
        teacher = self.normalize(teacher.to(states[-1].dtype))
        num_layers = len(states)

        cos_teacher, distance, gram_gap, energy = [], [], [], []
        if teacher_gram is not None:
            teacher_gram = teacher_gram.to(states[-1].dtype)
        for state in states:
            total, _, geo = self.energy(state, teacher, teacher_gram)
            cos_teacher.append(float((state * teacher).sum(dim=-1).mean()))
            distance.append(float(self.geodesic_distance(state, teacher).mean()))
            gram_gap.append(float(geo))
            energy.append(float(total))

        first_mass = self.guidance_mass(self._depth(0, num_layers))
        predicted_distance = [
            distance[0] * self.guidance_mass(self._depth(index, num_layers)) / first_mass
            if first_mass > 0.0
            else 0.0
            for index in range(num_layers)
        ]

        vel_residual, field_norm, step_norm, alignment = [], [], [], []
        for index in range(num_layers - 1):
            t = self._depth(index, num_layers)
            t_next = self._depth(index + 1, num_layers)
            current, actual = states[index], states[index + 1]
            field = self.step_fraction(t, t_next) * self.vector_field(
                current, teacher, teacher_gram
            )
            update = self.log_map(current, actual)
            cosine = float(
                F.cosine_similarity(update, field, dim=-1, eps=self.eps_norm).mean()
            )
            vel_residual.append(1.0 - cosine)
            field_norm.append(float(field.norm(dim=-1).mean()))
            step_norm.append(float(update.norm(dim=-1).mean()))
            alignment.append(cosine)

        def _violations(values: list[float], should_decrease: bool) -> int:
            pairs = itertools.pairwise(values)
            if should_decrease:
                return sum(1 for earlier, later in pairs if later > earlier)
            return sum(1 for earlier, later in pairs if later < earlier)

        # Hypothesis 1 says the discrepancy should fall *smoothly*, so a curvature
        # measure is reported next to the endpoints: a curve that drops all at once
        # in the last layer is a different outcome from one that drops gradually.
        curvature = [
            abs(cos_teacher[i + 1] - 2.0 * cos_teacher[i] + cos_teacher[i - 1])
            for i in range(1, num_layers - 1)
        ]

        return {
            "num_layers": num_layers,
            "alpha": self.alpha,
            "beta": self.beta,
            "layers": list(range(1, num_layers + 1)),
            "cos_teacher": cos_teacher,
            "geodesic_distance": distance,
            "predicted_geodesic_distance": predicted_distance,
            "gram_gap": gram_gap,
            "energy": energy,
            "vel_residual": vel_residual,
            "field_norm": field_norm,
            "step_norm": step_norm,
            "direction_alignment": alignment,
            "cos_first": cos_teacher[0],
            "cos_final": cos_teacher[-1],
            "cos_gain": cos_teacher[-1] - cos_teacher[0],
            "cos_curvature": sum(curvature) / len(curvature) if curvature else 0.0,
            "cos_violations": _violations(cos_teacher, should_decrease=False),
            "gram_gap_first": gram_gap[0],
            "gram_gap_final": gram_gap[-1],
            "gram_gap_contraction": gram_gap[0] - gram_gap[-1],
            "gram_violations": _violations(gram_gap, should_decrease=True),
            "energy_first": energy[0],
            "energy_final": energy[-1],
            "energy_violations": _violations(energy, should_decrease=True),
            "mean_vel_residual": sum(vel_residual) / len(vel_residual)
            if vel_residual
            else 0.0,
            "mean_alignment": sum(alignment) / len(alignment) if alignment else 0.0,
            "mean_field_norm": sum(field_norm) / len(field_norm) if field_norm else 0.0,
            "mean_step_norm": sum(step_norm) / len(step_norm) if step_norm else 0.0,
            "student_anisotropy": self._mean_offdiagonal_cosine(states[-1]),
            "teacher_anisotropy": self._mean_offdiagonal_cosine(teacher),
        }

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        hidden_states: Iterable[torch.Tensor],
        teacher: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        second_view: torch.Tensor | None = None,
        teacher_gram: torch.Tensor | None = None,
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
            teacher_gram: the teacher's native ``[B, B]`` cosine matrix for the
                relational energy; defaults to the Gram of the projected targets.
        """
        states = self.layer_states(hidden_states, attention_mask)
        if not states:
            raise ValueError("hidden_states contained no supervised layer")

        teacher = self.normalize(teacher.to(states[-1].dtype))
        if teacher.shape != states[-1].shape:
            raise ValueError(
                f"teacher targets have shape {tuple(teacher.shape)} but the student's "
                f"final state has shape {tuple(states[-1].shape)}; teacher embeddings "
                "must be projected into the student dimension before training"
            )

        if teacher_gram is not None:
            if teacher_gram.shape != (teacher.shape[0], teacher.shape[0]):
                raise ValueError(
                    f"teacher_gram must be [B, B] = {(teacher.shape[0],) * 2}, got "
                    f"{tuple(teacher_gram.shape)}"
                )
            teacher_gram = teacher_gram.to(teacher.dtype)
        loss_vel = self.velocity_loss(states, teacher, teacher_gram)
        loss_end = self.endpoint_loss(states[-1], teacher)

        if self.lambda_ctr > 0.0 and second_view is not None:
            loss_ctr = self.contrastive_loss(
                states[-1], self.normalize(second_view.float())
            )
        else:
            loss_ctr = torch.zeros((), device=teacher.device, dtype=teacher.dtype)

        total = (
            self.lambda_end * loss_end
            + self.lambda_vel * loss_vel
            + self.lambda_ctr * loss_ctr
        )

        with torch.no_grad():
            energy_first, _, geo_first = self.energy(states[0], teacher, teacher_gram)
            energy_last, _, geo_last = self.energy(states[-1], teacher, teacher_gram)
            metrics = {
                "loss_total": float(total.detach()),
                "loss_end": float(loss_end.detach()),
                "loss_vel": float(loss_vel.detach()),
                "loss_ctr": float(loss_ctr.detach()),
                # The hypothesis is that both discrepancies contract with depth, so
                # the shallow and final values are logged as a pair.
                "cos_first": float((states[0] * teacher).sum(dim=-1).mean()),
                "cos_final": float((states[-1] * teacher).sum(dim=-1).mean()),
                "gram_gap_first": float(geo_first),
                "gram_gap_final": float(geo_last),
                "energy_first": float(energy_first),
                "energy_final": float(energy_last),
            }

        return total, metrics
