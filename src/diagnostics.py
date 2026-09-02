"""Batch-level instrumentation: what the loss values alone do not say.

Every function here is a *measurement*, never a term of an objective. Nothing in
this module is differentiated through, nothing it returns reaches the optimizer,
and adding or removing a call changes no trajectory -- which is the property that
makes it safe to switch on inside a seeded ablation.

Three questions the training log of an endpoint distillation cannot otherwise
answer:

* **Who is pulling?** A loss curve reports a term's *value*; the optimizer reads
  its *gradient*. Two terms whose values differ by two orders of magnitude can
  contribute equal gradient, and a term whose weight was swept over decades can
  turn out never to have moved the student at all. :func:`grad_norms` reads the
  gradient each term sends into a shared node of the graph, already multiplied by
  the weight it enters the total with, so the numbers add up to the step that was
  actually taken.
* **Is the batch collapsing?** A cosine-to-target of 0.9 is compatible with a
  cloud that has folded onto a line, and the endpoint loss cannot see the
  difference. :func:`batch_spread`, :func:`effective_rank` and
  :func:`alignment_uniformity` are the three standard readings of that, at a cost
  set by the batch size rather than by the model.
* **Is the geometry following?** :func:`gram_agreement` is rung 2 of the
  structural ladder evaluated on the batch: it compares the two clouds' pairwise
  angles and needs no correspondence between the spaces' axes, so the student may
  be compared against the teacher in the teacher's own width.

Every function returns 0-d tensors, on the caller's device, so a criterion can
hand the whole dict to :func:`src.metrics.scalar_metrics` and pay one device
synchronisation for all of them together.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

__all__ = [
    "alignment_uniformity",
    "batch_spread",
    "effective_rank",
    "gram_agreement",
    "grad_norms",
    "offdiag_cosines",
]


def _unit(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(x.detach().float(), p=2, dim=-1, eps=eps)


def offdiag_cosines(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """The ``B (B - 1)`` off-diagonal entries of the batch's cosine Gram matrix."""
    u = _unit(x, eps)
    gram = u @ u.transpose(0, 1)
    mask = ~torch.eye(gram.shape[0], dtype=torch.bool, device=gram.device)
    return gram[mask]


@torch.no_grad()
def batch_spread(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """``1 -`` the mean off-diagonal cosine: 0 is a collapsed batch, 1 an orthogonal one.

    The same reading :class:`~src.criterions.relational_kd.RelationalKD` reports for
    its own two sides. On the teacher it is a constant of the corpus; on the student
    it is the number that says whether a rising cosine-to-target is the student
    finding the targets or the batch folding in on itself.
    """
    batch = x.shape[0]
    if batch < 2:
        return x.new_zeros((), dtype=torch.float32)
    return 1.0 - offdiag_cosines(x, eps).mean()


@torch.no_grad()
def effective_rank(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """RankMe (Garrido et al. 2023): ``exp(H(p))``, ``p_i = sigma_i / sum sigma``.

    Bounded by ``min(B, d)``, so on a training batch it is read as a fraction of the
    batch size, not of the student width. The SVD is of a ``[B, d]`` matrix and runs
    on the batch alone, but cuSOLVER's small-matrix path is slow enough relative to a
    22M-parameter step that this belongs behind a stride rather than on every step.
    """
    if x.shape[0] < 2:
        return x.new_zeros((), dtype=torch.float32)
    sigma = torch.linalg.svdvals(x.detach().float())
    p = sigma / sigma.sum().clamp(min=eps)
    p = p.clamp(min=eps)
    return torch.exp(-(p * p.log()).sum())


@torch.no_grad()
def alignment_uniformity(
    view_a: torch.Tensor, view_b: torch.Tensor, t: float = 2.0, eps: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wang & Isola (2020), the two halves InfoNCE trades against each other.

    ``alignment`` is the mean squared distance between the two views of the same
    sentence (lower is tighter); ``uniformity`` is ``log E exp(-t ||x_i - x_j||^2)``
    over distinct pairs of the first view (lower is more spread out, ``-2 t`` in
    the limit of a perfectly uniform sphere at ``t = 2``). Reported next to
    ``loss_ctr`` because the loss value alone cannot say which of the two the
    regulariser is currently buying.
    """
    a, b = _unit(view_a, eps), _unit(view_b, eps)
    alignment = ((a - b) ** 2).sum(dim=-1).mean()
    if a.shape[0] < 2:
        return alignment, a.new_zeros(())
    # ||x_i - x_j||^2 = 2 - 2 cos on the sphere; logsumexp for the numerics.
    squared = (2.0 - 2.0 * offdiag_cosines(a, eps)).clamp(min=0.0)
    uniformity = torch.logsumexp(-t * squared, dim=0) - torch.log(
        torch.tensor(float(squared.numel()), device=squared.device)
    )
    return alignment, uniformity


@torch.no_grad()
def gram_agreement(
    a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rung 2 on the batch: RMSE and Pearson correlation of the off-diagonal Grams.

    Both readings are invariant to an orthogonal rotation of either side and to the
    widths of the two spaces, so ``b`` may be the teacher in its own ``d_T``. The
    RMSE is the absolute angular distortion; the correlation is the same comparison
    with a global contraction of the cloud divided out, and the gap between the two
    is exactly the part of the mismatch that is a uniform shrink rather than a
    reordering of the neighbourhoods.
    """
    if a.shape[0] < 2:
        zero = a.new_zeros((), dtype=torch.float32)
        return zero, zero
    x, y = offdiag_cosines(a, eps), offdiag_cosines(b, eps)
    rmse = torch.sqrt(((x - y) ** 2).mean())
    xc, yc = x - x.mean(), y - y.mean()
    corr = (xc * yc).sum() / (xc.norm() * yc.norm()).clamp(min=eps)
    return rmse, corr


def grad_norms(
    node: torch.Tensor,
    terms: Mapping[str, torch.Tensor],
    weights: Mapping[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """``|| d(weight * term) / d(node) ||`` for each term, in one shared node.

    ``node`` is a tensor every term is downstream of -- in an endpoint distillation
    the student's final hidden state, i.e. the place the whole objective enters the
    encoder. Reading every term there puts them on one scale and makes the numbers
    comparable in the only sense that matters: the step the optimizer takes is the
    sum of these vectors.

    The weight is folded in, so a term that is reported large but enters at
    ``lambda = 0.01`` shows up as what it is. A term that is a constant (its weight
    is zero, or the batch was too small for it to be defined) has no path to ``node``
    and is reported as exactly 0 rather than omitted, so the series has no holes.

    Each call is a backward through the loss head only -- pooling, a normalisation
    and the term itself -- never through the encoder, whose graph is retained for the
    real backward that follows.
    """
    weights = weights or {}
    out: dict[str, torch.Tensor] = {}
    for name, term in terms.items():
        weight = float(weights.get(name, 1.0))
        if weight == 0.0 or not torch.is_tensor(term) or not term.requires_grad:
            out[f"g_{name}"] = node.new_zeros((), dtype=torch.float32)
            continue
        (grad,) = torch.autograd.grad(
            term, node, retain_graph=True, allow_unused=True, create_graph=False
        )
        out[f"g_{name}"] = (
            node.new_zeros((), dtype=torch.float32)
            if grad is None
            else weight * grad.detach().float().norm()
        )
    return out
