"""Fixed linear map from the teacher embedding space into the student dimension.

GeoODE-KD compares student layers with teacher embeddings on a shared hypersphere,
so a large teacher (d_T) has to be mapped into the student dimension (d_S) once,
before training. The map is fitted on the cached training embeddings only and then
frozen: it is a property of the cached targets, not a module that learns alongside
the student, and nothing in it survives into inference.

The map has two factors, P_T = P_PCA R (Eq. 8). P_PCA picks the d_S-dimensional
subspace (Eckart-Young: the rank-d_S map that best preserves the teacher's Gram
matrix). R is an orthogonal matrix *inside* that subspace, fixed by orthogonal
Procrustes against the untrained student: every downstream metric and the
relational energy are invariant to a rotation of the student space, so the
coordinates PCA happens to return are an arbitrary gauge, and R removes it so that
the endpoint loss does not ask the student to rotate its whole pretrained space.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fit_pca_projection(
    embeddings: torch.Tensor,
    out_dim: int,
    center: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit the PCA map ``P_T`` of Eq. (8) on cached teacher embeddings.

    Args:
        embeddings: cached teacher embeddings, ``[N, d_T]``.
        out_dim: student dimension ``d_S``.
        center: subtract the corpus mean before the SVD. This only decides which
            directions PCA picks; whether the mean is also removed when the map is
            *applied* is a separate choice (see :func:`project_teacher_embeddings`).

    Returns:
        ``(P, mean)`` where ``P`` is ``[d_T, out_dim]`` with orthonormal columns and
        ``mean`` is ``[d_T]``. When ``out_dim >= d_T`` the map is the identity, which
        is the equal-dimension case called out in the paper.

    A corpus with fewer rows than ``out_dim`` spans too few directions to fill the
    map. Rather than refusing (small debug corpora are a normal thing to run), the
    principal directions it does span are kept and the remaining columns are filled
    with a deterministic orthonormal complement, so the map stays a well-defined
    isometry onto the student dimension.
    """
    if embeddings.dim() != 2:
        raise ValueError(
            f"expected a [N, d_T] matrix, got shape {tuple(embeddings.shape)}"
        )
    if out_dim <= 0:
        raise ValueError(f"out_dim must be positive, got {out_dim}")

    matrix = embeddings.detach().to(torch.float32)
    n_rows, teacher_dim = matrix.shape
    mean = matrix.mean(dim=0)

    if out_dim >= teacher_dim:
        return torch.eye(teacher_dim, dtype=torch.float32), mean

    centered = matrix - mean if center else matrix
    # Right singular vectors of the (centered) data are the principal directions.
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    principal = vh[:out_dim]  # [min(out_dim, rank), d_T]

    if principal.shape[0] < out_dim:
        # QR fills the deficit: it orthonormalises left to right, so the principal
        # directions stay first and the identity columns supply the complement.
        stacked = torch.cat(
            [principal.transpose(0, 1), torch.eye(teacher_dim, dtype=matrix.dtype)],
            dim=1,
        )
        completed, _ = torch.linalg.qr(stacked)
        projection = completed[:, :out_dim].contiguous()
    else:
        projection = principal.transpose(0, 1).contiguous()  # [d_T, out_dim]
    return projection, mean


def project_teacher_embeddings(
    embeddings: torch.Tensor,
    projection: torch.Tensor,
    mean: torch.Tensor | None = None,
    subtract_mean: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Apply Eq. (8): ``tau_i = norm(f_T(x_i) P_T)``.

    ``subtract_mean`` is off by default so that the applied map is exactly the linear
    ``P_T`` of the paper; turning it on makes the transform the textbook PCA one.
    """
    matrix = embeddings.detach().to(torch.float32)
    if subtract_mean:
        if mean is None:
            raise ValueError("subtract_mean=True requires the fitted mean")
        matrix = matrix - mean.to(matrix.dtype)
    projected = matrix @ projection.to(matrix.dtype)
    return F.normalize(projected, p=2, dim=-1, eps=eps)


def fit_gauge_alignment(
    targets: torch.Tensor,
    student: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Orthogonal Procrustes: the rotation of the target space closest to the student.

    Solves ``R = argmin_{R in O(d)} ||targets R - student||_F`` in closed form
    (Schoenemann, 1966): with ``U S V^T = svd(targets^T student)``, ``R = U V^T``.

    Args:
        targets: projected, normalised teacher targets ``[N, d_S]``.
        student: the student's own embeddings of the same texts, ``[N, d_S]``,
            normalised, taken *before* training.

    Returns:
        ``(R, stats)`` with ``R`` ``[d_S, d_S]`` orthogonal and ``stats`` reporting
        the mean student-target cosine before and after the rotation. ``N`` should
        comfortably exceed ``d_S`` for the cross-covariance to be well conditioned.

    ``R`` acts inside the subspace ``P_PCA`` already chose, so it leaves the Gram
    matrix of the targets, and hence the relational energy and the retained
    variance, exactly unchanged.
    """
    if targets.shape != student.shape or targets.dim() != 2:
        raise ValueError(
            f"targets {tuple(targets.shape)} and student {tuple(student.shape)} must be "
            "two matrices of the same shape"
        )
    T = F.normalize(targets.detach().to(torch.float32), dim=-1)
    Z = F.normalize(student.detach().to(torch.float32), dim=-1)
    u, _, vh = torch.linalg.svd(T.transpose(0, 1) @ Z, full_matrices=False)
    rotation = u @ vh
    before = float((T * Z).sum(dim=-1).mean())
    after = float(((T @ rotation) * Z).sum(dim=-1).mean())
    return rotation, {"cos_before": before, "cos_after": after, "samples": T.shape[0]}
