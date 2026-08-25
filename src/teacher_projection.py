"""Fixed linear map from the teacher embedding space into the student dimension.

GeoODE-KD compares student layers with teacher embeddings on a shared hypersphere,
so a large teacher (d_T) has to be mapped into the student dimension (d_S) once,
before training. The map is fitted on the cached training embeddings only and then
frozen: it is a property of the cached targets, not a module that learns alongside
the student, and nothing in it survives into inference.
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

    if n_rows < out_dim:
        raise ValueError(
            f"cannot fit a {out_dim}-dimensional PCA map from {n_rows} cached embeddings"
        )

    centered = matrix - mean if center else matrix
    # Right singular vectors of the (centered) data are the principal directions.
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    projection = vh[:out_dim].transpose(0, 1).contiguous()  # [d_T, out_dim]
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
