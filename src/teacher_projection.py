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

Both factors are claims that need controls, so each is a switch:

* the *subspace* factor: centered PCA (``fit_pca_projection(center=True)``),
  uncentered SVD (``center=False``), or a data-independent random map
  (:func:`fit_random_projection`, orthonormal or Gaussian). If the teacher's
  leading spectral subspace is what carries the signal, PCA must beat both random
  arms; if it does not, "spectral" is decoration and the paper has no §3.2.
* the *orientation* factor: Procrustes against the student init
  (:func:`fit_gauge_alignment`), no rotation at all, or a Haar-random rotation
  (:func:`random_orthogonal`). The random rotation is the control that separates
  "R is the *right* orientation" from "R is *an* orientation": PCA alone is
  already an arbitrary gauge, so PCA+Procrustes only means something if it beats
  PCA+Q with Q random, at matched cost.

:func:`fit_teacher_projection` dispatches the subspace arms by name so a run
records which one it used.
"""

from __future__ import annotations

import numpy as np
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


def _haar_orthonormal(
    rows: int, cols: int, generator: torch.Generator
) -> torch.Tensor:
    """A ``[rows, cols]`` matrix with orthonormal columns, Haar-uniform in O(rows).

    QR of a Gaussian matrix is orthonormal but not Haar-uniform: LAPACK is free to
    pick the sign of each column, and its convention biases the result. Multiplying
    by the signs of ``diag(R)`` removes that bias (Mezzadri, 2007), which matters
    here because a *biased* random baseline would be a weaker control than it looks.
    """
    if cols > rows:
        raise ValueError(f"cannot orthonormalise {cols} columns in {rows} dimensions")
    gaussian = torch.randn(rows, cols, generator=generator, dtype=torch.float32)
    q, r = torch.linalg.qr(gaussian)
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


def random_orthogonal(dim: int, seed: int = 0) -> torch.Tensor:
    """A Haar-random orthogonal ``[dim, dim]`` matrix: the control gauge ``Q``.

    Used in place of the Procrustes ``R`` to test whether the *specific* orientation
    matters or merely the fact that some orientation was applied. Drawn from a
    private CPU generator so the same ``seed`` gives the same ``Q`` no matter where
    in the run it is called or what the global RNG has done.
    """
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return _haar_orthonormal(dim, dim, generator)


def fit_random_projection(
    embeddings: torch.Tensor,
    out_dim: int,
    seed: int = 0,
    orthonormal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Data-independent control for :func:`fit_pca_projection`.

    Same signature and same contract, but the subspace is drawn at random instead
    of read off the teacher's spectrum. This is what isolates the *Eckart-Young*
    claim: if random targets train as well as spectral ones, the leading teacher
    subspace was not what mattered.

    Args:
        embeddings: cached teacher embeddings, ``[N, d_T]``. Only their shape and
            mean are used; the map itself never looks at the data.
        out_dim: student dimension ``d_S``.
        seed: draw index. Different seeds are different draws of the same control,
            so a spread over seeds is the null band PCA has to clear.
        orthonormal: ``True`` draws a Haar-random ``d_S``-subspace (orthonormal
            columns, exactly the property PCA's map has, so the two arms differ
            *only* in which subspace). ``False`` draws iid Gaussian entries scaled
            by ``1/sqrt(d_S)`` -- the Johnson-Lindenstrauss map, norm-preserving in
            expectation but not an isometry, which is the arm that also gives up the
            orthonormality.

    At a given ``seed`` the two flavours are drawn from the *same* Gaussian matrix,
    so they span the same random subspace and differ only in the conditioning inside
    it. Comparing them therefore isolates orthonormality alone, with the subspace
    held fixed.

    Returns:
        ``(P, mean)`` with the same shapes as :func:`fit_pca_projection`. The mean
        is still the corpus mean, so ``subtract_mean`` behaves identically here.
    """
    if embeddings.dim() != 2:
        raise ValueError(
            f"expected a [N, d_T] matrix, got shape {tuple(embeddings.shape)}"
        )
    if out_dim <= 0:
        raise ValueError(f"out_dim must be positive, got {out_dim}")

    matrix = embeddings.detach().to(torch.float32)
    teacher_dim = matrix.shape[1]
    mean = matrix.mean(dim=0)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))

    if out_dim >= teacher_dim:
        # The PCA arm returns the identity here (nothing to discard). The random
        # arm's counterpart is a random rotation: it also discards nothing, so the
        # two stay comparable and the only difference is still the coordinates.
        return _haar_orthonormal(teacher_dim, teacher_dim, generator), mean
    if orthonormal:
        return _haar_orthonormal(teacher_dim, out_dim, generator), mean
    projection = torch.randn(
        teacher_dim, out_dim, generator=generator, dtype=torch.float32
    ) / (out_dim**0.5)
    return projection, mean


def fit_mrl_prefix_projection(
    embeddings: torch.Tensor, out_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The Matryoshka-prefix interface: keep the teacher's first ``out_dim`` coordinates.

    Same contract as :func:`fit_pca_projection`. The map is the first ``out_dim``
    columns of the identity, so it is orthonormal like PCA and random-orthonormal,
    but the subspace is chosen by *coordinate order* rather than by the spectrum or
    at random. For a teacher trained with Matryoshka representation learning the
    leading coordinates are meant to carry the most information, so this arm is
    expected between random and PCA; for any other teacher it is a random-looking
    axis-aligned subspace and should behave like the random arm. Data-independent:
    only the shape and mean of ``embeddings`` are read.
    """
    if embeddings.dim() != 2:
        raise ValueError(
            f"expected a [N, d_T] matrix, got shape {tuple(embeddings.shape)}"
        )
    if out_dim <= 0:
        raise ValueError(f"out_dim must be positive, got {out_dim}")
    matrix = embeddings.detach().to(torch.float32)
    teacher_dim = matrix.shape[1]
    mean = matrix.mean(dim=0)
    if out_dim >= teacher_dim:
        return torch.eye(teacher_dim, dtype=torch.float32), mean
    return torch.eye(teacher_dim, dtype=torch.float32)[:, :out_dim].contiguous(), mean


PROJECTION_TYPES = ("pca", "random", "random_gaussian", "mrl_prefix")


def fit_teacher_projection(
    embeddings: torch.Tensor,
    out_dim: int,
    projection_type: str = "pca",
    center: bool = True,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit the subspace factor of ``P_T`` by name, so a run records its arm.

    ``"pca"`` with ``center=True`` is the paper's map (centered PCA) and with
    ``center=False`` the uncentered SVD ablation -- the same routine, differing only
    in whether the corpus mean is removed before the SVD, i.e. whether the first
    retained direction is allowed to be the teacher's mean vector. ``"random"`` and
    ``"random_gaussian"`` are the two data-independent controls, and
    ``"mrl_prefix"`` (the leading coordinates) is the third fixed interface.
    """
    if projection_type == "pca":
        return fit_pca_projection(embeddings, out_dim=out_dim, center=center)
    if projection_type in ("random", "random_gaussian"):
        return fit_random_projection(
            embeddings,
            out_dim=out_dim,
            seed=seed,
            orthonormal=projection_type == "random",
        )
    if projection_type == "mrl_prefix":
        return fit_mrl_prefix_projection(embeddings, out_dim=out_dim)
    if projection_type in ("learned_t2s", "learned_s2t"):
        raise ValueError(
            f"projection_type={projection_type!r} is a trained map, not a fitted one: "
            "it is a parameter of the criterion (see src/target_projector.py) and "
            "cannot be applied to the cache once before training"
        )
    raise ValueError(
        f"unknown projection_type={projection_type!r}; expected one of "
        f"{', '.join(PROJECTION_TYPES)}"
    )


def retained_energy(embeddings: torch.Tensor, projection: torch.Tensor) -> float:
    """Share of the cached embedding energy that survives the map's column space.

    Measured against an orthonormal basis of ``span(P)`` rather than against ``P``
    itself, so the number means "how much of the teacher does this subspace keep"
    for every arm. For a map that already has orthonormal columns (PCA, uncentered
    SVD, random orthonormal) this is exactly ``||X P||_F^2 / ||X||_F^2``; for the
    Gaussian map it strips the within-subspace scaling that would otherwise make
    the number incomparable.
    """
    matrix = embeddings.detach().to(torch.float32)
    basis, _ = torch.linalg.qr(projection.detach().to(torch.float32))
    total = matrix.pow(2).sum()
    return float((matrix @ basis).pow(2).sum() / total.clamp(min=1e-12))


def project_teacher_embeddings(
    embeddings: torch.Tensor,
    projection: torch.Tensor,
    mean: torch.Tensor | None = None,
    subtract_mean: bool = False,
    eps: float = 1e-12,
    renormalize: bool = True,
) -> torch.Tensor:
    """Apply Eq. (8): ``tau_i = norm(f_T(x_i) P_T)``.

    ``subtract_mean`` is off by default so that the applied map is exactly the linear
    ``P_T`` of the paper; turning it on makes the transform the textbook PCA one.
    ``renormalize=False`` skips the final ``norm(.)`` and returns ``f_T(x_i) P_T`` as
    is -- the target the MSE baseline regresses onto (sentence-transformers recipe).
    """
    matrix = embeddings.detach().to(torch.float32)
    if subtract_mean:
        if mean is None:
            raise ValueError("subtract_mean=True requires the fitted mean")
        matrix = matrix - mean.to(matrix.dtype)
    projected = matrix @ projection.to(matrix.dtype)
    if not renormalize:
        return projected
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
        the mean student-target cosine before and after the rotation, plus two
        summaries of the cross-covariance spectrum. ``N`` should comfortably exceed
        ``d_S`` for the cross-covariance to be well conditioned.

    ``participation_ratio`` is ``(sum sigma_i)^2 / sum sigma_i^2``, the effective
    number of directions ``R`` actually aligns. It is the diagnostic that says in
    advance whether the gauge can matter at all: at ``PR ~ 1`` the cross-covariance
    is rank-one, ``sigma_1 = ||mean tau|| ||mean z||``, and ``R`` does nothing but
    rotate one mean vector onto the other -- so the whole orientation factor is
    expected to be null on that student, and the run should say so rather than let
    the ablation look like noise. ``top_singular_share`` is ``sigma_1 / sum sigma_i``,
    the same statement read off the leading direction.

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
    u, sigma, vh = torch.linalg.svd(T.transpose(0, 1) @ Z, full_matrices=False)
    rotation = u @ vh
    before = float((T * Z).sum(dim=-1).mean())
    after = float(((T @ rotation) * Z).sum(dim=-1).mean())
    total = float(sigma.sum().clamp(min=1e-12))
    return rotation, {
        "cos_before": before,
        "cos_after": after,
        "samples": T.shape[0],
        "participation_ratio": total**2 / float(sigma.pow(2).sum().clamp(min=1e-12)),
        "top_singular_share": float(sigma[0]) / total,
    }


GAUGE_ROTATIONS = ("procrustes", "random", "interpolate", "rank_one")


def interpolate_rotation(
    start: torch.Tensor, end: torch.Tensor, theta: float
) -> tuple[torch.Tensor, bool]:
    """Geodesic on the rotation group from ``start`` (theta = 0) to ``end`` (theta = 1):
    ``Q(theta) = start expm(theta logm(start^T end))``.

    Returns ``(Q(theta), endpoint_reflected)``.

    O(d) has two connected components and a geodesic cannot leave the one it starts
    in, so when ``start`` is a reflection and ``end`` a rotation (or the other way
    round) **no continuous path between them exists at all** -- equivalently, a
    reflection has no real logarithm. The last column of ``end`` is then negated,
    which moves it into ``start``'s component; ``end`` is a Haar draw, so the flipped
    matrix is one just as well, and it is the nearest thing to the requested endpoint
    that O(d) actually contains. What it is *not* is the matrix a ``random`` arm with
    the same seed would use, so the caller is told whether this happened rather than
    left to assume the two ends of the curve coincide. Determinants of a Haar draw
    are +-1 with equal probability, so it happens for about half of all seeds.

    The logarithm is projected onto its skew-symmetric part before exponentiation,
    which makes every ``Q(theta)`` exactly orthogonal rather than orthogonal up to
    rounding.
    """
    import scipy.linalg

    if not 0.0 <= float(theta) <= 1.0:
        raise ValueError(f"theta must lie in [0, 1], got {theta!r}")
    a = start.detach().to(torch.float64).cpu().numpy()
    b = end.detach().to(torch.float64).cpu().numpy().copy()
    reflected = bool(np.linalg.det(a) * np.linalg.det(b) < 0)
    if reflected:
        b[:, -1] *= -1.0
    log = np.real(scipy.linalg.logm(a.T @ b))
    log = (log - log.T) / 2.0
    rotation = a @ scipy.linalg.expm(float(theta) * log)
    return torch.from_numpy(np.ascontiguousarray(rotation)).to(start.dtype), reflected


def rank_one_rotation(targets: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    """The rank-one part of the gauge: a Householder reflection sending the mean
    target direction onto the mean student direction and touching nothing else.

    On a student whose initial states are nearly one-dimensional the Procrustes
    cross-covariance is rank-one and its gauge can only do this much; this control
    says whether the full ``R`` ever does more.
    """
    u = F.normalize(F.normalize(targets.detach().to(torch.float32), dim=-1).mean(dim=0), dim=0)
    w = F.normalize(F.normalize(student.detach().to(torch.float32), dim=-1).mean(dim=0), dim=0)
    v = u - w
    eye = torch.eye(targets.shape[1], dtype=torch.float32)
    norm_sq = float(v @ v)
    if norm_sq < 1e-12:
        return eye
    return eye - 2.0 * torch.outer(v, v) / norm_sq


def fit_gauge_rotation(
    targets: torch.Tensor,
    student: torch.Tensor,
    mode: str = "procrustes",
    seed: int = 0,
    theta: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Pick the orientation factor by name.

    ``procrustes`` is the fitted gauge; ``random`` a Haar rotation of identical cost;
    ``interpolate`` the geodesic point ``Q(theta)`` between the two (theta = 0 is the
    Procrustes solution, theta = 1 the random one); ``rank_one`` the Householder map
    that aligns only the two mean directions.

    The Procrustes fit is solved either way, even when another rotation is the one
    returned: it costs one ``d_S x d_S`` SVD and it puts the number every control
    exists to be compared against (``cos_procrustes``) in the same log line and the
    same saved stats as the number the control achieved.
    """
    if mode not in GAUGE_ROTATIONS:
        raise ValueError(
            f"unknown gauge rotation {mode!r}; expected one of "
            f"{', '.join(GAUGE_ROTATIONS)}"
        )
    procrustes, stats = fit_gauge_alignment(targets, student)
    stats["rotation"] = mode
    if mode == "procrustes":
        return procrustes, stats

    T = F.normalize(targets.detach().to(torch.float32), dim=-1)
    Z = F.normalize(student.detach().to(torch.float32), dim=-1)
    if mode == "random":
        rotation = random_orthogonal(targets.shape[1], seed=seed)
        stats["rotation_seed"] = int(seed)
    elif mode == "interpolate":
        if theta is None:
            raise ValueError("gauge rotation 'interpolate' needs theta in [0, 1]")
        rotation, reflected = interpolate_rotation(
            procrustes, random_orthogonal(targets.shape[1], seed=seed), theta
        )
        stats["rotation_seed"] = int(seed)
        stats["theta"] = float(theta)
        # True means theta = 1 is *not* the gauge the 'random' arm with this seed
        # uses, so the right-hand end of the interpolation curve does not sit on the
        # random arm's plotted point. See interpolate_rotation.
        stats["endpoint_reflected"] = reflected
    else:
        rotation = rank_one_rotation(T, Z)
    # cos_after is always the cosine under the rotation that was actually applied.
    stats["cos_procrustes"] = stats["cos_after"]
    stats["cos_after"] = float(((T @ rotation) * Z).sum(dim=-1).mean())
    return rotation, stats
