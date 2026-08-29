"""Post-hoc measurements of the structural audit (protocol §1–§2).

Every number the audit reports is computed *after* training, from three things:

* the probe set (:mod:`src.probe_set`) and the teacher's embeddings of it;
* per-epoch student weights, encoded on the probe set (:func:`encode_texts`);
* the frozen target map each arm trained against (``teacher_projection.pt``).

Nothing here touches training, so a metric can be added or changed without
rerunning an arm. The functions are grouped by the rung of the structural ladder
they belong to:

1. coordinates -- cosine to the target (:func:`cosine_to_target`);
2. second-order structure -- Gram distance, linear CKA, Procrustes distance
   (Kornblith et al. 2019; Ding et al. 2021);
3. neighbourhoods -- k-NN overlap and mutual-k-NN agreement at several probe sizes;
4. connectivity -- the H0 persistence barcode, i.e. the distribution of minimum
   spanning tree edge lengths, compared by Wasserstein-1.

Plus the quantities of Table 2 and the depth/spectrum figures: effective rank
(RankMe), TwoNN intrinsic dimension, principal angles between subspaces, residual
spectra, anisotropy, and the STS / pair-classification scores an embedding matrix
gets on its own (the "ceiling" of a target and the truncation curve of a student).

Every metric that compares two embedding sets is invariant to an orthogonal
rotation of either set, except rung 1 -- which is the point: rung 1 is the only
place the gauge shows, everything above it is geometry.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.metrics import average_precision_score

from src.pooling import pool_sentence_embedding
from src.teacher_projection import (  # noqa: F401  (retained_energy is re-exported)
    fit_teacher_projection,
    project_teacher_embeddings,
    retained_energy,
)

# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #


def load_student(
    model_name: str,
    weights_path: str | os.PathLike[str] | None = None,
    device: str | torch.device = "cpu",
):
    """The student encoder at a checkpoint: the pretrained weights, then a state dict.

    ``weights_path`` is either a ``student_epoch_N.pt`` file (``--weights_dir``) or a
    full ``checkpoint_epoch_N.pt``; both carry ``model_state_dict``. ``None`` is the
    untrained student, i.e. step 0.
    """
    from transformers import AutoModel
    from transformers import __version__ as transformers_version

    try:
        major = int(transformers_version.split(".", maxsplit=1)[0])
    except (TypeError, ValueError):
        major = 4
    kwargs = {("dtype" if major >= 5 else "torch_dtype"): torch.float32}
    model = AutoModel.from_pretrained(model_name, **kwargs)
    if weights_path is not None:
        payload = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any("embeddings" in name or "layer" in name for name in missing):
            raise RuntimeError(
                f"{weights_path} does not match {model_name}: "
                f"missing={sorted(missing)[:5]} unexpected={sorted(unexpected)[:5]}"
            )
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def encode_texts(
    model,
    tokenizer,
    texts: list[str],
    *,
    device: str | torch.device = "cpu",
    pooling: str = "cls",
    batch_size: int = 256,
    max_length: int = 256,
    layers: bool = False,
    amp: bool = True,
    progress: bool = False,
) -> dict[str, torch.Tensor | None]:
    """Pooled, **unnormalised** sentence vectors of ``texts``, in their original order.

    Returns ``{"final": [N, d] float32, "layers": [L+1, N, d] float16 or None}``.
    ``layers`` holds the pooled state of the embedding output (index 0) and of
    every Transformer layer, which is what the depth profiles read. Batches are
    formed over length-sorted indices, as the evaluation code does, so padding is
    minimal and the per-sentence result is unchanged.
    """
    total = len(texts)
    order = sorted(range(total), key=lambda index: len(texts[index]))
    batches = [order[start : start + batch_size] for start in range(0, total, batch_size)]
    iterator = batches
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(batches, leave=False)

    final = None
    per_layer = None
    use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
    autocast = torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp and use_cuda)
    with autocast:
        for indices in iterator:
            encoded = tokenizer(
                [texts[index] for index in indices],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=layers,
                return_dict=True,
            )
            pooled = pool_sentence_embedding(output.last_hidden_state, attention_mask, pooling)
            pooled = pooled.float().cpu()
            if final is None:
                final = torch.empty(total, pooled.shape[1], dtype=torch.float32)
            final[indices] = pooled
            if layers:
                stack = torch.stack(
                    [
                        pool_sentence_embedding(state, attention_mask, pooling).to(torch.float16).cpu()
                        for state in output.hidden_states
                    ]
                )
                if per_layer is None:
                    per_layer = torch.empty(
                        stack.shape[0], total, stack.shape[2], dtype=torch.float16
                    )
                per_layer[:, indices] = stack
    if final is None:
        final = torch.empty(0, 0)
    return {"final": final, "layers": per_layer}


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #


def load_saved_projection(path: str | os.PathLike[str]) -> dict:
    """The ``teacher_projection.pt`` a run wrote: the map it actually trained against."""
    return torch.load(path, map_location="cpu", weights_only=False)


def apply_map(
    teacher: torch.Tensor,
    projection: torch.Tensor,
    mean: torch.Tensor | None = None,
    subtract_mean: bool = False,
    gauge: torch.Tensor | None = None,
    renormalize: bool = True,
) -> torch.Tensor:
    """``tau = norm(f_T(x) P) R`` -- Eq. (8) with the optional gauge, on any embeddings."""
    targets = project_teacher_embeddings(
        teacher, projection, mean=mean, subtract_mean=subtract_mean, renormalize=renormalize
    )
    if gauge is not None:
        targets = targets @ gauge.to(targets.dtype)
        if renormalize:
            targets = F.normalize(targets, dim=-1)
    return targets


def targets_from_saved(teacher: torch.Tensor, saved: dict, renormalize: bool = True) -> torch.Tensor:
    """Apply a run's saved frozen map (and gauge) to new teacher embeddings."""
    if saved.get("projection") is None:
        raise ValueError(
            f"{saved.get('projection_type')!r} is a learned map; its targets are "
            "W applied to the teacher, read W from the checkpoint instead"
        )
    return apply_map(
        teacher,
        saved["projection"],
        mean=saved.get("mean"),
        subtract_mean=bool(saved.get("pca_subtract_mean", False)),
        gauge=saved.get("gauge_matrix"),
        renormalize=renormalize,
    )


def fit_variant(
    cache: torch.Tensor,
    kind: str,
    k: int,
    seed: int = 0,
    center: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A rank-``k`` fixed interface (``pca`` / ``random`` / ``random_gaussian`` /
    ``mrl_prefix``) fitted on the training cache: the PCA-k / random-k / MRL-k
    family of Figures 6–7."""
    return fit_teacher_projection(cache, out_dim=k, projection_type=kind, center=center, seed=seed)


def pad_to(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Zero-pad columns up to ``dim`` (a rank-k target inside a d_S-wide student)."""
    if x.shape[1] >= dim:
        return x[:, :dim]
    return torch.cat([x, x.new_zeros(x.shape[0], dim - x.shape[1])], dim=1)


# --------------------------------------------------------------------------- #
# Basic geometry
# --------------------------------------------------------------------------- #


def _as_float(x) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.detach().to(torch.float32)


def unit(x) -> torch.Tensor:
    return F.normalize(_as_float(x), dim=-1)


def cosine_to_target(z, tau) -> torch.Tensor:
    """Rung 1, per row: ``<z_i, tau_i>`` on the sphere."""
    return (unit(z) * unit(tau)).sum(dim=-1)


def effective_rank(x) -> float:
    """RankMe (Garrido et al. 2023): ``exp(H(p))`` with ``p_i = sigma_i / sum sigma``."""
    sigma = torch.linalg.svdvals(_as_float(x))
    p = sigma / sigma.sum().clamp(min=1e-12)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def singular_values(x, center: bool = True) -> np.ndarray:
    matrix = _as_float(x)
    if center:
        matrix = matrix - matrix.mean(dim=0)
    return torch.linalg.svdvals(matrix).numpy()


def random_pairs(n: int, n_pairs: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``n_pairs`` random index pairs ``(i, j)`` with ``i != j``."""
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n - 1, size=n_pairs)
    j = j + (j >= i)
    return i, j


def pairwise_cosines(x, pairs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    u = unit(x)
    i, j = pairs
    return (u[i] * u[j]).sum(dim=-1).numpy()


def anisotropy(x, n_pairs: int = 100_000, seed: int = 0) -> float:
    """Mean cosine between random pairs (Ethayarajh 2019): 0 isotropic, 1 collapsed."""
    return float(pairwise_cosines(x, random_pairs(_as_float(x).shape[0], n_pairs, seed)).mean())


# --------------------------------------------------------------------------- #
# Rung 2: second-order structure
# --------------------------------------------------------------------------- #


def _subsample(a, b, max_rows: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    a, b = _as_float(a), _as_float(b)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"row mismatch: {a.shape[0]} vs {b.shape[0]}")
    if a.shape[0] > max_rows:
        rng = np.random.default_rng(seed)
        keep = torch.from_numpy(np.sort(rng.choice(a.shape[0], size=max_rows, replace=False)))
        a, b = a[keep], b[keep]
    return a, b


def _offdiag(gram: torch.Tensor) -> torch.Tensor:
    n = gram.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    return gram[mask]


def gram_rmse(a, b, max_rows: int = 4096, seed: int = 0) -> float:
    """RMS difference of the two cosine Gram matrices off the diagonal.

    The "angular distortion" of the audit: for a target against the full teacher
    it says how far the interface bends pairwise angles; for a student against the
    teacher it is rung 2 in its most literal form.
    """
    a, b = _subsample(a, b, max_rows, seed)
    ga, gb = unit(a) @ unit(a).T, unit(b) @ unit(b).T
    return float(torch.sqrt(((_offdiag(ga) - _offdiag(gb)) ** 2).mean()))


def gram_correlation(a, b, max_rows: int = 4096, seed: int = 0) -> float:
    """Pearson correlation of the off-diagonal cosine Gram entries."""
    a, b = _subsample(a, b, max_rows, seed)
    x = _offdiag(unit(a) @ unit(a).T)
    y = _offdiag(unit(b) @ unit(b).T)
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm()).clamp(min=1e-12))


def linear_cka(a, b) -> float:
    """Linear CKA (Kornblith et al. 2019), feature-space form; 1 = identical up to
    isotropic scaling and rotation."""
    a, b = _as_float(a), _as_float(b)
    a = a - a.mean(dim=0)
    b = b - b.mean(dim=0)
    cross = (b.T @ a).norm() ** 2
    return float(cross / ((a.T @ a).norm() * (b.T @ b).norm()).clamp(min=1e-12))


def procrustes_distance(a, b) -> float:
    """Orthogonal Procrustes distance (Ding et al. 2021) on centred, Frobenius-
    normalised matrices: ``2 - 2 ||A^T B||_*``, 0 = identical up to rotation."""
    a, b = _as_float(a), _as_float(b)
    a = a - a.mean(dim=0)
    b = b - b.mean(dim=0)
    a = a / a.norm().clamp(min=1e-12)
    b = b / b.norm().clamp(min=1e-12)
    nuclear = torch.linalg.svdvals(a.T @ b).sum()
    return float(2.0 - 2.0 * nuclear)


# --------------------------------------------------------------------------- #
# Rung 3: neighbourhoods
# --------------------------------------------------------------------------- #


def knn_indices(x, k: int, chunk: int = 2048, device: str | torch.device | None = None) -> torch.Tensor:
    """Indices of the ``k`` nearest neighbours by cosine, self excluded: ``[N, k]``."""
    u = unit(x)
    n = u.shape[0]
    if k >= n:
        raise ValueError(f"k={k} must be smaller than the number of rows {n}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    u = u.to(device)
    out = torch.empty(n, k, dtype=torch.long)
    for start in range(0, n, chunk):
        sims = u[start : start + chunk] @ u.T
        rows = torch.arange(sims.shape[0], device=device)
        sims[rows, rows + start] = -math.inf
        out[start : start + chunk] = sims.topk(k, dim=1).indices.cpu()
    return out


def knn_overlap(a, b, k: int, chunk: int = 2048, neighbours_a=None, neighbours_b=None) -> float:
    """Mean ``|N_a(i) ∩ N_b(i)| / k`` -- the k-NN overlap (mutual-kNN alignment of
    Huh et al. 2024). Pass precomputed neighbour tables to reuse them across k."""
    na = knn_indices(a, k, chunk) if neighbours_a is None else neighbours_a[:, :k]
    nb = knn_indices(b, k, chunk) if neighbours_b is None else neighbours_b[:, :k]
    n = na.shape[0]
    # Membership via a [N, N]-free trick: sort and count matches per row.
    sa, sb = na.sort(dim=1).values, nb.sort(dim=1).values
    both = torch.cat([sa, sb], dim=1).sort(dim=1).values
    dup = (both[:, 1:] == both[:, :-1]).sum(dim=1)
    return float(dup.float().mean() / k) if n else float("nan")


def mutual_knn_edges(neighbours: torch.Tensor) -> set[tuple[int, int]]:
    """Edges ``(i, j)``, ``i < j``, where each point is in the other's k-NN list."""
    n, k = neighbours.shape
    rows = torch.arange(n).unsqueeze(1).expand(n, k)
    pairs = torch.stack([rows.reshape(-1), neighbours.reshape(-1)], dim=1)
    directed = set(map(tuple, pairs.tolist()))
    return {(i, j) for i, j in directed if i < j and (j, i) in directed}


def mutual_knn_jaccard(a, b, k: int, chunk: int = 2048, neighbours_a=None, neighbours_b=None) -> float:
    """Jaccard overlap of the two mutual-k-NN graphs' edge sets."""
    na = knn_indices(a, k, chunk) if neighbours_a is None else neighbours_a[:, :k]
    nb = knn_indices(b, k, chunk) if neighbours_b is None else neighbours_b[:, :k]
    ea, eb = mutual_knn_edges(na), mutual_knn_edges(nb)
    union = len(ea | eb)
    return float(len(ea & eb) / union) if union else float("nan")


# --------------------------------------------------------------------------- #
# Rung 4: connectivity (H0 persistence)
# --------------------------------------------------------------------------- #


def mst_edge_weights(x) -> np.ndarray:
    """H0 death times: the edge lengths of the minimum spanning tree under cosine
    distance. ``N - 1`` values; their distribution is the H0 barcode."""
    u = unit(x)
    distance = (1.0 - u @ u.T).clamp(min=0.0).numpy().astype(np.float64)
    # scipy treats exact zeros as "no edge", so every edge gets a floor.
    distance = distance + 1e-9
    np.fill_diagonal(distance, 0.0)
    tree = minimum_spanning_tree(distance)
    return np.sort(tree.data)


def h0_barcode_distance(a, b) -> float:
    """Wasserstein-1 between the two MST edge-length distributions."""
    return float(wasserstein_distance(mst_edge_weights(a), mst_edge_weights(b)))


def h0_barcode_distance_sampled(a, b, n: int = 2000, draws: int = 3, seed: int = 0) -> tuple[float, float]:
    """The same on ``draws`` random subsamples of ``n`` rows: mean and std."""
    a, b = _as_float(a), _as_float(b)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        keep = torch.from_numpy(np.sort(rng.choice(a.shape[0], size=min(n, a.shape[0]), replace=False)))
        values.append(h0_barcode_distance(a[keep], b[keep]))
    return float(np.mean(values)), float(np.std(values))


# --------------------------------------------------------------------------- #
# Depth and spectra
# --------------------------------------------------------------------------- #


def twonn_intrinsic_dimension(x, discard_fraction: float = 0.1, max_rows: int = 4096, seed: int = 0) -> float:
    """TwoNN (Facco et al. 2017): fit ``-log(1 - F(mu))`` against ``log mu`` through
    the origin, ``mu = r_2 / r_1`` the ratio of the two nearest-neighbour distances."""
    matrix = _as_float(x)
    if matrix.shape[0] > max_rows:
        rng = np.random.default_rng(seed)
        keep = torch.from_numpy(np.sort(rng.choice(matrix.shape[0], size=max_rows, replace=False)))
        matrix = matrix[keep]
    distance = torch.cdist(matrix, matrix)
    distance.fill_diagonal_(math.inf)
    r1, r2 = distance.topk(2, dim=1, largest=False).values.unbind(dim=1)
    valid = r1 > 0
    mu = (r2[valid] / r1[valid]).sort().values.numpy()
    n = len(mu)
    if n < 10:
        return float("nan")
    keep = int(math.floor(n * (1.0 - discard_fraction)))
    mu = mu[:keep]
    empirical = np.arange(1, keep + 1) / n
    xs = np.log(mu)
    ys = -np.log(1.0 - empirical)
    return float((xs * ys).sum() / max((xs * xs).sum(), 1e-12))


def principal_angles(a_basis, b_basis) -> np.ndarray:
    """Principal angles (radians) between ``span(A)`` and ``span(B)``, ``[d, k]`` each."""
    # float64: arccos loses everything near 1, and "the same span" is exactly
    # the case where every cosine is 1 up to rounding.
    qa, _ = torch.linalg.qr(_as_float(a_basis).double())
    qb, _ = torch.linalg.qr(_as_float(b_basis).double())
    cosines = torch.linalg.svdvals(qa.T @ qb).clamp(-1.0, 1.0)
    return torch.arccos(cosines).numpy()


def residual_spectrum(z, tau) -> dict[str, np.ndarray]:
    """Singular values of the residual ``norm(z) - tau`` next to those of ``tau``
    (Observation 3.1): few dominant residual directions = a shared offset."""
    residual = unit(z) - unit(tau)
    return {
        "residual": torch.linalg.svdvals(residual).numpy(),
        "target": torch.linalg.svdvals(unit(tau)).numpy(),
    }


def layer_drift(layers_now, layers_init) -> list[dict[str, float]]:
    """Per layer: CKA and Procrustes distance to the step-0 state of the same layer."""
    rows = []
    for index, (now, init) in enumerate(zip(layers_now, layers_init)):
        rows.append(
            {
                "layer": index,
                "cka_to_init": linear_cka(now, init),
                "procrustes_to_init": procrustes_distance(now, init),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Downstream scores from an embedding matrix
# --------------------------------------------------------------------------- #


def sts_spearman(emb, left: np.ndarray, right: np.ndarray, scores: np.ndarray) -> float:
    """Spearman between pairwise cosine and the gold score (the affine rescaling
    the evaluation code applies is monotone, so it changes nothing)."""
    u = unit(emb)
    cosine = (u[left] * u[right]).sum(dim=-1).numpy()
    return float(spearmanr(cosine, scores).correlation)


def pair_average_precision(emb, left: np.ndarray, right: np.ndarray, labels: np.ndarray) -> float:
    u = unit(emb)
    cosine = (u[left] * u[right]).sum(dim=-1).numpy()
    return float(average_precision_score(labels.astype(int), (cosine + 1.0) / 2.0))


def pair_task_scores(emb, pairs: dict[str, dict]) -> dict[str, float]:
    """Score every task of :func:`src.probe_set.eval_pairs_in_probe` on ``emb``."""
    out = {}
    for name, spec in pairs.items():
        if len(spec["left"]) == 0:
            out[name] = float("nan")
        elif spec["kind"] == "sts":
            out[name] = sts_spearman(emb, spec["left"], spec["right"], spec["target"])
        else:
            out[name] = pair_average_precision(emb, spec["left"], spec["right"], spec["target"])
    return out


def truncation_curve(emb, pairs: dict[str, dict], dims: list[int]) -> list[dict]:
    """MRL-style: scores of the leading ``k`` coordinates for every ``k`` in ``dims``."""
    rows = []
    matrix = _as_float(emb)
    for k in dims:
        scores = pair_task_scores(matrix[:, :k], pairs)
        rows.extend({"dim": k, "task": task, "score": value} for task, value in scores.items())
    return rows


# --------------------------------------------------------------------------- #
# The ladder
# --------------------------------------------------------------------------- #


def ladder(
    student,
    teacher,
    target=None,
    *,
    ks: tuple[int, ...] = (1, 10, 50),
    gram_rows: int = 4096,
    h0_rows: int = 2000,
    h0_draws: int = 3,
    seed: int = 0,
    chunk: int = 2048,
) -> dict[str, float]:
    """All four rungs, student vs teacher on the same rows.

    ``target`` (the arm's own frozen target, or any reference in the student
    space) feeds rung 1 only; rungs 2–4 compare the student with the *full*
    teacher and never need the target. Neighbour tables are computed once at the
    largest ``k`` and sliced for the smaller ones.
    """
    out: dict[str, float] = {}
    if target is not None:
        out["cos_to_target"] = float(cosine_to_target(student, target).mean())
    out["gram_rmse"] = gram_rmse(student, teacher, max_rows=gram_rows, seed=seed)
    out["gram_corr"] = gram_correlation(student, teacher, max_rows=gram_rows, seed=seed)
    out["linear_cka"] = linear_cka(student, teacher)
    out["procrustes"] = procrustes_distance(student, teacher)
    kmax = max(ks)
    na = knn_indices(student, kmax, chunk)
    nb = knn_indices(teacher, kmax, chunk)
    for k in ks:
        out[f"knn@{k}"] = knn_overlap(student, teacher, k, neighbours_a=na, neighbours_b=nb)
        out[f"mutual_knn@{k}"] = mutual_knn_jaccard(student, teacher, k, neighbours_a=na, neighbours_b=nb)
    mean, std = h0_barcode_distance_sampled(student, teacher, n=h0_rows, draws=h0_draws, seed=seed)
    out["h0_w1"] = mean
    out["h0_w1_std"] = std
    return out


# Direction of every rung: +1 when larger means closer to the teacher, -1 when
# smaller does. Used to normalise the ladder so the ceiling reads 1 and the floor 0.
RUNG_SIGN = {
    "cos_to_target": +1,
    "gram_rmse": -1,
    "gram_corr": +1,
    "linear_cka": +1,
    "procrustes": -1,
    "h0_w1": -1,
}


def normalise_rung(value: float, ceiling: float, floor: float) -> float:
    """Affine map with the ceiling at 1 and the floor at 0 (sign-agnostic)."""
    span = ceiling - floor
    if abs(span) < 1e-12:
        return float("nan")
    return float((value - floor) / span)
