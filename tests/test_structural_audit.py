"""The post-hoc metrics of the structural audit.

What is pinned here is what makes a rung a *rung*: every comparison above rung 1 is
invariant to an orthogonal rotation of either side (so the gauge cannot leak into
it), identical inputs sit at the ceiling, and unrelated inputs sit near the floor.
The dimension estimators are checked on data whose dimension is known.
"""

import math

import numpy as np
import pytest
import torch

from src.structural_audit import (
    anisotropy,
    apply_map,
    cosine_to_target,
    effective_rank,
    fit_variant,
    gram_correlation,
    gram_rmse,
    h0_barcode_distance,
    knn_indices,
    knn_overlap,
    ladder,
    linear_cka,
    mst_edge_weights,
    mutual_knn_jaccard,
    normalise_rung,
    pad_to,
    pair_average_precision,
    principal_angles,
    procrustes_distance,
    residual_spectrum,
    sts_spearman,
    targets_from_saved,
    truncation_curve,
    twonn_intrinsic_dimension,
)
from src.teacher_projection import random_orthogonal


def _data(n=300, d=16, seed=0):
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed))


@pytest.fixture
def x():
    return _data()


@pytest.fixture
def rotated(x):
    return x @ random_orthogonal(x.shape[1], seed=3)


# --------------------------------------------------------------------------- #
# Rotation invariance: the gauge shows on rung 1 and nowhere else
# --------------------------------------------------------------------------- #


def test_rung_one_sees_the_rotation_and_nothing_above_it_does(x, rotated):
    assert float(cosine_to_target(x, x).mean()) == pytest.approx(1.0, abs=1e-6)
    assert float(cosine_to_target(x, rotated).mean()) < 0.5

    assert gram_rmse(x, rotated) == pytest.approx(0.0, abs=1e-5)
    assert gram_correlation(x, rotated) == pytest.approx(1.0, abs=1e-5)
    assert linear_cka(x, rotated) == pytest.approx(1.0, abs=1e-5)
    assert procrustes_distance(x, rotated) == pytest.approx(0.0, abs=1e-4)
    assert knn_overlap(x, rotated, k=10) == pytest.approx(1.0)
    assert mutual_knn_jaccard(x, rotated, k=10) == pytest.approx(1.0)
    assert h0_barcode_distance(x, rotated) == pytest.approx(0.0, abs=1e-6)


def test_unrelated_embeddings_sit_near_the_floor(x):
    other = _data(seed=99)

    assert linear_cka(x, other) < 0.2
    assert procrustes_distance(x, other) > 1.0
    assert knn_overlap(x, other, k=10) < 0.15
    assert abs(gram_correlation(x, other)) < 0.15


def test_gram_rmse_measures_angular_distortion_of_an_interface():
    teacher = _data(n=500, d=64, seed=1)
    pca, _ = fit_variant(teacher, "pca", 32)
    random, _ = fit_variant(teacher, "random", 8, seed=0)

    faithful = gram_rmse(apply_map(teacher, pca), teacher)
    crude = gram_rmse(apply_map(teacher, random), teacher)
    assert faithful < crude


# --------------------------------------------------------------------------- #
# Neighbourhoods
# --------------------------------------------------------------------------- #


def test_knn_excludes_self_and_slices_consistently(x):
    table = knn_indices(x, k=20, chunk=64)
    assert table.shape == (x.shape[0], 20)
    assert not (table == torch.arange(x.shape[0]).unsqueeze(1)).any()
    # Reusing the k=20 table for k=5 must equal a fresh k=5 computation.
    assert knn_overlap(x, x, k=5, neighbours_a=table, neighbours_b=table) == pytest.approx(1.0)
    assert torch.equal(table[:, :5], knn_indices(x, k=5, chunk=64))


def test_knn_rejects_k_not_smaller_than_n():
    with pytest.raises(ValueError, match="must be smaller"):
        knn_indices(_data(n=10), k=10)


# --------------------------------------------------------------------------- #
# Connectivity
# --------------------------------------------------------------------------- #


def test_mst_has_n_minus_one_edges_and_sees_a_split():
    together = _data(n=200, d=8, seed=5)
    apart = torch.cat([together[:100] + 20.0, together[100:] - 20.0])

    assert mst_edge_weights(together).shape == (199,)
    # Two far-apart clusters have one very long bridging edge; the barcode moves.
    assert h0_barcode_distance(together, apart) > 0.0


# --------------------------------------------------------------------------- #
# Ranks and dimensions
# --------------------------------------------------------------------------- #


def test_effective_rank_reads_the_spectrum():
    isotropic = _data(n=2000, d=16, seed=7)
    collapsed = torch.outer(torch.randn(2000, generator=torch.Generator().manual_seed(8)), torch.ones(16))

    assert effective_rank(isotropic) == pytest.approx(16.0, abs=0.5)
    assert effective_rank(collapsed) == pytest.approx(1.0, abs=1e-3)


def test_twonn_recovers_the_dimension_of_a_planar_manifold():
    generator = torch.Generator().manual_seed(9)
    plane = torch.randn(3000, 2, generator=generator)
    embedded = plane @ torch.randn(2, 12, generator=generator)

    assert twonn_intrinsic_dimension(embedded) == pytest.approx(2.0, abs=0.4)


def test_principal_angles_vanish_for_the_same_span_and_grow_for_random_spans():
    basis = _data(n=32, d=6, seed=10)
    same = basis @ random_orthogonal(6, seed=11)
    other = _data(n=32, d=6, seed=12)

    assert np.allclose(principal_angles(basis, same), 0.0, atol=1e-4)
    assert principal_angles(basis, other).max() > 0.5


def test_residual_spectrum_is_flat_for_noise_and_peaked_for_a_shared_offset(x):
    noisy = residual_spectrum(x, x + 0.1 * _data(seed=13))
    offset = residual_spectrum(x, x + 3.0 * torch.ones_like(x))

    assert noisy["target"].shape == (16,)
    assert offset["residual"][0] / offset["residual"][1:].sum() > noisy["residual"][0] / noisy["residual"][1:].sum()


def test_anisotropy_is_zero_for_isotropic_and_one_for_collapsed():
    assert abs(anisotropy(_data(n=2000, d=32, seed=14), n_pairs=20000)) < 0.05
    ones = torch.ones(50, 8)
    assert anisotropy(ones, n_pairs=1000) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #


def test_targets_from_saved_replays_the_map_and_refuses_a_learned_arm():
    teacher = _data(n=100, d=32, seed=15)
    projection, mean = fit_variant(teacher, "pca", 8)
    gauge = random_orthogonal(8, seed=16)
    saved = {"projection": projection, "mean": mean, "pca_subtract_mean": False, "gauge_matrix": gauge}

    expected = torch.nn.functional.normalize((teacher @ projection) @ gauge, dim=-1)
    assert torch.allclose(targets_from_saved(teacher, saved), expected, atol=1e-5)

    with pytest.raises(ValueError, match="learned map"):
        targets_from_saved(teacher, {"projection": None, "projection_type": "learned_t2s"})


def test_pad_to_widens_a_rank_k_target_without_moving_it():
    narrow = _data(n=10, d=4)
    wide = pad_to(narrow, 8)
    assert wide.shape == (10, 8)
    assert torch.equal(wide[:, :4], narrow)
    assert torch.equal(wide[:, 4:], torch.zeros(10, 4))
    assert torch.equal(pad_to(wide, 4), narrow)


# --------------------------------------------------------------------------- #
# Downstream from embeddings
# --------------------------------------------------------------------------- #


def test_sts_and_pair_scores_read_cosine_between_probe_rows():
    generator = torch.Generator().manual_seed(17)
    emb = torch.randn(60, 8, generator=generator)
    left = np.arange(0, 30)
    right = np.arange(30, 60)
    cosine = (torch.nn.functional.normalize(emb[left], dim=-1) * torch.nn.functional.normalize(emb[right], dim=-1)).sum(-1).numpy()

    assert sts_spearman(emb, left, right, cosine) == pytest.approx(1.0)
    labels = (cosine > np.median(cosine)).astype(int)
    assert pair_average_precision(emb, left, right, labels) == pytest.approx(1.0)

    pairs = {"sts": {"left": left, "right": right, "target": cosine, "kind": "sts"}}
    rows = truncation_curve(emb, pairs, dims=[8, 2])
    assert [row["dim"] for row in rows] == [8, 2]
    assert rows[0]["score"] == pytest.approx(1.0)
    assert rows[1]["score"] < 1.0


# --------------------------------------------------------------------------- #
# The ladder as a whole
# --------------------------------------------------------------------------- #


def test_ladder_reports_every_rung_and_the_ceiling_normalises_to_one(x, rotated):
    rungs = ladder(rotated, x, target=x, ks=(1, 5), h0_rows=100, h0_draws=2)

    for key in ("cos_to_target", "gram_rmse", "gram_corr", "linear_cka", "procrustes",
                "knn@1", "knn@5", "mutual_knn@1", "mutual_knn@5", "h0_w1"):
        assert key in rungs and math.isfinite(rungs[key]), key
    assert rungs["linear_cka"] == pytest.approx(1.0, abs=1e-5)
    assert rungs["cos_to_target"] < 0.5

    assert normalise_rung(0.9, ceiling=0.9, floor=0.1) == pytest.approx(1.0)
    assert normalise_rung(0.1, ceiling=0.9, floor=0.1) == pytest.approx(0.0)
    assert math.isnan(normalise_rung(0.5, ceiling=0.5, floor=0.5))
