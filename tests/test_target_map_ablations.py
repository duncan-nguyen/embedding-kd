"""Controls for the two factors of the frozen target map ``P_T = P_PCA R``.

The recipe makes two separate claims, and each needs its own null:

* the *subspace* claim (Eckart-Young): the teacher's leading spectral subspace is
  what carries the signal, so centered PCA must beat a random subspace of the same
  rank. ``projection_type`` switches between them, and ``pca_center_fit`` switches
  the spectral arm between centered PCA and the uncentered SVD that is allowed to
  spend its first direction on the teacher's mean vector.
* the *orientation* claim (Schoenemann): the gauge fitted to the student init is
  better than the arbitrary gauge PCA happens to return. Since PCA's own
  coordinates are already arbitrary, the sharp control is not "no rotation" but a
  Haar-random rotation of identical cost -- ``gauge_rotation="random"``.

These tests pin the arms themselves (determinism, what each one is, that they are
genuinely different) rather than their downstream scores, which only training can
report.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from config import GeoODEConfig
from distiller import KnowledgeDistiller
from main import get_config, parse_args
from scripts.ablations import run_target_map_ablation as ablation

REPO_ROOT = Path(__file__).resolve().parent.parent
from src.criterions.geoode_kd import GeoODEKD
from src.target_projector import LearnedTargetProjector
from src.teacher_projection import (
    fit_gauge_alignment,
    fit_gauge_rotation,
    fit_pca_projection,
    fit_random_projection,
    fit_teacher_projection,
    interpolate_rotation,
    project_teacher_embeddings,
    random_orthogonal,
    retained_energy,
)


def _spectral_data(rows: int = 400, dim: int = 32, seed: int = 0) -> torch.Tensor:
    """Embeddings with a hard spectral gap: all the energy is in two directions."""
    generator = torch.Generator().manual_seed(seed)
    latent = torch.randn(rows, 2, generator=generator)
    basis = torch.zeros(2, dim)
    basis[0, 0] = 10.0
    basis[1, 1] = 6.0
    return latent @ basis + 1e-2 * torch.randn(rows, dim, generator=generator)


# --------------------------------------------------------------------------- #
# Factor 1: the subspace
# --------------------------------------------------------------------------- #


def test_random_projection_has_the_same_contract_as_the_pca_map():
    embeddings = torch.randn(64, 32)

    projection, mean = fit_random_projection(embeddings, out_dim=8, seed=0)

    assert projection.shape == (32, 8)
    assert mean.shape == (32,)
    # Orthonormal columns, exactly like the PCA map: the two arms differ only in
    # *which* subspace they pick, never in the kind of map they are.
    assert torch.allclose(projection.T @ projection, torch.eye(8), atol=1e-5)
    targets = project_teacher_embeddings(embeddings, projection, mean=mean)
    assert torch.allclose(targets.norm(dim=-1), torch.ones(64), atol=1e-5)


def test_random_projection_is_reproducible_and_seed_dependent():
    embeddings = torch.randn(64, 32)

    first, _ = fit_random_projection(embeddings, out_dim=8, seed=3)
    again, _ = fit_random_projection(embeddings, out_dim=8, seed=3)
    other, _ = fit_random_projection(embeddings, out_dim=8, seed=4)

    assert torch.equal(first, again)
    # Separate draws of the control; their spread is the null band PCA must clear.
    assert not torch.allclose(first, other, atol=1e-3)


def test_random_projection_ignores_the_global_rng():
    """The arm must be a property of its seed, not of where the run called it."""
    embeddings = torch.randn(64, 32)

    torch.manual_seed(1234)
    first, _ = fit_random_projection(embeddings, out_dim=8, seed=5)
    torch.manual_seed(99)
    torch.randn(1000)  # advance the global generator
    again, _ = fit_random_projection(embeddings, out_dim=8, seed=5)

    assert torch.equal(first, again)


def test_random_projection_does_not_look_at_the_data():
    projection, _ = fit_random_projection(_spectral_data(), out_dim=8, seed=6)
    other, _ = fit_random_projection(torch.randn(400, 32), out_dim=8, seed=6)

    assert torch.equal(projection, other)


def test_gaussian_and_orthonormal_draws_share_a_subspace():
    """At one seed the two random arms span the same subspace, so comparing them
    isolates orthonormality with the subspace held fixed."""
    embeddings = _spectral_data()

    orthonormal, _ = fit_random_projection(embeddings, out_dim=8, seed=7)
    gaussian, _ = fit_random_projection(
        embeddings, out_dim=8, seed=7, orthonormal=False
    )

    assert not torch.allclose(
        gaussian.T @ gaussian, torch.eye(8), atol=1e-2
    )  # the Gaussian map is not an isometry
    assert retained_energy(embeddings, gaussian) == pytest.approx(
        retained_energy(embeddings, orthonormal), abs=1e-5
    )


def test_pca_retains_far_more_energy_than_a_random_subspace():
    """The Eckart-Young claim, measured: this is the gap the ablation is testing."""
    embeddings = _spectral_data()

    pca, _ = fit_pca_projection(embeddings, out_dim=4)
    control, _ = fit_random_projection(embeddings, out_dim=4, seed=8)

    assert retained_energy(embeddings, pca) > 0.99
    assert retained_energy(embeddings, control) < 0.5


def test_uncentered_svd_spends_its_first_direction_on_the_mean():
    """What ``--no-pca_center_fit`` costs: with an off-origin corpus the leading
    uncentered direction is the mean vector, not a direction of variation."""
    generator = torch.Generator().manual_seed(9)
    offset = torch.zeros(32)
    offset[5] = 50.0
    embeddings = _spectral_data(seed=9) + offset

    centered, _ = fit_pca_projection(embeddings, out_dim=1, center=True)
    uncentered, _ = fit_pca_projection(embeddings, out_dim=1, center=False)

    mean_direction = torch.nn.functional.normalize(embeddings.mean(dim=0), dim=0)
    assert abs(float(uncentered[:, 0] @ mean_direction)) > 0.99
    assert abs(float(centered[:, 0] @ mean_direction)) < 0.1


def test_fit_teacher_projection_dispatches_every_arm():
    embeddings = _spectral_data()

    for kind in ("pca", "random", "random_gaussian", "mrl_prefix"):
        projection, mean = fit_teacher_projection(
            embeddings, out_dim=8, projection_type=kind, seed=10
        )
        assert projection.shape == (32, 8)
        assert mean.shape == (32,)

    # The Matryoshka prefix is the leading coordinates: orthonormal, data-blind.
    prefix, _ = fit_teacher_projection(embeddings, out_dim=8, projection_type="mrl_prefix")
    assert torch.equal(prefix, torch.eye(32)[:, :8])
    assert torch.equal(prefix, fit_teacher_projection(torch.randn(5, 32), out_dim=8, projection_type="mrl_prefix")[0])

    expected, _ = fit_pca_projection(embeddings, out_dim=8, center=False)
    dispatched, _ = fit_teacher_projection(
        embeddings, out_dim=8, projection_type="pca", center=False
    )
    assert torch.equal(dispatched, expected)


def test_unknown_projection_type_is_rejected():
    with pytest.raises(ValueError, match="unknown projection_type"):
        fit_teacher_projection(torch.randn(16, 32), out_dim=8, projection_type="svd")


def test_retained_energy_agrees_with_the_direct_ratio_for_isometries():
    embeddings = _spectral_data()
    projection, _ = fit_pca_projection(embeddings, out_dim=6)

    direct = float(
        (embeddings @ projection).pow(2).sum() / embeddings.pow(2).sum()
    )
    assert retained_energy(embeddings, projection) == pytest.approx(direct, abs=1e-5)


# --------------------------------------------------------------------------- #
# Factor 2: the orientation
# --------------------------------------------------------------------------- #


def test_random_orthogonal_is_orthogonal_and_reproducible():
    first = random_orthogonal(16, seed=11)
    again = random_orthogonal(16, seed=11)
    other = random_orthogonal(16, seed=12)

    assert torch.allclose(first @ first.T, torch.eye(16), atol=1e-5)
    assert torch.equal(first, again)
    assert not torch.allclose(first, other, atol=1e-3)


def test_random_gauge_rotates_without_aligning():
    """The control's whole content: it moves the coordinates exactly as much as
    Procrustes does, and gains none of the alignment."""
    generator = torch.Generator().manual_seed(13)
    targets = torch.nn.functional.normalize(torch.randn(400, 16, generator=generator), dim=-1)
    student = torch.nn.functional.normalize(
        targets + 0.9 * torch.randn(400, 16, generator=generator), dim=-1
    )

    rotation, stats = fit_gauge_rotation(targets, student, mode="random", seed=14)

    assert torch.allclose(rotation @ rotation.T, torch.eye(16), atol=1e-5)
    # Orthogonal either way, so the relational geometry is untouched by both arms.
    rotated = targets @ rotation
    assert torch.allclose(rotated @ rotated.T, targets @ targets.T, atol=1e-4)
    assert stats["rotation"] == "random"
    assert stats["cos_after"] < stats["cos_procrustes"]


def test_random_gauge_reports_the_procrustes_number_it_is_compared_against():
    generator = torch.Generator().manual_seed(15)
    targets = torch.nn.functional.normalize(torch.randn(300, 12, generator=generator), dim=-1)
    student = torch.nn.functional.normalize(torch.randn(300, 12, generator=generator), dim=-1)

    _, reference = fit_gauge_alignment(targets, student)
    _, stats = fit_gauge_rotation(targets, student, mode="random", seed=16)

    assert stats["cos_procrustes"] == pytest.approx(reference["cos_after"], abs=1e-6)
    assert stats["cos_before"] == pytest.approx(reference["cos_before"], abs=1e-6)


def test_procrustes_mode_is_the_unrotated_default():
    generator = torch.Generator().manual_seed(17)
    targets = torch.nn.functional.normalize(torch.randn(300, 12, generator=generator), dim=-1)
    student = torch.nn.functional.normalize(torch.randn(300, 12, generator=generator), dim=-1)

    expected, _ = fit_gauge_alignment(targets, student)
    rotation, stats = fit_gauge_rotation(targets, student, mode="procrustes")

    assert torch.equal(rotation, expected)
    assert "cos_procrustes" not in stats


def test_unknown_gauge_rotation_is_rejected():
    with pytest.raises(ValueError, match="unknown gauge rotation"):
        fit_gauge_rotation(torch.randn(10, 4), torch.randn(10, 4), mode="haar")


def test_participation_ratio_detects_a_rank_one_cross_covariance():
    """PR is the diagnostic that predicts a null gauge ablation in advance: at PR ~ 1
    the rotation can only map one mean vector onto another."""
    generator = torch.Generator().manual_seed(18)
    direction_t = torch.nn.functional.normalize(torch.randn(16, generator=generator), dim=0)
    direction_z = torch.nn.functional.normalize(torch.randn(16, generator=generator), dim=0)
    # Both clouds sit almost on top of one direction: the cross-covariance is rank one.
    targets = torch.nn.functional.normalize(
        direction_t + 0.02 * torch.randn(600, 16, generator=generator), dim=-1
    )
    student = torch.nn.functional.normalize(
        direction_z + 0.02 * torch.randn(600, 16, generator=generator), dim=-1
    )

    _, degenerate = fit_gauge_alignment(targets, student)
    _, isotropic = fit_gauge_alignment(
        torch.nn.functional.normalize(torch.randn(600, 16, generator=generator), dim=-1),
        torch.nn.functional.normalize(torch.randn(600, 16, generator=generator), dim=-1),
    )

    assert degenerate["participation_ratio"] < 1.2
    assert degenerate["top_singular_share"] > 0.9
    assert isotropic["participation_ratio"] > 5.0


# --------------------------------------------------------------------------- #
# Wiring: the run has to apply the arm it was asked for, and record it
# --------------------------------------------------------------------------- #


def _stub_distiller(config, student_dim, student_init):
    """A distiller reduced to what ``_project_teacher_targets`` actually touches."""
    instance = object.__new__(KnowledgeDistiller)
    instance.config = config
    instance.model_student = SimpleNamespace(
        config=SimpleNamespace(hidden_size=student_dim)
    )
    instance._student_initial_embeddings = lambda texts: student_init[: len(texts)]
    return instance


def _targets_for(save_dir="", **overrides):
    student_dim, rows = 8, 64
    settings = {
        "gauge_align_samples": rows,
        "gauge_refit_every": 0,
        **overrides,
    }
    config = GeoODEConfig(save_dir=save_dir, **settings)
    generator = torch.Generator().manual_seed(19)
    teacher = _spectral_data(rows=rows, dim=32, seed=19)
    student_init = torch.nn.functional.normalize(
        torch.randn(rows, student_dim, generator=generator), dim=-1
    )
    distiller = _stub_distiller(config, student_dim, student_init)
    texts = [f"sentence {i}" for i in range(rows)]
    return distiller._project_teacher_targets(teacher, texts)


def test_each_arm_produces_a_different_set_of_targets():
    baseline = _targets_for()
    arms = {
        "uncentered": _targets_for(pca_center_fit=False),
        "subtract_mean": _targets_for(pca_subtract_mean=True),
        "random": _targets_for(projection_type="random"),
        "random_gaussian": _targets_for(projection_type="random_gaussian"),
        "no_gauge": _targets_for(gauge_align=False),
        "random_gauge": _targets_for(gauge_rotation="random"),
    }

    for name, targets in arms.items():
        assert targets.shape == baseline.shape, name
        assert not torch.allclose(targets, baseline, atol=1e-3), name


def test_the_gauge_arms_leave_the_target_geometry_alone():
    """Both rotations act inside the subspace, so only the no-gauge/other-subspace
    arms may move the Gram matrix. This is what makes the gauge ablation a clean
    one-factor comparison."""
    ungauged = _targets_for(gauge_align=False)
    procrustes = _targets_for(gauge_rotation="procrustes")
    random_gauge = _targets_for(gauge_rotation="random")

    reference = ungauged @ ungauged.T
    assert torch.allclose(procrustes @ procrustes.T, reference, atol=1e-4)
    assert torch.allclose(random_gauge @ random_gauge.T, reference, atol=1e-4)


def test_the_random_projection_seed_changes_the_targets():
    first = _targets_for(projection_type="random", projection_seed=0)
    second = _targets_for(projection_type="random", projection_seed=1)

    assert not torch.allclose(first, second, atol=1e-3)


def test_a_random_gauge_cannot_be_refit():
    with pytest.raises(ValueError, match="only defined for gauge_rotation"):
        _targets_for(gauge_rotation="random", gauge_refit_every=1)


def test_the_interpolated_gauge_walks_from_procrustes_to_the_random_one(tmp_path):
    """Q(theta) is the geodesic between the two gauges the paper compares, so the
    interpolation arm has to reduce to each of them at its endpoints -- otherwise the
    curve in Figure A1 would not connect the two points it is drawn between.

    At theta = 1 that identity holds only up to the component of O(d). The geodesic
    cannot leave the component it starts in, so when the Procrustes gauge is a
    reflection and this seed's Haar draw is a rotation no continuous path between
    them exists at all, and the walk ends at that draw with its last column negated
    -- an equally valid Haar matrix, but not the one --gauge_rotation random would
    use. The run records which of the two cases it was, so this asserts against the
    endpoint the arm actually claims rather than the one it cannot always reach.
    """
    procrustes = _targets_for(gauge_rotation="procrustes")
    random_gauge = _targets_for(gauge_rotation="random", gauge_random_seed=0)

    at_zero = _targets_for(gauge_rotation="interpolate", gauge_theta=0.0, gauge_random_seed=0)
    at_one = _targets_for(
        save_dir=str(tmp_path),
        gauge_rotation="interpolate", gauge_theta=1.0, gauge_random_seed=0,
    )
    halfway = _targets_for(gauge_rotation="interpolate", gauge_theta=0.5, gauge_random_seed=0)

    assert torch.allclose(at_zero, procrustes, atol=1e-4)

    saved = torch.load(tmp_path / "teacher_projection.pt", map_location="cpu")
    endpoint = random_gauge.clone()
    if saved["gauge_stats"]["endpoint_reflected"]:
        # T @ (Q with its last column negated) is T @ Q with its last column negated.
        endpoint[:, -1] *= -1
    assert torch.allclose(at_one, endpoint, atol=1e-4)

    assert not torch.allclose(halfway, procrustes, atol=1e-3)
    assert not torch.allclose(halfway, random_gauge, atol=1e-3)
    # Every point of the path is still an orthogonal map, so the Gram matrix of the
    # targets -- and with it the whole one-factor reading of the ablation -- is untouched.
    ungauged = _targets_for(gauge_align=False)
    assert torch.allclose(halfway @ halfway.T, ungauged @ ungauged.T, atol=1e-4)
    # ... including at the endpoint, whichever component it landed in.
    assert torch.allclose(at_one @ at_one.T, ungauged @ ungauged.T, atol=1e-4)


def test_the_geodesic_cannot_leave_the_component_of_o_d_it_starts_in():
    """The topology behind the endpoint caveat above, stated directly.

    det = +1 and det = -1 are the two connected components of O(d), and a continuous
    path within O(d) cannot cross between them. So a request to interpolate from a
    reflection to a rotation is not something an implementation can satisfy: it
    reports that it walked to the reflected image of the endpoint instead. When both
    lie in the same component nothing is touched and theta = 1 is exact.
    """
    def _draw(seed: int, positive: bool) -> torch.Tensor:
        """An independent Haar draw forced into the requested component of O(6)."""
        matrix = random_orthogonal(6, seed=seed).clone()
        if (float(torch.det(matrix.double())) > 0) != positive:
            matrix[:, 0] *= -1
        return matrix

    start = _draw(1, positive=True)
    same_component = _draw(2, positive=True)
    other_component = _draw(3, positive=False)

    reached, reflected = interpolate_rotation(start, same_component, 1.0)
    assert not reflected
    assert torch.allclose(reached, same_component, atol=1e-5)

    crossed, reflected = interpolate_rotation(start, other_component, 1.0)
    assert reflected
    expected = other_component.clone()
    expected[:, -1] *= -1
    assert torch.allclose(crossed, expected, atol=1e-5)
    # Whatever it walked to is still exactly orthogonal, and in the start's component.
    assert torch.allclose(crossed.T @ crossed, torch.eye(6), atol=1e-5)
    assert float(torch.det(crossed.double())) > 0


def test_the_rank_one_gauge_only_aligns_the_two_mean_directions():
    """The control for a student whose initial states are nearly one-dimensional: if
    the Householder map recovers the Procrustes gain, R was doing nothing more than
    matching the means."""
    rank_one = _targets_for(gauge_rotation="rank_one")
    ungauged = _targets_for(gauge_align=False)

    assert torch.allclose(rank_one @ rank_one.T, ungauged @ ungauged.T, atol=1e-4)
    assert not torch.allclose(rank_one, ungauged, atol=1e-3)


@pytest.mark.parametrize("mode", ["random", "interpolate", "rank_one"])
def test_only_the_fitted_gauge_can_be_refit(mode):
    with pytest.raises(ValueError, match="only defined for gauge_rotation"):
        _targets_for(gauge_rotation=mode, gauge_theta=0.5, gauge_refit_every=1)


def test_the_interpolated_gauge_needs_a_theta():
    with pytest.raises(ValueError, match="theta"):
        _targets_for(gauge_rotation="interpolate", gauge_theta=None)


def test_the_run_records_which_arm_it_used(tmp_path):
    _targets_for(
        save_dir=str(tmp_path),
        projection_type="random",
        projection_seed=2,
        pca_center_fit=False,
        gauge_rotation="random",
        gauge_random_seed=3,
    )

    saved = torch.load(tmp_path / "teacher_projection.pt", map_location="cpu")

    assert saved["projection_type"] == "random"
    assert saved["projection_seed"] == 2
    assert saved["pca_center_fit"] is False
    assert saved["gauge_rotation"] == "random"
    assert saved["gauge_random_seed"] == 3
    assert saved["gauge_matrix"].shape == (8, 8)
    assert "cos_procrustes" in saved["gauge_stats"]
    # Only the interpolate arm has an endpoint that can need reflecting.
    assert "endpoint_reflected" not in saved["gauge_stats"]
    assert saved["explained_energy"] < 1.0


def test_the_gauge_subset_is_selected_once_and_saved(tmp_path):
    _targets_for(
        save_dir=str(tmp_path),
        gauge_rotation="procrustes",
        gauge_align_samples=16,
        gauge_refit_every=1,
    )

    saved = torch.load(tmp_path / "teacher_projection.pt", map_location="cpu")
    indices = saved["gauge_fit_indices"]

    assert len(indices) == 16
    assert len(indices.unique()) == 16
    assert saved["gauge_refit_every"] == 1
    assert saved["gauge_subset_policy"] == "fixed_evenly_spaced"
    assert len(saved["gauge_history"]) == 1


def test_epoch_refits_reuse_the_initial_calibration_subset():
    rows, student_dim, fit_samples = 32, 8, 7
    config = GeoODEConfig(
        save_dir="",
        gauge_align_samples=fit_samples,
        gauge_refit_every=1,
    )
    teacher = _spectral_data(rows=rows, dim=24, seed=23)
    generator = torch.Generator().manual_seed(23)
    student_states = [
        torch.nn.functional.normalize(
            torch.randn(fit_samples, student_dim, generator=generator), dim=-1
        )
        for _ in range(3)
    ]
    calls = []
    distiller = _stub_distiller(config, student_dim, student_states[0])

    def encode(texts):
        calls.append(tuple(texts))
        return student_states[len(calls) - 1]

    distiller._student_initial_embeddings = encode
    texts = [f"sentence {i}" for i in range(rows)]
    distiller.teacher_cls_all = distiller._project_teacher_targets(teacher, texts)
    distiller.log_experiment_record = lambda record: None

    distiller._refit_gauge(0)
    distiller._refit_gauge(1)

    assert len(calls) == 3
    assert calls[0] == calls[1] == calls[2]
    assert len(calls[0]) == fit_samples
    assert len(distiller._gauge_state["history"]) == 3


# --------------------------------------------------------------------------- #
# The grid runner: it emits main.py commands, so it can rot silently
# --------------------------------------------------------------------------- #


def _plan(*argv):
    args = ablation.parse_args(list(argv))
    return args, ablation.build_plan(args)


def _config_from(command):
    """Parse a generated command back through main.py's own parser."""
    original = sys.argv
    # Drop the interpreter and the main.py path; keep the flags.
    sys.argv = ["main.py", *command[2:]]
    try:
        return get_config("geoode", parse_args())
    finally:
        sys.argv = original


@pytest.mark.parametrize("grid", ["requested", "full"])
def test_every_planned_cell_parses_into_the_arm_it_names(grid):
    """The runner writes main.py flags by hand, so this is what catches a flag that
    was renamed on one side only -- the failure mode that would otherwise show up as
    a grid of runs that all quietly used the default map."""
    args, plan = _plan("--grid", grid, "--draws", "2")
    expected_type = {"pca": "pca", "pca_full": "pca", "svd": "pca"}

    for cell in plan:
        config = _config_from(ablation.build_command(args, cell))
        subspace, gauge = cell["subspace"], cell["gauge"]

        assert config.projection_type == expected_type.get(subspace, subspace)
        assert config.pca_center_fit is (subspace != "svd")
        assert config.pca_subtract_mean is (subspace == "pca_full")
        assert config.gauge_align is (gauge != "none")
        if gauge != "none":
            assert config.gauge_rotation == gauge
        assert config.gauge_refit_every == (1 if gauge == "procrustes" else 0)
        if subspace in ablation.STOCHASTIC_SUBSPACES:
            assert config.projection_seed == cell["draw"]
        if gauge == "random":
            assert config.gauge_random_seed == cell["draw"]
        # The objective is held fixed across the grid: only the map varies.
        assert config.lambda_ctr == args.lambda_ctr


def test_cell_names_are_unique_so_runs_cannot_overwrite_each_other():
    _, plan = _plan("--grid", "full", "--draws", "3", "--seeds", "1", "2")
    names = [cell["name"] for cell in plan]

    assert len(names) == len(set(names))
    # 20 cells = 6 subspaces x 3 gauges, plus the two learned arms which have no
    # gauge column. 10 of those are deterministic ({pca, pca_full, svd, mrl_prefix}
    # x {none, procrustes} and the two learned ones); the other 10 are drawn three
    # times, and every cell is repeated at both training seeds.
    assert len(names) == (10 + 10 * 3) * 2


def test_draws_only_multiply_the_stochastic_arms():
    _, single = _plan("--grid", "full")
    _, tripled = _plan("--grid", "full", "--draws", "3")

    deterministic = [
        cell
        for cell in tripled
        if cell["subspace"] not in ablation.STOCHASTIC_SUBSPACES
        and cell["gauge"] != "random"
    ]
    # {pca, pca_full, svd, mrl_prefix} x {none, procrustes}, plus the two learned
    # arms: the learned map varies with the training seed, not with a draw of the map.
    assert len(deterministic) == 10
    assert len(tripled) == len(single) + 2 * (len(single) - len(deterministic))


def test_collect_reports_finished_and_missing_cells(tmp_path, capsys):
    args, plan = _plan("--grid", "requested", "--out", str(tmp_path))
    finished = plan[1]
    save_dir = tmp_path / finished["name"]
    save_dir.mkdir(parents=True)
    (save_dir / "metrics.jsonl").write_text(
        json.dumps({"train": {"epoch": 5}, "test": {"summary": {"avg_all": 0.1}}})
        + "\n"
        + json.dumps({"test": {"summary": {"avg_iod": 0.8, "avg_all": 0.795}}})
        + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "projection_type": "pca",
            "pca_center_fit": True,
            "pca_subtract_mean": False,
            "explained_energy": 0.918,
            "gauge_align": True,
            "gauge_rotation": "procrustes",
            "gauge_stats": {"cos_before": 0.1, "cos_after": 0.55, "participation_ratio": 1.05},
        },
        save_dir / "teacher_projection.pt",
    )

    rows = ablation.collect(args, plan)

    by_name = {row["name"]: row for row in rows}
    assert len(rows) == len(plan)
    # The per-epoch record must not be mistaken for the end-of-run one.
    assert by_name[finished["name"]]["avg_all"] == pytest.approx(0.795)
    assert by_name[finished["name"]]["participation_ratio"] == pytest.approx(1.05)
    assert by_name[finished["name"]]["status"] == "done"
    assert all(
        row["status"] == "missing" for row in rows if row["name"] != finished["name"]
    )
    assert (tmp_path / "target_map_ablation.csv").is_file()
    assert "79.50" in capsys.readouterr().out


def test_an_unfinished_cell_is_not_reported_as_done(tmp_path):
    args, plan = _plan("--grid", "requested", "--out", str(tmp_path))
    save_dir = tmp_path / plan[0]["name"]
    save_dir.mkdir(parents=True)
    # Only per-epoch records: the run died before the final test evaluation.
    (save_dir / "metrics.jsonl").write_text(
        json.dumps({"train": {"epoch": 1}, "test": {"summary": {"avg_all": 0.5}}}) + "\n",
        encoding="utf-8",
    )

    assert ablation.final_test_record(save_dir) is None
    assert ablation.collect(args, plan)[0]["status"] == "missing"


# --------------------------------------------------------------------------- #
# The audit notebook is the run vehicle, and it builds every arm's command by hand
# --------------------------------------------------------------------------- #

NOTEBOOK = REPO_ROOT / "notebooks" / "audit_qwen_minilm.ipynb"


def _notebook_cell(prefix: str) -> str:
    """One cell of the audit notebook, found by its header comment rather than index."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = ["".join(cell["source"]) for cell in notebook["cells"]]
    matches = [source for source in sources if source.startswith(prefix)]
    assert len(matches) == 1, f"expected exactly one cell starting {prefix!r}"
    return matches[0]


def _run_notebook_setup(tmp_path, **overrides):
    """Execute the config cell and the arm-registry cell only.

    Cells 2-4 clone the repo, check the GPU, dedup the corpus and encode the probe
    set, so what they define is stubbed instead. Nothing here trains: cell 5 only
    builds command lists.
    """
    namespace = {"__name__": "__nb__"}
    exec(_notebook_cell("# 1. "), namespace)
    namespace.update(overrides)
    datasets = namespace["DATASETS"]
    namespace.update(
        {
            "sys": sys,
            "os": os,
            "subprocess": subprocess,
            "PROJECT_DIR": REPO_ROOT,
            # Cell 3 picks these: the run's own directory, the output root the shared
            # teacher cache hangs off, and the (dedup) corpus of every dataset.
            "OUTPUT_BASE": tmp_path,
            "RUN_ROOT": tmp_path / namespace["RUN_NAME"],
            "IN_COLAB": False,
            "TRAIN_DATA_BY_DATASET": {key: REPO_ROOT / spec["path"] for key, spec in datasets.items()},
            "TRAIN_DATA": REPO_ROOT / datasets[namespace["DATASET"]]["path"],
        }
    )
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    exec(_notebook_cell("# 5. "), namespace)
    return namespace


def _config_for(command):
    """Parse a generated command back through main.py's own parser, any method."""
    original = sys.argv
    sys.argv = ["main.py", *command[2:]]
    try:
        args = parse_args()
        return get_config(args.method, args)
    finally:
        sys.argv = original


def test_the_notebook_builds_the_arms_it_names(tmp_path, capsys):
    """Every arm's flags are appended by hand, so this is what catches a flag that
    was renamed on one side only -- a whole plan that silently trained the default
    map would otherwise look like a clean null result."""
    namespace = _run_notebook_setup(tmp_path, DRAWS=2)
    capsys.readouterr()

    plan = namespace["ARM_PLAN"]
    commands = namespace["ARM_COMMANDS"]
    assert plan, "the plan is empty"
    for arm in plan:
        if arm["needs"] is not None:
            assert arm["name"] not in commands
            continue
        command = commands[arm["name"]]
        config = _config_for(command)
        assert config.distill_method == arm["method"]
        for key, value in (arm["settings"] or {}).items():
            assert getattr(config, key) == value, (arm["name"], key)
        assert config.seed == arm["seed"]
        # One save_dir and one seed: a second copy of either would leave argparse
        # silently taking the last one.
        assert command.count("--save_dir") == 1
        assert command.count("--seed") == 1
        assert command[command.index("--save_dir") + 1].endswith(arm["name"])
        # Per-epoch weights are what the probe dump reads.
        assert command[command.index("--save_every") + 1] == "1"
        assert "--weights_dir" in command


def test_the_notebook_covers_the_protocol_arms_and_multiplies_only_the_random_ones(tmp_path, capsys):
    namespace = _run_notebook_setup(
        tmp_path, DRAWS=3, SEEDS=[1, 2], GROUPS={"E1": True, "E2": True, "G": True, "A5": True}
    )
    capsys.readouterr()
    names = [arm["name"] for arm in namespace["ARM_PLAN"]]
    bases = {arm["base"] for arm in namespace["ARM_PLAN"]}

    assert len(names) == len(set(names))
    assert all(name.endswith(("__s1", "__s2")) for name in names)
    # One arm per interface family (E1), one row per design choice (E2), the gauge
    # controls (G) and the sanity variants (A5).
    for expected in ("pca__procrustes", "random__none__d2", "learned_t2s__lr1", "learned_t2s__lr5",
                     "learned_s2t__lr1", "learned_s2t__lr5", "procrustes_per_batch",
                     "ours__mse", "ours__no_ctr", "ours__gram_w1", "ours__gram_w10", "pca__none", "ours__refit",
                     "ours__no_teacher",
                     "pca__random__d2", "gauge_theta0.25", "gauge_theta0.5", "gauge_theta0.75", "gauge_rank_one",
                     "random_gaussian__none__d0", "pca_full__procrustes", "svd__procrustes"):
        assert expected in bases, expected
    assert sum(base.startswith("pca__random__d") for base in bases) == 3
    assert sum(base.startswith("random__none__d") for base in bases) == 3
    # Every arm of the protocol has code now: nothing is left blocked.
    assert not any(arm["needs"] for arm in namespace["ARM_PLAN"])
    # The groups that are off by default stay out of the plan.
    default = _run_notebook_setup(tmp_path / "default")
    capsys.readouterr()
    assert not any(arm["group"] == "A5" for arm in default["ARM_PLAN"])


def test_the_notebook_shares_the_teacher_cache_and_holds_the_matched_hp(tmp_path, capsys):
    """Every arm has the same teacher and corpus, so the teacher is encoded once for
    the whole plan, out of a directory that outlives the run; and the matched-HP
    protocol means one lr / batch / epoch count / lambda for every arm."""
    namespace = _run_notebook_setup(tmp_path)
    capsys.readouterr()
    hp = namespace["MATCHED_HP"]
    caches = set()
    for arm in namespace["ARM_PLAN"]:
        if arm["needs"] is not None:
            continue
        command = namespace["ARM_COMMANDS"][arm["name"]]
        config = _config_for(command)
        assert config.learning_rate == hp["learning_rate"]
        assert config.batch_size == hp["batch_size"]
        assert config.epochs == hp["epochs"]
        if config.distill_method == "geoode" and "--lambda_ctr" not in arm["flags"]:
            assert config.lambda_ctr == hp["lambda_ctr"]
        if config.distill_method in ("geoode", "talas", "rkd"):
            caches.add(Path(command[command.index("--cache_dir") + 1]))
    assert len(caches) == 1
    cache = caches.pop()
    assert namespace["RUN_ROOT"] not in cache.parents and cache != namespace["RUN_ROOT"]
    # The recipe ablation changes one thing per row and leaves the rest of the recipe alone.
    mse = _config_for(namespace["ARM_COMMANDS"]["ours__mse"])
    assert mse.endpoint_loss == "mse" and mse.lambda_ctr == hp["lambda_ctr"] and mse.gauge_align is True
    no_ctr = _config_for(namespace["ARM_COMMANDS"]["ours__no_ctr"])
    assert no_ctr.lambda_ctr == 0.0 and no_ctr.endpoint_loss == "cosine"
    gram = _config_for(namespace["ARM_COMMANDS"]["ours__gram_w10"])
    assert gram.lambda_gram == 10.0 and gram.gauge_align is True
    per_batch = _config_for(namespace["ARM_COMMANDS"]["procrustes_per_batch"])
    assert per_batch.endpoint_loss == "procrustes" and per_batch.gauge_align is False
    theta = _config_for(namespace["ARM_COMMANDS"]["gauge_theta0.5"])
    assert theta.gauge_rotation == "interpolate" and theta.gauge_theta == 0.5
    assert _config_for(namespace["ARM_COMMANDS"]["gauge_rank_one"]).gauge_rotation == "rank_one"


# --------------------------------------------------------------------------- #
# The learned-projector baselines: the map trained instead of frozen
# --------------------------------------------------------------------------- #


def _hidden(batch=6, tokens=5, dim=8, layers=4, seed=500):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(batch, tokens, dim, generator=generator) for _ in range(layers + 1)]


@pytest.mark.parametrize(
    ("direction", "weight_shape", "comparison_dim"),
    [("t2s", (8, 32), 8), ("s2t", (32, 8), 32)],
)
def test_the_learned_map_is_a_bare_linear_layer(direction, weight_shape, comparison_dim):
    projector = LearnedTargetProjector(teacher_dim=32, student_dim=8, direction=direction)

    assert projector.linear.weight.shape == weight_shape
    # No bias: a shift would move the targets off the sphere every downstream metric
    # is measured on, and none of the maps being compared has one.
    assert projector.linear.bias is None
    assert projector.comparison_dim == comparison_dim


def test_the_learned_map_brings_both_sides_into_one_space():
    projector = LearnedTargetProjector(teacher_dim=32, student_dim=8, direction="s2t")
    states = [torch.nn.functional.normalize(torch.randn(6, 8), dim=-1) for _ in range(4)]
    teacher = torch.nn.functional.normalize(torch.randn(6, 32), dim=-1)

    aligned, target = projector.align(states, teacher)

    assert len(aligned) == len(states)
    assert all(state.shape == (6, 32) for state in aligned)
    # Every layer lands on the sphere, so the depth diagnostics mean the same thing
    # for a learned map as for a frozen one.
    for state in aligned:
        assert torch.allclose(state.norm(dim=-1), torch.ones(6), atol=1e-5)
    assert torch.equal(target, teacher)


def test_the_t2s_map_leaves_the_student_untouched():
    projector = LearnedTargetProjector(teacher_dim=32, student_dim=8, direction="t2s")
    states = [torch.nn.functional.normalize(torch.randn(6, 8), dim=-1) for _ in range(3)]
    teacher = torch.nn.functional.normalize(torch.randn(6, 32), dim=-1)

    aligned, target = projector.align(states, teacher)

    assert all(torch.equal(a, b) for a, b in zip(aligned, states))
    assert target.shape == (6, 8)
    assert torch.allclose(target.norm(dim=-1), torch.ones(6), atol=1e-5)


def test_an_unknown_direction_is_rejected():
    with pytest.raises(ValueError, match="unknown direction"):
        LearnedTargetProjector(teacher_dim=32, student_dim=8, direction="both")


@pytest.mark.parametrize("direction", ["t2s", "s2t"])
def test_the_objective_trains_the_learned_map(direction):
    """The whole point of the baseline: the map adapts to lower the loss."""
    projector = LearnedTargetProjector(teacher_dim=32, student_dim=8, direction=direction)
    criterion = GeoODEKD(target_projector=projector)
    teacher = torch.nn.functional.normalize(torch.randn(6, 32), dim=-1)

    loss, metrics = criterion(
        hidden_states=_hidden(), teacher=teacher, second_view=torch.randn(6, 8)
    )
    loss.backward()

    gradient = projector.linear.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert float(gradient.norm()) > 0
    assert metrics["loss_end"] > 0


@pytest.mark.parametrize("direction", ["t2s", "s2t"])
def test_the_learned_map_does_not_touch_the_contrastive_term(direction):
    """The regulariser is a statement about the student's own space, so it has to
    read the same number under every arm -- otherwise the arms differ in two things
    at once and the comparison is not a controlled one."""
    hidden = _hidden()
    second_view = torch.randn(6, 8)
    frozen_teacher = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    learned_teacher = torch.nn.functional.normalize(torch.randn(6, 32), dim=-1)

    _, frozen = GeoODEKD()(
        hidden_states=hidden, teacher=frozen_teacher, second_view=second_view
    )
    _, learned = GeoODEKD(
        target_projector=LearnedTargetProjector(32, 8, direction)
    )(hidden_states=hidden, teacher=learned_teacher, second_view=second_view)

    assert learned["loss_ctr"] == pytest.approx(frozen["loss_ctr"], abs=1e-6)


def test_the_criterions_parameters_are_exactly_the_learned_map():
    """The distiller adds ``criterion.parameters()`` to the optimizer as one group.
    If the criterion ever grows a parameter of its own, that group would silently
    start training something else too."""
    projector = LearnedTargetProjector(teacher_dim=32, student_dim=8, direction="t2s")

    assert list(GeoODEKD().parameters()) == []
    criterion_params = list(GeoODEKD(target_projector=projector).parameters())
    assert len(criterion_params) == 1
    assert criterion_params[0] is projector.linear.weight


@pytest.mark.parametrize("arm", ["learned_t2s", "learned_s2t"])
def test_a_learned_arm_fits_no_map_and_leaves_the_targets_in_teacher_space(arm, tmp_path):
    targets = _targets_for(save_dir=str(tmp_path), projection_type=arm)

    # 32-dimensional: the cache is handed to the criterion unmapped, because the map
    # is a parameter and cannot be applied once up front.
    assert targets.shape == (64, 32)
    assert torch.allclose(targets.norm(dim=-1), torch.ones(64), atol=1e-5)
    saved = torch.load(tmp_path / "teacher_projection.pt", map_location="cpu")
    assert saved["projection_type"] == arm
    assert saved["projection"] is None
    assert saved["learned_direction"] == arm.removeprefix("learned_")
    # gauge_align defaults to True, and a learned map has no frozen basis to orient.
    assert saved["gauge_align"] is False
    assert saved["gauge_matrix"] is None


def test_the_learned_arm_records_the_dimensions_the_projector_needs(capsys):
    """The criterion is constructed after the targets, so the dimensions have to
    survive the trip -- the parameters must exist before the optimizer group is
    added."""
    _targets_for(projection_type="learned_s2t")
    output = capsys.readouterr().out

    assert "Learned target map learned_s2t" in output
    # gauge_align is on by default; the run has to say it does not apply.
    assert "Gauge alignment does not apply" in output


def test_the_learned_arms_have_no_gauge_column():
    _, plan = _plan("--subspace", "learned_t2s", "learned_s2t", "--gauge", "none", "procrustes", "random")

    assert [cell["name"] for cell in plan] == ["learned_t2s__none", "learned_s2t__none"]


def test_the_requested_grid_is_ours_against_the_four_controls():
    _, plan = _plan("--grid", "requested")

    assert [cell["name"] for cell in plan] == [
        "pca__procrustes",   # ours
        "pca__none",         # PCA only
        "pca__random",       # PCA + random orthogonal rotation
        "random__none",      # random projection
        "learned_t2s__none",
        "learned_s2t__none",
    ]


def test_the_learned_map_can_join_the_optimizer_after_the_scheduler_exists():
    """The distiller builds the optimizer in setup_training and the criterion after
    it, so a learned arm adds its parameter group late. A group added after the
    scheduler was constructed has no matching base_lr and ``scheduler.step()`` then
    fails on the length mismatch -- which is why the scheduler is rebuilt. This is
    that sequence, in order."""
    student = torch.nn.Linear(8, 8)
    config = GeoODEConfig(epochs=2, learning_rate=2e-5, min_lr=2e-6)
    stub = object.__new__(KnowledgeDistiller)
    stub.config = config
    stub.train_loader = range(4)  # _build_scheduler only needs its length

    stub.optimizer = torch.optim.AdamW(
        [{"params": list(student.parameters()), "lr": config.learning_rate}],
        lr=config.learning_rate,
    )
    stub.scheduler = stub._build_scheduler()

    projector = LearnedTargetProjector(teacher_dim=32, student_dim=8, direction="t2s")
    criterion = GeoODEKD(target_projector=projector)
    scale = config.learned_projector_lr_scale
    stub.optimizer.add_param_group(
        {"params": criterion.parameters(), "lr": config.learning_rate * scale}
    )
    stub.scheduler = stub._build_scheduler()

    assert len(stub.optimizer.param_groups) == 2
    for _ in range(len(stub.train_loader) * config.epochs):
        stub.optimizer.step()
        stub.scheduler.step()
    # Both groups are still being scheduled, and the map got the rate it was given.
    assert len(stub.scheduler.base_lrs) == 2
    assert stub.scheduler.base_lrs[1] == pytest.approx(config.learning_rate * scale)


def test_fitting_a_learned_map_is_refused_with_the_reason():
    """A learned arm never reaches the fitter, but saying so beats "unknown type"."""
    with pytest.raises(ValueError, match="trained map, not a fitted one"):
        fit_teacher_projection(
            torch.randn(16, 32), out_dim=8, projection_type="learned_t2s"
        )
