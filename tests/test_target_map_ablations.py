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
from scripts import run_target_map_ablation as ablation

REPO_ROOT = Path(__file__).resolve().parent.parent
from src.teacher_projection import (
    fit_gauge_alignment,
    fit_gauge_rotation,
    fit_pca_projection,
    fit_random_projection,
    fit_teacher_projection,
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

    for kind in ("pca", "random", "random_gaussian"):
        projection, mean = fit_teacher_projection(
            embeddings, out_dim=8, projection_type=kind, seed=10
        )
        assert projection.shape == (32, 8)
        assert mean.shape == (32,)

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
    config = GeoODEConfig(
        save_dir=save_dir, gauge_align_samples=rows, **overrides
    )
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
    assert saved["explained_energy"] < 1.0


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
        if subspace in ablation.STOCHASTIC_SUBSPACES:
            assert config.projection_seed == cell["draw"]
        if gauge == "random":
            assert config.gauge_random_seed == cell["draw"]
        # The objective is held fixed across the grid: only the map varies.
        assert config.lambda_vel == args.lambda_vel
        assert config.lambda_ctr == args.lambda_ctr


def test_cell_names_are_unique_so_runs_cannot_overwrite_each_other():
    _, plan = _plan("--grid", "full", "--draws", "3", "--seeds", "1", "2")
    names = [cell["name"] for cell in plan]

    assert len(names) == len(set(names))
    # 15 cells = 6 deterministic + 9 stochastic; the stochastic ones are drawn three
    # times, and every cell is repeated at both training seeds.
    assert len(names) == (6 + 9 * 3) * 2


def test_draws_only_multiply_the_stochastic_arms():
    _, single = _plan("--grid", "full")
    _, tripled = _plan("--grid", "full", "--draws", "3")

    deterministic = [
        cell
        for cell in tripled
        if cell["subspace"] not in ablation.STOCHASTIC_SUBSPACES
        and cell["gauge"] != "random"
    ]
    assert len(deterministic) == 6  # {pca, pca_full, svd} x {none, procrustes}
    assert len(tripled) == len(single) + 2 * (len(single) - len(deterministic))


def test_the_requested_grid_contains_the_control_it_exists_for():
    _, plan = _plan("--grid", "requested")
    cells = {(cell["subspace"], cell["gauge"]) for cell in plan}

    # PCA with and without Procrustes, and against a random rotation of equal cost.
    assert {("pca", "none"), ("pca", "procrustes"), ("pca", "random")} <= cells
    # Every subspace arm with and without Procrustes.
    for subspace in ("pca", "svd", "random"):
        assert {(subspace, "none"), (subspace, "procrustes")} <= cells


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
# The notebook is the run vehicle, and its ablation cells build commands by hand
# --------------------------------------------------------------------------- #

NOTEBOOK = Path(__file__).resolve().parent.parent / "test_mdd.ipynb"


def _notebook_cell(prefix: str) -> str:
    """One cell of test_mdd.ipynb, found by its header comment rather than index."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = ["".join(cell["source"]) for cell in notebook["cells"]]
    matches = [source for source in sources if source.startswith(prefix)]
    assert len(matches) == 1, f"expected exactly one cell starting {prefix!r}"
    return matches[0]


def _run_notebook_setup(tmp_path, **overrides):
    """Execute the config / command-building / ablation-plan cells only.

    Cells 2 and 3 clone the repo and check the GPU, so what they define is stubbed
    instead. Nothing here trains: cell 4 and cell 9 only build command lists.
    """
    namespace = {"__name__": "__nb__"}
    exec(_notebook_cell("# 1. "), namespace)
    namespace.update(overrides)
    namespace.update(
        {
            "sys": sys,
            "os": os,
            "subprocess": subprocess,
            "PROJECT_DIR": REPO_ROOT,
            "TRAIN_DATA": REPO_ROOT / namespace["TRAIN_DATA_REL"],
            "RUN_ROOT": tmp_path / namespace["RUN_NAME"],
            "IN_COLAB": False,
        }
    )
    exec(_notebook_cell("# 4. "), namespace)
    exec(_notebook_cell("# 9. "), namespace)
    return namespace


def test_the_notebook_builds_the_same_arms_as_the_script(tmp_path, capsys):
    """The notebook appends arm flags to cell 4's command by hand, so this is what
    catches the two drifting apart -- a whole grid that silently trained the
    default map would otherwise look like a clean null result."""
    namespace = _run_notebook_setup(tmp_path, ABLATION_GRID="full", ABLATION_DRAWS=2)
    capsys.readouterr()
    expected_type = {"pca": "pca", "pca_full": "pca", "svd": "pca"}

    for cell in namespace["ABLATION_PLAN"]:
        command = namespace["ABLATION_COMMANDS"][cell["name"]]
        config = _config_from(command)

        assert config.projection_type == expected_type.get(cell["subspace"], cell["subspace"])
        assert config.pca_center_fit is (cell["subspace"] != "svd")
        assert config.pca_subtract_mean is (cell["subspace"] == "pca_full")
        assert config.gauge_align is (cell["gauge"] != "none")
        if cell["gauge"] != "none":
            assert config.gauge_rotation == cell["gauge"]
        assert config.seed == cell["seed"]
        # One save_dir and one seed: appending a second copy of either would leave
        # argparse silently taking the last one.
        assert command.count("--save_dir") == 1
        assert command.count("--seed") == 1
        assert command[command.index("--save_dir") + 1].endswith(cell["name"])


def test_the_notebook_grid_shares_the_teacher_cache_of_the_method_runs(tmp_path, capsys):
    """Every cell has the same teacher and corpus, so the teacher must be encoded
    once for the whole grid and the eight method runs together."""
    namespace = _run_notebook_setup(tmp_path)
    capsys.readouterr()
    method_command = namespace["COMMANDS"]["geoode"]
    cache = method_command[method_command.index("--cache_path") + 1]

    for command in namespace["ABLATION_COMMANDS"].values():
        assert command[command.index("--cache_path") + 1] == cache


def test_the_notebook_holds_the_objective_fixed_across_the_grid(tmp_path, capsys):
    namespace = _run_notebook_setup(tmp_path)
    capsys.readouterr()

    for command in namespace["ABLATION_COMMANDS"].values():
        config = _config_from(command)
        assert config.lambda_vel == namespace["ABLATION_LAMBDA_VEL"]
        assert config.lambda_ctr == namespace["ABLATION_LAMBDA_CTR"]
