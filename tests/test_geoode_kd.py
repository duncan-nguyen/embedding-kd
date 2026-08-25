"""Tests for GeoODE-KD.

The theory in Section 3.9 is what makes the objective meaningful, so the geometry is
tested against it directly: the flow has to stay on the sphere (Prop. 1), the
teacher-conditioned energy has to fall along it (Prop. 2), and the instance-only field
has to raise teacher cosine (Cor. 1). The loss tests then pin the pieces those
propositions are wired into.
"""

import itertools

import pytest
import torch

from src.criterions.geoode_kd import GeoODEKD
from src.teacher_projection import fit_pca_projection, project_teacher_embeddings


def _sphere(batch: int, dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(
        torch.randn(batch, dim, generator=generator, dtype=torch.float64), dim=-1
    )


def _criterion(**kwargs) -> GeoODEKD:
    defaults = {"guidance_schedule": "constant"}
    defaults.update(kwargs)
    return GeoODEKD(**defaults)


def test_criterion_has_no_trainable_parameters():
    # The paper's claim that training adds no module and inference is unchanged rests
    # on this: the vector field is analytic in the cached targets.
    assert list(_criterion().parameters()) == []


def test_tangent_projection_is_orthogonal_to_the_state():
    Z = _sphere(6, 8, seed=0)
    U = torch.randn(6, 8, dtype=torch.float64)

    tangent = GeoODEKD.tangent_project(Z, U)

    assert torch.allclose(
        (Z * tangent).sum(dim=-1), torch.zeros(6, dtype=torch.float64), atol=1e-12
    )


def test_retraction_preserves_unit_norm():
    criterion = _criterion()
    Z = _sphere(6, 8, seed=1)
    V = 0.3 * GeoODEKD.tangent_project(Z, torch.randn(6, 8, dtype=torch.float64))

    retracted = criterion.retract(Z, V)

    assert torch.allclose(
        retracted.norm(dim=-1), torch.ones(6, dtype=torch.float64), atol=1e-12
    )


def test_euler_steps_stay_on_the_sphere():
    """Proposition 1, in the discretised form the training loss actually uses."""
    criterion = _criterion(alpha=1.0, beta=1.0)
    Z = _sphere(8, 16, seed=2)
    T = _sphere(8, 16, seed=3)

    for _ in range(50):
        Z = criterion.euler_step(Z, T, t=1.0, dt=0.05)
        assert torch.allclose(
            Z.norm(dim=-1), torch.ones(8, dtype=torch.float64), atol=1e-10
        )


@pytest.mark.parametrize("beta", [0.0, 1.0, 5.0])
def test_energy_decreases_along_the_flow(beta):
    """Proposition 2: dE/dt <= 0 under the teacher-conditioned dynamics."""
    criterion = _criterion(alpha=1.0, beta=beta)
    Z = _sphere(8, 16, seed=4)
    T = _sphere(8, 16, seed=5)

    energies = []
    for _ in range(200):
        energies.append(float(criterion.energy(Z, T)[0]))
        Z = criterion.euler_step(Z, T, t=1.0, dt=1e-3)
    energies.append(float(criterion.energy(Z, T)[0]))

    diffs = torch.tensor(energies[1:]) - torch.tensor(energies[:-1])
    assert bool((diffs <= 1e-12).all()), f"energy rose along the flow: max diff {diffs.max()}"
    assert energies[-1] < energies[0]


def test_pointwise_flow_increases_teacher_cosine():
    """Corollary 1: with beta = 0 the cosine to the teacher rises monotonically."""
    criterion = _criterion(alpha=1.0, beta=0.0)
    Z = _sphere(8, 16, seed=6)
    T = _sphere(8, 16, seed=7)

    cosines = []
    for _ in range(300):
        cosines.append((Z * T).sum(dim=-1).clone())
        Z = criterion.euler_step(Z, T, t=1.0, dt=1e-2)
    cosines.append((Z * T).sum(dim=-1))

    for earlier, later in itertools.pairwise(cosines):
        assert bool((later >= earlier - 1e-12).all())
    assert bool((cosines[-1] > cosines[0]).all())


def test_relational_energy_gradient_matches_autograd():
    """Eq. (24) is hand-derived; check it against the autograd gradient of Eq. (19)."""
    criterion = _criterion(alpha=0.0, beta=1.0, guidance_schedule="constant")
    Z = _sphere(6, 10, seed=8).requires_grad_(True)
    T = _sphere(6, 10, seed=9)

    energy = criterion.energy(Z, T)[2]
    (analytic_free,) = torch.autograd.grad(energy, Z)

    batch = Z.shape[0]
    closed_form = (4.0 / (batch * batch)) * (
        (Z.detach() @ Z.detach().T - T @ T.T) @ Z.detach()
    )
    assert torch.allclose(analytic_free, closed_form, atol=1e-10)


def test_dynamics_loss_vanishes_when_layers_follow_the_flow():
    criterion = _criterion(alpha=1.0, beta=1.0, guidance_schedule="linear")
    T = _sphere(8, 16, seed=10)
    num_layers = 6
    dt = 1.0 / num_layers

    states = [_sphere(8, 16, seed=11)]
    for index in range(num_layers - 1):
        t = (index + 1) / num_layers
        states.append(criterion.euler_step(states[-1], T, t, dt))

    loss = criterion.dynamics_loss(states, T)

    assert float(loss) == pytest.approx(0.0, abs=1e-12)


def test_dynamics_loss_is_positive_for_an_unrelated_trajectory():
    criterion = _criterion(alpha=1.0, beta=1.0)
    T = _sphere(8, 16, seed=12)
    states = [_sphere(8, 16, seed=13 + index) for index in range(5)]

    loss = criterion.dynamics_loss(states, T)

    assert float(loss) > 0.1


def test_stop_gradient_target_only_trains_the_next_layer():
    criterion = _criterion(alpha=1.0, beta=1.0)
    T = _sphere(4, 8, seed=20)
    states = [_sphere(4, 8, seed=21 + index).requires_grad_(True) for index in range(3)]

    loss = criterion.dynamics_loss(states, T)
    loss.backward()

    # Z^(1) is only ever the *source* of a prediction, and sg[.] cuts that path.
    assert states[0].grad is None or torch.allclose(
        states[0].grad, torch.zeros_like(states[0])
    )
    assert states[1].grad is not None and states[1].grad.abs().sum() > 0


def test_disabling_stop_gradient_propagates_into_the_source_layer():
    criterion = _criterion(alpha=1.0, beta=1.0, stop_grad_target=False)
    T = _sphere(4, 8, seed=30)
    states = [_sphere(4, 8, seed=31 + index).requires_grad_(True) for index in range(3)]

    loss = criterion.dynamics_loss(states, T)
    loss.backward()

    assert states[0].grad is not None and states[0].grad.abs().sum() > 0


def test_forward_combines_the_three_weighted_terms():
    criterion = _criterion(
        alpha=1.0, beta=1.0, lambda_end=0.7, lambda_dyn=2.0, lambda_ctr=0.3
    )
    batch, dim, tokens, layers = 4, 8, 5, 4
    generator = torch.Generator().manual_seed(40)
    hidden_states = [
        torch.randn(batch, tokens, dim, generator=generator) for _ in range(layers + 1)
    ]
    teacher = torch.nn.functional.normalize(
        torch.randn(batch, dim, generator=generator), dim=-1
    )
    second_view = torch.randn(batch, dim, generator=generator)

    total, metrics = criterion(
        hidden_states=hidden_states, teacher=teacher, second_view=second_view
    )

    expected = (
        0.7 * metrics["loss_end"] + 2.0 * metrics["loss_dyn"] + 0.3 * metrics["loss_ctr"]
    )
    assert float(total) == pytest.approx(expected, rel=1e-5)
    assert metrics["loss_ctr"] > 0.0
    assert set(metrics) >= {
        "loss_total",
        "loss_end",
        "loss_dyn",
        "loss_ctr",
        "cos_first",
        "cos_final",
        "gram_gap_first",
        "gram_gap_final",
    }


def test_contrastive_term_is_skipped_without_a_second_view():
    criterion = _criterion(lambda_ctr=0.5)
    hidden_states = [torch.randn(4, 5, 8) for _ in range(4)]
    teacher = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)

    _, metrics = criterion(hidden_states=hidden_states, teacher=teacher)

    assert metrics["loss_ctr"] == 0.0


def test_embedding_layer_is_excluded_by_default():
    hidden_states = [torch.randn(4, 5, 8) for _ in range(7)]

    assert len(_criterion().layer_states(hidden_states, None)) == 6
    assert (
        len(_criterion(include_embedding_layer=True).layer_states(hidden_states, None))
        == 7
    )


def test_layer_states_are_normalised_and_respect_the_mask():
    criterion = _criterion(pooling="mean")
    hidden_states = [torch.randn(3, 4, 8) for _ in range(3)]
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]])

    states = criterion.layer_states(hidden_states, mask)

    assert all(torch.allclose(z.norm(dim=-1), torch.ones(3), atol=1e-5) for z in states)
    padded = hidden_states[-1][2, 0, :]
    assert torch.allclose(
        states[-1][2], torch.nn.functional.normalize(padded, dim=-1), atol=1e-5
    )


def test_teacher_dimension_mismatch_is_rejected():
    criterion = _criterion()
    hidden_states = [torch.randn(4, 5, 8) for _ in range(3)]

    with pytest.raises(ValueError, match="projected into the student dimension"):
        criterion(hidden_states=hidden_states, teacher=torch.randn(4, 16))


@pytest.mark.parametrize(
    ("schedule", "kwargs", "expected"),
    [
        ("linear", {}, 0.25),
        ("constant", {}, 1.0),
        ("power", {"guidance_power": 2.0}, 0.0625),
    ],
)
def test_guidance_schedules(schedule, kwargs, expected):
    criterion = GeoODEKD(guidance_schedule=schedule, **kwargs)

    assert criterion.guidance(0.25) == pytest.approx(expected)


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError, match="guidance_schedule"):
        GeoODEKD(guidance_schedule="cosine")


def test_pca_projection_maps_onto_the_student_sphere():
    generator = torch.Generator().manual_seed(50)
    embeddings = torch.randn(64, 32, generator=generator)

    projection, mean = fit_pca_projection(embeddings, out_dim=8)
    targets = project_teacher_embeddings(embeddings, projection, mean=mean)

    assert projection.shape == (32, 8)
    assert mean.shape == (32,)
    # Orthonormal columns: the map is a rotation onto the leading subspace.
    assert torch.allclose(projection.T @ projection, torch.eye(8), atol=1e-5)
    assert targets.shape == (64, 8)
    assert torch.allclose(targets.norm(dim=-1), torch.ones(64), atol=1e-5)


def test_pca_projection_is_the_identity_when_dimensions_match():
    embeddings = torch.randn(16, 8)

    projection, _ = fit_pca_projection(embeddings, out_dim=8)

    assert torch.allclose(projection, torch.eye(8))


def test_pca_projection_keeps_the_dominant_directions():
    generator = torch.Generator().manual_seed(51)
    latent = torch.randn(200, 2, generator=generator)
    basis = torch.zeros(2, 6)
    basis[0, 0] = 10.0
    basis[1, 1] = 5.0
    embeddings = latent @ basis + 1e-3 * torch.randn(200, 6, generator=generator)

    projection, _ = fit_pca_projection(embeddings, out_dim=2)

    # The two retained directions have to span the plane the data actually lives in.
    plane = torch.eye(6)[:, :2]
    residual = projection - plane @ (plane.T @ projection)
    assert float(residual.abs().max()) < 1e-2


def test_pca_projection_rejects_an_undersized_corpus():
    with pytest.raises(ValueError, match="cannot fit"):
        fit_pca_projection(torch.randn(4, 32), out_dim=8)


def _report(criterion, states, teacher):
    return criterion.depth_report(states, teacher)


def test_depth_report_measures_a_flow_following_trajectory():
    """On a trajectory generated by the flow itself, every claim should hold."""
    criterion = _criterion(alpha=1.0, beta=1.0, guidance_schedule="linear")
    T = _sphere(8, 16, seed=60)
    num_layers = 8
    dt = 1.0 / num_layers

    states = [_sphere(8, 16, seed=61)]
    for index in range(num_layers - 1):
        states.append(criterion.euler_step(states[-1], T, (index + 1) / num_layers, dt))

    report = _report(criterion, states, T)

    assert report["num_layers"] == num_layers
    assert len(report["cos_teacher"]) == num_layers
    assert len(report["dyn_residual"]) == num_layers - 1
    # The trajectory *is* the discretised flow, so it follows the field exactly and
    # cannot raise the energy.
    assert report["mean_dyn_residual"] == pytest.approx(0.0, abs=1e-12)
    # Not exactly 1: the retraction bends the realized displacement away from the
    # tangent field by O(step^2), which is precisely the discretisation error the
    # alignment curve is meant to expose.
    assert report["mean_alignment"] == pytest.approx(1.0, abs=1e-4)
    assert report["energy_violations"] == 0
    assert report["cos_gain"] > 0


def test_depth_report_flags_a_trajectory_that_ignores_the_field():
    criterion = _criterion(alpha=1.0, beta=1.0)
    T = _sphere(8, 16, seed=70)
    states = [_sphere(8, 16, seed=71 + index) for index in range(6)]

    report = _report(criterion, states, T)

    assert report["mean_dyn_residual"] > 0.1
    assert abs(report["mean_alignment"]) < 0.5
    assert report["energy_violations"] > 0


def test_depth_report_separates_step_size_from_direction():
    """A student that moves the right way but far too far still scores a high
    alignment; the two norms are what expose it."""
    criterion = _criterion(alpha=1.0, beta=0.0)
    T = _sphere(8, 16, seed=80)
    num_layers = 4
    dt = 1.0 / num_layers

    states = [_sphere(8, 16, seed=81)]
    for index in range(num_layers - 1):
        field = criterion.vector_field(states[-1], T, (index + 1) / num_layers)
        states.append(criterion.retract(states[-1], 20.0 * dt * field))

    report = _report(criterion, states, T)

    assert report["mean_alignment"] > 0.9
    assert report["mean_step_norm"] > 5 * report["mean_field_norm"]


def test_depth_report_anisotropy_matches_a_collapsed_batch():
    criterion = _criterion()
    collapsed = torch.ones(6, 8, dtype=torch.float64)
    collapsed = torch.nn.functional.normalize(collapsed, dim=-1)
    teacher = _sphere(6, 8, seed=90)

    report = _report(criterion, [collapsed, collapsed], teacher)

    assert report["student_anisotropy"] == pytest.approx(1.0, abs=1e-9)
    assert report["teacher_anisotropy"] < 0.5


def test_depth_report_counts_monotonicity_violations_per_curve():
    criterion = _criterion(alpha=1.0, beta=0.0)
    T = _sphere(6, 8, seed=100)
    # A trajectory that walks away from the teacher: cosine falls at every depth.
    states = [T, criterion.retract(T, 0.5 * _sphere(6, 8, seed=101)), -T]

    report = _report(criterion, states, T)

    assert report["cos_violations"] == 2
    assert report["cos_gain"] < 0
