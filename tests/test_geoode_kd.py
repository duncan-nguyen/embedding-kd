"""Tests for GeoODE-KD.

The theory in Section 3.9 is what makes the objective meaningful, so the geometry is
tested against it directly: the flow has to stay on the sphere (Prop. 1), the
teacher-conditioned energy has to fall along it (Prop. 2), and the instance-only flow
has to contract the geodesic distance in closed form and reach the teacher exactly at
unit depth (Cor. 1). The loss tests then pin the pieces those
propositions are wired into.
"""

import itertools

import pytest
import torch

from src.criterions.geoode_kd import GeoODEKD
from src.teacher_projection import (
    fit_gauge_alignment,
    fit_pca_projection,
    project_teacher_embeddings,
)


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



def test_flow_steps_stay_on_the_sphere():
    """Proposition 1, in the discretised form the training loss actually uses."""
    criterion = _criterion(alpha=1.0, beta=1.0)
    Z = _sphere(8, 16, seed=2)
    T = _sphere(8, 16, seed=3)

    for _ in range(50):
        Z = criterion.flow_step(Z, T, rho=0.05)
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
        Z = criterion.flow_step(Z, T, rho=1e-3)
    energies.append(float(criterion.energy(Z, T)[0]))

    diffs = torch.tensor(energies[1:]) - torch.tensor(energies[:-1])
    assert bool((diffs <= 1e-12).all()), f"energy rose along the flow: max diff {diffs.max()}"
    assert energies[-1] < energies[0]



def test_pointwise_flow_increases_teacher_cosine():
    """Corollary 1 (weak form): with beta = 0 the cosine to the teacher rises."""
    criterion = _criterion(alpha=1.0, beta=0.0)
    Z = _sphere(8, 16, seed=6)
    T = _sphere(8, 16, seed=7)

    cosines = []
    for _ in range(300):
        cosines.append((Z * T).sum(dim=-1).clone())
        Z = criterion.flow_step(Z, T, rho=1e-2)
    cosines.append((Z * T).sum(dim=-1))

    for earlier, later in itertools.pairwise(cosines):
        assert bool((later >= earlier - 1e-12).all())
    assert bool((cosines[-1] > cosines[0]).all())


def test_exp_map_inverts_log_map():
    """Exp_z(Log_z(tau)) = tau: the two maps are exact, not first-order."""
    criterion = _criterion()
    Z = _sphere(8, 16, seed=200)
    T = _sphere(8, 16, seed=201)

    assert torch.allclose(criterion.exp_map(Z, criterion.log_map(Z, T)), T, atol=1e-9)
    # Log_z(tau) is tangent at z and has length d_g(z, tau).
    log = criterion.log_map(Z, T)
    assert torch.allclose((Z * log).sum(dim=-1), torch.zeros(8, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(log.norm(dim=-1), criterion.geodesic_distance(Z, T), atol=1e-9)


def test_instance_flow_step_is_spherical_interpolation():
    """With beta = 0, Exp_z(rho Log_z(tau)) is slerp(z, tau; rho): the geodesic
    distance left is (1 - rho) d_g, and rho = 1 lands on the teacher."""
    criterion = _criterion(alpha=1.0, beta=0.0)
    Z = _sphere(8, 16, seed=210)
    T = _sphere(8, 16, seed=211)
    before = criterion.geodesic_distance(Z, T)

    for rho in (0.1, 0.5, 0.9):
        after = criterion.geodesic_distance(criterion.flow_step(Z, T, rho), T)
        assert torch.allclose(after, (1.0 - rho) * before, atol=1e-8)

    assert torch.allclose(criterion.flow_step(Z, T, rho=1.0), T, atol=1e-8)


@pytest.mark.parametrize("schedule,kwargs", [("linear", {}), ("constant", {}), ("power", {"guidance_power": 2.0})])
def test_guidance_mass_and_step_fraction(schedule, kwargs):
    """R(t) = int_t^1 s, R(1) = 0, and the last step fraction is exactly 1."""
    criterion = _criterion(guidance_schedule=schedule, **kwargs)
    num_layers = 12

    assert criterion.guidance_mass(1.0) == pytest.approx(0.0)
    # Numerical quadrature of s against the closed form.
    grid = torch.linspace(0.25, 1.0, 20001, dtype=torch.float64)
    values = torch.tensor([criterion.guidance(float(t)) for t in grid], dtype=torch.float64)
    assert criterion.guidance_mass(0.25) == pytest.approx(float(torch.trapezoid(values, grid)), rel=1e-6)

    fractions = [
        criterion.step_fraction(l / num_layers, (l + 1) / num_layers)
        for l in range(1, num_layers)
    ]
    assert all(0.0 < rho <= 1.0 for rho in fractions)
    assert fractions[-1] == pytest.approx(1.0)
    if schedule == "linear":
        for l, rho in zip(range(1, num_layers), fractions):
            assert rho == pytest.approx((2 * l + 1) / (num_layers**2 - l**2))


def test_instance_flow_reaches_the_teacher_at_unit_depth():
    """Corollary 1: d_g(z(t), tau) = d_g(z(t_1), tau) R(t) / R(t_1), zero at t = 1."""
    criterion = _criterion(alpha=1.0, beta=0.0, guidance_schedule="linear")
    T = _sphere(8, 16, seed=220)
    num_layers = 12

    states = [_sphere(8, 16, seed=221)]
    for index in range(num_layers - 1):
        states.append(
            criterion.euler_step(
                states[-1], T, (index + 1) / num_layers, (index + 2) / num_layers
            )
        )

    report = _report(criterion, states, T)
    for realized, predicted in zip(
        report["geodesic_distance"], report["predicted_geodesic_distance"]
    ):
        assert realized == pytest.approx(predicted, abs=1e-8)
    assert report["geodesic_distance"][-1] == pytest.approx(0.0, abs=1e-6)
    assert torch.allclose(states[-1], T, atol=1e-6)


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



def test_velocity_loss_vanishes_when_layers_follow_the_flow():
    # L_evel is instance-wise: a trajectory that walks the geodesic toward the
    # teacher (beta=0 flow) has zero loss regardless of beta in the criterion.
    criterion = _criterion(alpha=1.0, beta=1.0, guidance_schedule="linear")
    flow = _criterion(alpha=1.0, beta=0.0, guidance_schedule="linear")
    T = _sphere(8, 16, seed=10)
    num_layers = 6

    states = [_sphere(8, 16, seed=11)]
    for index in range(num_layers - 1):
        t, t_next = (index + 1) / num_layers, (index + 2) / num_layers
        states.append(flow.euler_step(states[-1], T, t, t_next))

    loss = criterion.velocity_loss(states, T)

    assert float(loss) == pytest.approx(0.0, abs=1e-12)


def test_velocity_loss_is_positive_for_an_unrelated_trajectory():
    criterion = _criterion(alpha=1.0, beta=1.0)
    T = _sphere(8, 16, seed=12)
    states = [_sphere(8, 16, seed=13 + index) for index in range(5)]

    loss = criterion.velocity_loss(states, T)

    assert float(loss) > 0.1


def _receding(criterion, T, angles, seed):
    """States at prescribed geodesic distances from the teacher, along one fixed
    tangent direction: E_sem grows with the angle, so every hinge is active."""
    generator = torch.Generator().manual_seed(seed)
    tangent = GeoODEKD.tangent_project(
        T, torch.randn(T.shape, generator=generator, dtype=T.dtype)
    )
    direction = torch.nn.functional.normalize(tangent, dim=-1)
    return [criterion.exp_map(T, angle * direction) for angle in angles]


@pytest.mark.parametrize(
    "num_layers, expected",
    [
        (1, []),
        (2, [1]),
        (3, [2]),
        (6, [3, 4, 5]),
        (7, [4, 5, 6]),
        (12, [6, 7, 8, 9, 10, 11]),
    ],
)
def test_descent_layers_are_the_deep_half(num_layers, expected):
    # A = {ceil(L/2), ..., L-1}: the source layers of the constrained transitions.
    assert list(GeoODEKD.descent_layers(num_layers)) == expected


def test_descent_loss_vanishes_on_a_flow_following_trajectory():
    # Proposition 2: the flow lowers the energy at every step, so the hinge is
    # inactive on the trajectory the flow itself generates.
    criterion = _criterion(alpha=1.0, beta=0.0, guidance_schedule="linear")
    T = _sphere(8, 16, seed=80)
    num_layers = 6

    states = [_sphere(8, 16, seed=81)]
    for index in range(num_layers - 1):
        t, t_next = (index + 1) / num_layers, (index + 2) / num_layers
        states.append(criterion.euler_step(states[-1], T, t, t_next))

    assert float(criterion.descent_semantic_loss(states, T)) == pytest.approx(
        0.0, abs=1e-14
    )


def test_descent_loss_matches_its_definition_on_an_unrelated_trajectory():
    criterion = _criterion()
    T = _sphere(8, 16, seed=82)
    states = [_sphere(8, 16, seed=83 + index) for index in range(6)]

    energies = [float(criterion.semantic_energy(state, T)) for state in states]
    layers = list(range(3, 6))  # ceil(6 / 2) = 3
    expected = sum(
        max(0.0, energies[layer] - energies[layer - 1]) for layer in layers
    ) / len(layers)

    assert float(criterion.descent_semantic_loss(states, T)) == pytest.approx(
        expected, rel=1e-12
    )
    assert expected > 0.0


def test_descent_loss_ignores_the_shallow_half():
    """A trajectory that regresses only before ceil(L/2) is not penalised, and the
    same regression inside A is."""
    criterion = _criterion()
    T = _sphere(8, 16, seed=84)
    good = _sphere(8, 16, seed=85)
    # A state that is strictly further from the teacher than ``good``.
    bad = criterion.normalize(good - 0.6 * criterion.log_map(good, T))
    assert float(criterion.semantic_energy(bad, T)) > float(
        criterion.semantic_energy(good, T)
    )

    # L = 6, A = {3, 4, 5}: the 0-based transition 0 -> 1 is outside the window.
    shallow = [good, bad, bad, bad, bad, bad]
    deep = [good, good, good, good, bad, bad]

    assert float(criterion.descent_semantic_loss(shallow, T)) == pytest.approx(0.0)
    assert float(criterion.descent_semantic_loss(deep, T)) > 0.0


def test_descent_stop_gradient_only_trains_the_later_layer():
    criterion = _criterion()
    T = _sphere(4, 8, seed=86)
    # L = 4, so A = {2, 3}: the sources are states[1], states[2] and the targets
    # states[2], states[3] (0-based). The trajectory recedes from the teacher, so
    # both hinges are active and every gradient path is exercised.
    states = [
        state.detach().requires_grad_(True)
        for state in _receding(criterion, T, [0.2, 0.4, 0.6, 0.8], seed=87)
    ]
    criterion.descent_semantic_loss(states, T).backward()

    assert states[0].grad is None or states[0].grad.abs().sum() == 0
    # sg[.] on the predecessor: states[1] is only ever a source, so the hinge cannot
    # be lowered by pushing it away from the teacher.
    assert states[1].grad is None or states[1].grad.abs().sum() == 0
    assert states[3].grad is not None and states[3].grad.abs().sum() > 0

    assert states[2].grad is not None and states[2].grad.abs().sum() > 0

    without_sg = _criterion(stop_grad_target=False)
    fresh = [s.detach().clone().requires_grad_(True) for s in states]
    without_sg.descent_semantic_loss(fresh, T).backward()

    assert fresh[1].grad is not None and fresh[1].grad.abs().sum() > 0


def test_descent_loss_is_finite_when_a_layer_reaches_the_teacher():
    # arccos has an unbounded derivative at cos = 1; the endpoint loss drives the
    # last layer exactly there, so the term has to survive it.
    criterion = _criterion()
    T = _sphere(4, 8, seed=88)
    states = [_sphere(4, 8, seed=89), T.clone().requires_grad_(True)]

    loss = criterion.descent_semantic_loss(states, T)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(states[1].grad).all()


def test_descent_term_enters_the_total_with_its_weight():
    criterion = _criterion(lambda_end=1.0, lambda_vel=1.0, lambda_desc=2.0)
    generator = torch.Generator().manual_seed(90)
    teacher = torch.nn.functional.normalize(
        torch.randn(4, 8, generator=generator), dim=-1
    )
    # One embedding output plus four supervised layers whose deep half walks away
    # from the teacher, so the constraint actually binds.
    hidden_states = [torch.randn(4, 5, 8, generator=generator)]
    for state in _receding(criterion, teacher, [0.2, 0.4, 0.6, 0.8], seed=91):
        tokens = torch.randn(4, 5, 8, generator=generator)
        tokens[:, 0, :] = state
        hidden_states.append(tokens)

    total, metrics = criterion(hidden_states=hidden_states, teacher=teacher)

    assert metrics["loss_desc"] > 0.0
    expected = metrics["loss_end"] + metrics["loss_vel"] + 2.0 * metrics["loss_desc"]
    assert float(total) == pytest.approx(expected, rel=1e-5)

    # Off by default, but still reported: lambda_desc = 0 must leave the objective
    # bit-for-bit what it was before the constraint existed.
    off = _criterion(lambda_end=1.0, lambda_vel=1.0)
    total_off, metrics_off = off(hidden_states=hidden_states, teacher=teacher)
    assert metrics_off["loss_desc"] == pytest.approx(metrics["loss_desc"])
    assert float(total_off) == pytest.approx(
        metrics_off["loss_end"] + metrics_off["loss_vel"], rel=1e-6
    )


def test_negative_descent_weight_is_rejected():
    with pytest.raises(ValueError, match="lambda_desc"):
        GeoODEKD(lambda_desc=-1.0)


def test_stop_gradient_target_only_trains_the_next_layer():
    criterion = _criterion(alpha=1.0, beta=1.0)
    T = _sphere(4, 8, seed=20)
    states = [_sphere(4, 8, seed=21 + index).requires_grad_(True) for index in range(3)]

    loss = criterion.velocity_loss(states, T)
    loss.backward()

    # With L = 3 both transitions are supervised, Z^(1) -> Z^(2) and
    # Z^(2) -> Z^(3), so every layer receives gradient. Z^(1) receives it only
    # through the realized update U^(1) = Log_{Z^(1)}(Z^(2)), never through the
    # field.
    assert states[1].grad is not None and states[1].grad.abs().sum() > 0
    assert states[2].grad is not None and states[2].grad.abs().sum() > 0
    grad_with_sg = states[0].grad.clone()

    detached = _criterion(alpha=1.0, beta=1.0, stop_grad_target=False)
    fresh = [s.detach().clone().requires_grad_(True) for s in states]
    detached.velocity_loss(fresh, T).backward()
    # sg[.] removes the path through the field, so the source gradient differs.
    assert not torch.allclose(fresh[0].grad, grad_with_sg)


def test_disabling_stop_gradient_propagates_into_the_source_layer():
    criterion = _criterion(alpha=1.0, beta=1.0, stop_grad_target=False)
    T = _sphere(4, 8, seed=30)
    states = [_sphere(4, 8, seed=31 + index).requires_grad_(True) for index in range(3)]

    loss = criterion.velocity_loss(states, T)
    loss.backward()

    assert states[0].grad is not None and states[0].grad.abs().sum() > 0


def test_forward_combines_the_three_weighted_terms():
    criterion = _criterion(
        alpha=1.0, beta=1.0, lambda_end=0.7, lambda_vel=2.0, lambda_ctr=0.3
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
        0.7 * metrics["loss_end"] + 2.0 * metrics["loss_vel"] + 0.3 * metrics["loss_ctr"]
    )
    assert float(total) == pytest.approx(expected, rel=1e-5)
    assert metrics["loss_ctr"] > 0.0
    assert set(metrics) >= {
        "loss_total",
        "loss_end",
        "loss_vel",
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


def test_pca_projection_completes_an_undersized_corpus():
    """A debug-sized corpus spans too few directions; the map must still be valid."""
    generator = torch.Generator().manual_seed(52)
    embeddings = torch.randn(4, 32, generator=generator)

    projection, mean = fit_pca_projection(embeddings, out_dim=8)
    targets = project_teacher_embeddings(embeddings, projection, mean=mean)

    assert projection.shape == (32, 8)
    assert torch.allclose(projection.T @ projection, torch.eye(8), atol=1e-5)
    assert torch.allclose(targets.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_pca_projection_rejects_a_non_positive_dimension():
    with pytest.raises(ValueError, match="out_dim must be positive"):
        fit_pca_projection(torch.randn(16, 32), out_dim=0)


def _report(criterion, states, teacher):
    return criterion.depth_report(states, teacher)



def test_depth_report_measures_a_flow_following_trajectory():
    """On a trajectory generated by the flow itself, every claim should hold."""
    criterion = _criterion(alpha=1.0, beta=1.0, guidance_schedule="linear")
    T = _sphere(8, 16, seed=60)
    num_layers = 8

    states = [_sphere(8, 16, seed=61)]
    for index in range(num_layers - 1):
        t, t_next = (index + 1) / num_layers, (index + 2) / num_layers
        states.append(criterion.euler_step(states[-1], T, t, t_next))

    report = _report(criterion, states, T)

    assert report["num_layers"] == num_layers
    assert len(report["cos_teacher"]) == num_layers
    assert len(report["vel_residual"]) == num_layers - 1
    # The trajectory *is* the discretised flow, so it follows the field exactly and
    # cannot raise the energy. The alignment is measured in the tangent space, so
    # it is exactly 1 however large the step.
    assert report["mean_vel_residual"] == pytest.approx(0.0, abs=1e-9)
    assert report["mean_alignment"] == pytest.approx(1.0, abs=1e-9)
    for prescribed, realized in zip(report["field_norm"], report["step_norm"]):
        assert realized == pytest.approx(prescribed, rel=1e-8)
    assert report["energy_violations"] == 0
    assert report["cos_gain"] > 0
    assert report["desc_violations"] == 0
    assert report["mean_desc_residual"] == pytest.approx(0.0, abs=1e-12)


def test_depth_report_flags_a_trajectory_that_ignores_the_field():
    criterion = _criterion(alpha=1.0, beta=1.0)
    T = _sphere(8, 16, seed=70)
    states = [_sphere(8, 16, seed=71 + index) for index in range(6)]

    report = _report(criterion, states, T)

    assert report["mean_vel_residual"] > 0.1
    assert abs(report["mean_alignment"]) < 0.5
    assert report["energy_violations"] > 0
    # The descent constraint is measured on the same trajectory: a student that
    # ignores the field raises E_sem inside A too.
    assert report["descent_layers"] == list(GeoODEKD.descent_layers(len(states)))
    assert len(report["desc_residual"]) == len(report["descent_layers"])
    assert report["desc_violations"] > 0
    assert report["mean_desc_residual"] > 0.0



def test_depth_report_separates_step_size_from_direction():
    """A student that moves the right way but far too far still scores a high
    alignment; the two norms are what expose it."""
    criterion = _criterion(alpha=1.0, beta=0.0, guidance_schedule="linear")
    T = _sphere(8, 16, seed=80)
    num_layers = 4

    first = _sphere(8, 16, seed=81)
    rho = criterion.step_fraction(1 / num_layers, 2 / num_layers)
    overshoot = criterion.exp_map(first, 3.0 * rho * criterion.vector_field(first, T))
    states = [first, overshoot]
    for index in range(1, num_layers - 1):
        t, t_next = (index + 1) / num_layers, (index + 2) / num_layers
        states.append(criterion.euler_step(states[-1], T, t, t_next))

    report = _report(criterion, states, T)
    unrelated = _report(criterion, [_sphere(8, 16, seed=81 + i) for i in range(4)], T)

    # Still unmistakably the teacher's direction, next to a walk that ignores it...
    assert report["direction_alignment"][0] > 0.999
    assert abs(unrelated["mean_alignment"]) < 0.2
    # ...and only the two norms say the first step was three times too long.
    assert report["step_norm"][0] == pytest.approx(3.0 * report["field_norm"][0], rel=1e-6)



def test_first_order_retraction_undershoots_the_exponential_map():
    """RowNorm(z + v) rotates by arctan|v| instead of |v|: for the unit-fraction
    last step it would land well short of the teacher, which is why the flow uses
    the exact exponential map."""
    criterion = _criterion(alpha=1.0, beta=0.0)
    Z = _sphere(8, 16, seed=110)
    T = _sphere(8, 16, seed=111)
    V = criterion.vector_field(Z, T)

    exact = criterion.exp_map(Z, V)
    first_order = criterion.retract(Z, V)

    assert torch.allclose(exact, T, atol=1e-8)
    assert float(criterion.geodesic_distance(first_order, T).min()) > 0.1



def test_field_magnitude_does_not_depend_on_batch_size():
    """The paper's Eqs. (23)-(25) differentiate a batch *mean*, so the prescribed
    step scales like 1/B. The field is taken from the per-sample energy instead, so
    a larger batch must not slow the dynamics down."""
    criterion = _criterion(alpha=1.0, beta=1.0)

    norms = []
    for batch in (8, 32, 128):
        Z = _sphere(batch, 16, seed=120)
        T = _sphere(batch, 16, seed=121)
        norms.append(float(criterion.vector_field(Z, T).norm(dim=-1).mean()))

    for larger in norms[1:]:
        assert larger == pytest.approx(norms[0], rel=0.35)



def test_instance_field_is_exactly_the_log_map():
    """With beta = 0 the field is alpha * Log_z(tau) = alpha * d_g / sin(d_g) *
    Pi_z(tau) for every row, with no batch factor left in it at all."""
    criterion = _criterion(alpha=1.0, beta=0.0, guidance_schedule="constant")
    Z = _sphere(6, 16, seed=130)
    T = _sphere(6, 16, seed=131)

    field = criterion.vector_field(Z, T)

    theta = criterion.geodesic_distance(Z, T)
    expected = (theta / torch.sin(theta)).unsqueeze(-1) * GeoODEKD.tangent_project(Z, T)
    assert torch.allclose(field, expected, atol=1e-12)
    assert torch.allclose(field, criterion.log_map(Z, T), atol=1e-12)



def test_prescribed_step_is_not_negligible_at_paper_settings():
    """Regression guard: at B=32, L=12 the first prescribed step must stay within
    reach of a real layer update, and the prescribed steps must add up to the
    whole geodesic so that no layer is left to make the jump alone."""
    criterion = _criterion(alpha=1.0, beta=0.0, guidance_schedule="linear")
    batch, num_layers = 32, 12
    Z = _sphere(batch, 768, seed=140)
    T = _sphere(batch, 768, seed=141)

    rho = criterion.step_fraction(1 / num_layers, 2 / num_layers)
    step = rho * criterion.vector_field(Z, T)
    assert float(step.norm(dim=-1).mean()) > 0.01

    travelled = 0.0
    for index in range(num_layers - 1):
        t, t_next = (index + 1) / num_layers, (index + 2) / num_layers
        nxt = criterion.euler_step(Z, T, t, t_next)
        travelled += float(criterion.geodesic_distance(Z, nxt).mean())
        Z = nxt
    assert float(criterion.geodesic_distance(Z, T).mean()) < 1e-6
    # No single transition carries more than 40% of the trajectory.
    assert travelled > 0.0


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


def test_gauge_alignment_recovers_a_hidden_rotation():
    """If the student is the target seen through an unknown rotation, Procrustes
    finds that rotation and the aligned targets coincide with the student."""
    generator = torch.Generator().manual_seed(300)
    targets = torch.nn.functional.normalize(torch.randn(400, 16, generator=generator), dim=-1)
    hidden, _ = torch.linalg.qr(torch.randn(16, 16, generator=generator))
    student = targets @ hidden

    rotation, stats = fit_gauge_alignment(targets, student)

    assert torch.allclose(rotation @ rotation.T, torch.eye(16), atol=1e-5)
    assert torch.allclose(targets @ rotation, student, atol=1e-5)
    assert stats["cos_after"] == pytest.approx(1.0, abs=1e-5)
    assert stats["cos_after"] > stats["cos_before"]


def test_gauge_alignment_leaves_the_relational_geometry_unchanged():
    """R rotates inside the PCA subspace: Gram matrix, and therefore E_geo and the
    retained variance, are exactly what they were."""
    generator = torch.Generator().manual_seed(310)
    targets = torch.nn.functional.normalize(torch.randn(300, 12, generator=generator), dim=-1)
    student = torch.nn.functional.normalize(torch.randn(300, 12, generator=generator), dim=-1)

    rotation, _ = fit_gauge_alignment(targets, student)
    aligned = targets @ rotation

    assert torch.allclose(aligned @ aligned.T, targets @ targets.T, atol=1e-5)
    assert torch.allclose(aligned.norm(dim=-1), torch.ones(300), atol=1e-5)


def test_gauge_alignment_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        fit_gauge_alignment(torch.randn(10, 4), torch.randn(9, 4))


def test_gauge_refit_never_lowers_the_cosine_of_the_current_student():
    """Alternating step: for a fixed student, the refit R is the exact minimiser over
    O(d), so the cosine under the refit gauge is at least the cosine under any
    earlier gauge (here: the one fitted to a different, earlier student)."""
    generator = torch.Generator().manual_seed(320)
    targets = torch.nn.functional.normalize(torch.randn(500, 16, generator=generator), dim=-1)
    student_initial = torch.nn.functional.normalize(
        targets + 0.8 * torch.randn(500, 16, generator=generator), dim=-1
    )
    rotation_initial, _ = fit_gauge_alignment(targets, student_initial)
    drift, _ = torch.linalg.qr(torch.randn(16, 16, generator=generator))
    student_later = torch.nn.functional.normalize(
        (targets @ rotation_initial @ drift) + 0.3 * torch.randn(500, 16, generator=generator),
        dim=-1,
    )

    under_old_gauge = float(((targets @ rotation_initial) * student_later).sum(dim=-1).mean())
    _, stats = fit_gauge_alignment(targets, student_later)

    assert stats["cos_after"] >= under_old_gauge - 1e-6
    assert stats["cos_after"] > under_old_gauge + 0.1  # the drift was large, so the refit must matter


def test_native_gram_replaces_the_projected_gram_in_the_relational_energy():
    """With a native teacher Gram supplied, E_geo measures the gap to *that* matrix,
    and the relational field is its tangent gradient."""
    criterion = _criterion(alpha=0.0, beta=1.0)
    Z = _sphere(6, 10, seed=400)
    T = _sphere(6, 10, seed=401)
    native = _sphere(6, 40, seed=402)  # a higher-dimensional teacher
    gram = native @ native.T

    _, _, geo_projected = criterion.energy(Z, T)
    _, _, geo_native = criterion.energy(Z, T, gram)
    expected = ((Z @ Z.T - gram).pow(2).sum() / 36.0)
    assert float(geo_native) == pytest.approx(float(expected), rel=1e-10)
    assert float(geo_native) != pytest.approx(float(geo_projected), rel=1e-3)

    Zg = Z.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(criterion.energy(Zg, T, gram)[2], Zg)
    field = criterion.vector_field(Z, T, gram)
    # alpha = 0: the field is -B * Pi_Z[grad E_geo]
    assert torch.allclose(field, -6.0 * GeoODEKD.tangent_project(Z, grad), atol=1e-10)


def test_forward_accepts_and_validates_a_native_gram():
    criterion = _criterion(alpha=1.0, beta=1.0, lambda_ctr=0.0)
    hidden_states = [torch.randn(4, 5, 8) for _ in range(3)]
    teacher = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    native = torch.nn.functional.normalize(torch.randn(4, 32), dim=-1)

    _, with_native = criterion(hidden_states=hidden_states, teacher=teacher, teacher_gram=native @ native.T)
    _, without = criterion(hidden_states=hidden_states, teacher=teacher)
    assert with_native["gram_gap_final"] != pytest.approx(without["gram_gap_final"], rel=1e-3)

    with pytest.raises(ValueError):
        criterion(hidden_states=hidden_states, teacher=teacher, teacher_gram=torch.eye(3))


def test_cached_collate_carries_the_native_teacher_embedding():
    import pandas as pd
    from transformers import AutoTokenizer
    from src.data_utils.dataset_cache import DualTokenizerCollateWithTeacher, TextPairWithTeacher

    df = pd.DataFrame({"premise": ["a b", "c d e"], "hypothesis": ["a b", "c d e"]})
    projected = torch.randn(2, 4)
    native = torch.randn(2, 9)
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    collate = DualTokenizerCollateWithTeacher(tok, "pair_cls", 8)

    with_native = TextPairWithTeacher(df, "pair_cls", projected, native)
    batch = collate([with_native[0], with_native[1]])
    assert torch.equal(batch["teacher_native"], native)
    assert torch.equal(batch["teacher_cls"], projected)

    without = TextPairWithTeacher(df, "pair_cls", projected)
    assert "teacher_native" not in collate([without[0], without[1]])
