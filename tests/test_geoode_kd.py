"""Tests for GeoODE-KD.

The objective is L_end + L_ctr: an endpoint anchored on the frozen teacher target
(Eq. 36) and an InfoNCE regulariser over two dropout views (Eq. 37). These tests pin
that objective, the pooling that feeds it, and the frozen map P_T = P_PCA R the
targets arrive through -- the part of the recipe that decides *where* the endpoint
is, now that nothing supervises the path taken to reach it.
"""

import pytest
import torch

from src.criterions.geoode_kd import GeoODEKD
from src.teacher_projection import (
    fit_gauge_alignment,
    fit_pca_projection,
    project_teacher_embeddings,
)


def _criterion(**kwargs) -> GeoODEKD:
    return GeoODEKD(**kwargs)


def test_criterion_has_no_trainable_parameters():
    # The paper's claim that training adds no module and inference is unchanged rests
    # on this: the objective is analytic in the cached targets.
    assert list(_criterion().parameters()) == []


def test_forward_combines_the_two_weighted_terms():
    criterion = _criterion(lambda_end=0.7, lambda_ctr=0.3)
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

    expected = 0.7 * metrics["loss_end"] + 0.3 * metrics["loss_ctr"]
    assert float(total) == pytest.approx(expected, rel=1e-5)
    assert metrics["loss_ctr"] > 0.0
    assert set(metrics) == {
        "loss_total",
        "loss_end",
        "loss_ctr",
        "loss_gram",
        "loss_topo",
        "cos_first",
        "cos_final",
    }


def test_negative_objective_weights_are_rejected():
    with pytest.raises(ValueError, match="must be non-negative"):
        GeoODEKD(lambda_end=-1.0)
    with pytest.raises(ValueError, match="must be non-negative"):
        GeoODEKD(lambda_ctr=-1.0)

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
    """R rotates inside the PCA subspace: the Gram matrix, and therefore the
    relational geometry and the retained variance, are exactly what they were."""
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

def test_the_supervised_depth_is_the_last_layer_alone():
    """``include_embedding_layer`` shifts which state ``cos_first`` reports, but only
    the final layer carries loss, so it cannot move the objective. The knob reads as
    an ablation in ``--help``; this is what keeps that reading honest."""
    generator = torch.Generator().manual_seed(400)
    hidden = [torch.randn(8, 5, 16, generator=generator) for _ in range(7)]
    teacher = torch.nn.functional.normalize(
        torch.randn(8, 16, generator=generator), dim=-1
    )
    second_view = torch.randn(8, 16, generator=generator)

    def run(**overrides):
        return GeoODEKD(**overrides)(
            hidden_states=hidden, teacher=teacher, second_view=second_view
        )

    reference, base_metrics = run()
    shifted, shifted_metrics = run(include_embedding_layer=True)

    assert float(shifted) == float(reference)
    assert shifted_metrics["cos_first"] != base_metrics["cos_first"]
    assert shifted_metrics["cos_final"] == base_metrics["cos_final"]


def test_both_active_weights_move_the_objective():
    """The other half of the claim: the test above passes because the knob is inert,
    not because the harness cannot see a change."""
    generator = torch.Generator().manual_seed(401)
    hidden = [torch.randn(8, 5, 16, generator=generator) for _ in range(7)]
    teacher = torch.nn.functional.normalize(
        torch.randn(8, 16, generator=generator), dim=-1
    )
    second_view = torch.randn(8, 16, generator=generator)

    def total(**overrides):
        loss, _ = GeoODEKD(**overrides)(
            hidden_states=hidden, teacher=teacher, second_view=second_view
        )
        return float(loss)

    reference = total()
    assert total(lambda_ctr=0.0) != reference
    assert total(lambda_end=0.0) != reference


def test_mse_endpoint_regresses_the_raw_state_onto_the_raw_target():
    """The MSE baseline reads both sides unnormalised; the reduction is the
    per-sample squared distance averaged over the batch (d_S x nn.MSELoss), so the
    term is on the scale of the cosine endpoint; the cosine diagnostics keep
    reporting on the normalised copies."""
    criterion = _criterion(lambda_end=1.0, lambda_ctr=0.0, endpoint_loss="mse")
    batch, dim, tokens, layers = 3, 6, 4, 2
    generator = torch.Generator().manual_seed(7)
    hidden_states = [
        torch.randn(batch, tokens, dim, generator=generator) for _ in range(layers + 1)
    ]
    teacher = 0.7 * torch.randn(batch, dim, generator=generator)  # deliberately not unit

    total, metrics = criterion(hidden_states=hidden_states, teacher=teacher)

    raw_final = hidden_states[-1][:, 0, :]
    expected = ((raw_final - teacher) ** 2).sum(dim=-1).mean()
    assert float(expected) == pytest.approx(dim * float(torch.nn.functional.mse_loss(raw_final, teacher)), rel=1e-5)
    assert float(total) == pytest.approx(float(expected), rel=1e-5)
    assert metrics["loss_end"] == pytest.approx(float(expected), rel=1e-5)
    unit_final = torch.nn.functional.normalize(raw_final, dim=-1)
    unit_teacher = torch.nn.functional.normalize(teacher, dim=-1)
    assert metrics["cos_final"] == pytest.approx(
        float((unit_final * unit_teacher).sum(-1).mean()), rel=1e-5
    )


def test_cosine_endpoint_is_unchanged_by_the_new_option():
    criterion = _criterion(lambda_end=1.0, lambda_ctr=0.0)
    batch, dim, tokens = 3, 6, 4
    generator = torch.Generator().manual_seed(8)
    hidden_states = [torch.randn(batch, tokens, dim, generator=generator) for _ in range(3)]
    teacher = torch.randn(batch, dim, generator=generator)
    total, _ = criterion(hidden_states=hidden_states, teacher=teacher)
    unit_final = torch.nn.functional.normalize(hidden_states[-1][:, 0, :], dim=-1)
    unit_teacher = torch.nn.functional.normalize(teacher, dim=-1)
    assert float(total) == pytest.approx(
        float((1 - (unit_final * unit_teacher).sum(-1)).mean()), rel=1e-5
    )


def test_unknown_endpoint_loss_and_mse_with_projector_are_rejected():
    with pytest.raises(ValueError, match="endpoint_loss"):
        _criterion(endpoint_loss="huber")
    with pytest.raises(ValueError, match="learned target projector"):
        _criterion(endpoint_loss="mse", target_projector=torch.nn.Linear(2, 2))


def test_projection_can_skip_the_final_renormalisation():
    generator = torch.Generator().manual_seed(9)
    teacher = torch.nn.functional.normalize(torch.randn(64, 12, generator=generator), dim=-1)
    projection, mean = fit_pca_projection(teacher, out_dim=4, center=True)
    raw = project_teacher_embeddings(teacher, projection, mean=mean, renormalize=False)
    unit = project_teacher_embeddings(teacher, projection, mean=mean)
    assert torch.allclose(raw, teacher @ projection)
    assert torch.allclose(unit, torch.nn.functional.normalize(raw, dim=-1))
    assert (raw.norm(dim=-1) <= 1.0 + 1e-6).all()


# --------------------------------------------------------------------------- #
# Reading only the depths the objective uses, and the diagram built elsewhere
# --------------------------------------------------------------------------- #


def _hidden_stack(layers: int, batch: int = 4, tokens: int = 5, dim: int = 8, seed: int = 5):
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randn(batch, tokens, dim, generator=generator) for _ in range(layers + 1)
    ]


@pytest.mark.parametrize("layers", [1, 2, 6, 12])
@pytest.mark.parametrize("pooling", ["cls", "mean"])
@pytest.mark.parametrize("include_embedding_layer", [False, True])
def test_endpoint_states_are_the_two_ends_of_layer_states(
    layers, pooling, include_embedding_layer
):
    """Every loss term reads Z^(L) and cos_first reads Z^(1); the L-2 depths in
    between were pooled and normalised only to be dropped."""
    criterion = _criterion(
        pooling=pooling, include_embedding_layer=include_embedding_layer
    )
    hidden_states = _hidden_stack(layers)
    mask = torch.ones(4, 5, dtype=torch.long)
    mask[1, 3:] = 0

    full = criterion.layer_states(hidden_states, mask)
    endpoints = criterion.endpoint_states(hidden_states, mask)

    assert torch.equal(endpoints[0], full[0])
    assert torch.equal(endpoints[-1], full[-1])
    assert len(endpoints) == (1 if len(full) == 1 else 2)


def test_endpoint_states_on_a_single_depth_is_one_state():
    """With one supervised depth ``states[0]`` and ``states[-1]`` are the same
    state, exactly as they were when every depth was pooled."""
    criterion = _criterion()
    endpoints = criterion.endpoint_states([torch.randn(3, 4, 8)], None)

    assert len(endpoints) == 1
    assert endpoints[0] is endpoints[-1]


def test_the_forward_reads_the_shallow_depth_for_cos_first():
    """cos_first has to keep meaning the first supervised depth, not the last."""
    criterion = _criterion(lambda_ctr=0.0)
    hidden_states = _hidden_stack(6)
    teacher = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)

    _, metrics = criterion(hidden_states=hidden_states, teacher=teacher)
    states = criterion.layer_states(hidden_states, None)
    normalised_teacher = criterion.normalize(teacher)

    assert metrics["cos_first"] == pytest.approx(
        float((states[0] * normalised_teacher).sum(dim=-1).mean()), abs=1e-6
    )
    assert metrics["cos_final"] == pytest.approx(
        float((states[-1] * normalised_teacher).sum(dim=-1).mean()), abs=1e-6
    )
    assert metrics["cos_first"] != pytest.approx(metrics["cos_final"], abs=1e-6)


def test_a_learned_projector_still_sees_both_endpoints():
    """The s2t baseline maps every state it is given into the teacher space; it has
    to be handed the shallow one too or cos_first reports an unmapped vector."""
    from src.target_projector import LearnedTargetProjector

    projector = LearnedTargetProjector(teacher_dim=16, student_dim=8, direction="s2t")
    criterion = _criterion(lambda_ctr=0.0, target_projector=projector)
    hidden_states = _hidden_stack(6)
    teacher = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)

    total, metrics = criterion(hidden_states=hidden_states, teacher=teacher)

    assert torch.isfinite(total)
    assert abs(metrics["cos_first"]) <= 1.0 and abs(metrics["cos_final"]) <= 1.0


def test_a_precomputed_teacher_diagram_matches_computing_it_in_the_step():
    """The H0 term's teacher side is a constant read from a frozen cache, so the
    collate may build it in a worker instead of the GPU building it mid-step."""
    from src.criterions.h0_topological_loss import h0_death_times

    criterion = _criterion(lambda_end=1.0, lambda_ctr=0.0, lambda_topo=0.5)
    hidden_states = _hidden_stack(4)
    teacher = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    teacher_topo = torch.randn(4, 32)

    in_step, in_step_metrics = criterion(
        hidden_states=hidden_states, teacher=teacher, teacher_topo=teacher_topo
    )
    with torch.no_grad():
        deaths = h0_death_times(teacher_topo, metric="chord", sort=True)
    precomputed, precomputed_metrics = criterion(
        hidden_states=hidden_states, teacher=teacher, teacher_deaths=deaths
    )

    assert float(in_step) == pytest.approx(float(precomputed), abs=1e-7)
    assert in_step_metrics["loss_topo"] == pytest.approx(
        precomputed_metrics["loss_topo"], abs=1e-7
    )
    assert precomputed_metrics["loss_topo"] > 0.0


def test_a_precomputed_diagram_wins_over_the_raw_cache():
    """Given both, the reduced one is what the raw one would have been reduced to,
    so it is read and the raw copy is ignored rather than recomputed."""
    from src.criterions.h0_topological_loss import h0_death_times

    criterion = _criterion(lambda_end=0.0, lambda_ctr=0.0, lambda_topo=1.0)
    hidden_states = _hidden_stack(4)
    teacher = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    topo = torch.randn(4, 32)

    _, metrics = criterion(
        hidden_states=hidden_states,
        teacher=teacher,
        teacher_topo=torch.zeros(4, 32),  # would give a different diagram
        teacher_deaths=h0_death_times(topo, metric="chord", sort=True),
    )
    _, reference = criterion(
        hidden_states=hidden_states, teacher=teacher, teacher_topo=topo
    )

    assert metrics["loss_topo"] == pytest.approx(reference["loss_topo"], abs=1e-7)
