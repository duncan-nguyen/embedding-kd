"""Tests for the in-training structural probe and the weight-drift tracker.

Both are instrumentation, and the property that makes them usable inside a seeded
ablation is that they cannot move the run: the probe encodes in ``eval()`` under
``no_grad``, so it draws no dropout mask and leaves the RNG stream where it was.
That, and the reading each one claims to give, is what these pin.
"""

import pytest
import torch

from src.training_probe import TrainingProbe, WeightDriftTracker


class _Encoder(torch.nn.Module):
    """A stand-in student: an embedding bag, a dropout, a linear "layer" stack."""

    def __init__(self, vocab: int = 40, dim: int = 8, layers: int = 3):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab, dim)
        self.encoder = torch.nn.ModuleDict(
            {"layer": torch.nn.ModuleList(torch.nn.Linear(dim, dim) for _ in range(layers))}
        )
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask, return_dict=True):
        hidden = self.dropout(self.embeddings(input_ids))
        for layer in self.encoder["layer"]:
            hidden = layer(hidden)
        return type("Out", (), {"last_hidden_state": hidden})()


class _Tokenizer:
    def __call__(self, texts, max_length, truncation, padding, return_tensors):
        rows = [[(len(t) + i) % 39 + 1 for i in range(4)] for t in texts]
        return {
            "input_ids": torch.tensor(rows),
            "attention_mask": torch.ones(len(rows), 4, dtype=torch.long),
        }


def _probe(n: int = 24, targets: torch.Tensor | None = None, **kwargs) -> TrainingProbe:
    generator = torch.Generator().manual_seed(5)
    texts = [f"sentence number {i}" for i in range(n)]
    teacher = torch.randn(n, 32, generator=generator)
    if targets is None:
        targets = torch.randn(n, 8, generator=generator)
    return TrainingProbe(
        texts=texts,
        teacher=teacher,
        targets=targets,
        tokenizer=_Tokenizer(),
        pool=lambda hidden, mask: hidden[:, 0, :],
        knn_k=kwargs.pop("knn_k", 4),
        **kwargs,
    )


def test_the_probe_reports_every_rung_of_the_ladder():
    probe = _probe()

    measured = probe.measure(_Encoder(), "cpu")

    assert {
        "probe_cos_target",  # rung 1
        "probe_gram_rmse_teacher",  # rung 2
        "probe_cka_teacher",
        "probe_knn_overlap_teacher",  # rung 3
        "probe_mutual_knn_teacher",
        "probe_h0_w1_teacher",  # rung 4
        "probe_effective_rank",
        "probe_twonn",
        "probe_anisotropy",
    } <= set(measured)
    assert all(isinstance(value, float) for value in measured.values())


def test_the_probe_draws_no_dropout_mask_and_leaves_the_rng_where_it_was():
    """The reason it is safe to switch on inside a seeded ablation."""
    probe = _probe()
    model = _Encoder()
    model.train()

    torch.manual_seed(0)
    without = torch.randn(3)

    torch.manual_seed(0)
    probe.measure(model, "cpu")
    with_probe = torch.randn(3)

    assert torch.equal(without, with_probe)
    # ... and it hands the model back in the mode it found it in.
    assert model.training


def test_the_probe_is_deterministic_on_unchanged_weights():
    probe = _probe()
    model = _Encoder()

    first = probe.measure(model, "cpu")
    second = probe.measure(model, "cpu")

    # repr rather than ==: TwoNN is undefined on a cloud with too few distinct
    # nearest-neighbour ratios and reports NaN there, which never equals itself.
    assert {k: repr(v) for k, v in first.items()} == {
        k: repr(v) for k, v in second.items()
    }


def test_a_method_without_a_target_reports_the_rungs_above_the_first():
    """Rung 1 is a coordinate cosine; without a map there is nothing to take it against."""
    probe = _probe()
    probe.targets = None

    measured = probe.measure(_Encoder(), "cpu")

    assert "probe_cos_target" not in measured
    assert "probe_procrustes_target" not in measured
    assert "probe_cka_teacher" in measured
    assert "target_cka_teacher" not in probe.reference()


def test_the_reference_row_is_the_ceiling_the_interface_leaves():
    """A perfect interface scores perfectly against the teacher; a student is read
    against that, not against 1."""
    generator = torch.Generator().manual_seed(6)
    teacher = torch.randn(24, 32, generator=generator)
    rotation, _ = torch.linalg.qr(torch.randn(32, 32, generator=generator))
    probe = TrainingProbe(
        texts=[f"sentence {i}" for i in range(24)],
        teacher=teacher,
        # The teacher seen through an orthogonal map keeps every rung above the first.
        targets=teacher @ rotation,
        tokenizer=_Tokenizer(),
        pool=lambda hidden, mask: hidden[:, 0, :],
        knn_k=4,
    )

    reference = probe.reference()

    assert reference["target_gram_rmse_teacher"] == pytest.approx(0.0, abs=1e-4)
    assert reference["target_cka_teacher"] == pytest.approx(1.0, abs=1e-4)
    assert reference["target_knn_overlap_teacher"] == pytest.approx(1.0, abs=1e-6)


def test_a_probe_too_small_for_its_neighbourhood_is_rejected_at_construction():
    with pytest.raises(ValueError, match="probe_knn_k"):
        _probe(n=4, knn_k=10)


def test_a_probe_whose_sides_disagree_on_length_is_rejected():
    with pytest.raises(ValueError, match="teacher"):
        TrainingProbe(
            texts=["a", "b"],
            teacher=torch.randn(3, 8),
            targets=None,
            tokenizer=_Tokenizer(),
            pool=lambda hidden, mask: hidden[:, 0, :],
            knn_k=1,
        )


# --------------------------------------------------------------------------- #
# Weight drift
# --------------------------------------------------------------------------- #


def test_drift_is_zero_at_the_start_and_names_one_group_per_depth():
    model = _Encoder(layers=3)
    tracker = WeightDriftTracker(model)

    measured = tracker.measure(model)

    assert tracker.groups == ["embeddings", "layer_00", "layer_01", "layer_02"]
    assert all(value == pytest.approx(0.0, abs=1e-6) for value in measured.values())


def test_drift_isolates_the_depth_that_moved():
    """The reading the depth question needs: which layers the objective reaches."""
    model = _Encoder(layers=3)
    tracker = WeightDriftTracker(model)

    with torch.no_grad():
        model.encoder["layer"][1].weight.mul_(1.1)

    measured = tracker.measure(model)

    assert measured["drift_layer_01"] > 1e-3
    assert measured["drift_layer_00"] == pytest.approx(0.0, abs=1e-6)
    assert measured["drift_layer_02"] == pytest.approx(0.0, abs=1e-6)
    assert measured["drift_embeddings"] == pytest.approx(0.0, abs=1e-6)
    # The whole-model reading is between the depth that moved and the ones that did not.
    assert 0.0 < measured["drift_model"] < measured["drift_layer_01"]


def test_drift_is_relative_so_depths_of_different_size_are_comparable():
    model = _Encoder(layers=2)
    tracker = WeightDriftTracker(model)

    with torch.no_grad():
        for layer in model.encoder["layer"]:
            layer.weight.mul_(1.05)
            layer.bias.mul_(1.05)

    measured = tracker.measure(model)

    assert measured["drift_layer_00"] == pytest.approx(
        measured["drift_layer_01"], rel=0.05
    )


@pytest.mark.parametrize(
    "name,expected",
    [
        ("embeddings.word_embeddings.weight", "embeddings"),
        ("encoder.layer.0.attention.self.query.weight", "layer_00"),
        ("encoder.layer.11.output.dense.bias", "layer_11"),
        ("model.layers.5.mlp.gate_proj.weight", "layer_05"),
        ("pooler.dense.weight", "other"),
    ],
)
def test_parameter_names_land_in_the_depth_they_belong_to(name, expected):
    assert WeightDriftTracker.group_of(name) == expected
