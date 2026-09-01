"""What the training loop stopped paying per step, and what it still computes.

Every optimisation here is meant to be invisible in the numbers: the step got
cheaper, not different. So each test pins the equivalence rather than the saving --
the batch still carries what the method reads, the metrics still say what they said,
the fused views are still the pair the objective asks for.

The one exception is called out where it is tested: fusing two dropout passes into
one draws the masks in a single RNG call, so a seeded run follows a different
trajectory. That is a reproducibility fact, not a numerical one, and
``--no-fused_views`` is the way back.
"""

import time
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn

from config import GeoODEConfig, RKDConfig, SimCSEConfig
from distiller import KnowledgeDistiller, StepTimer
from src.criterions.h0_topological_loss import h0_death_times
from src.data_utils.dataset_cache import (
    DualTokenizerCollateWithTeacher,
    TextPairWithTeacher,
)
from src.metrics import scalar_metrics

# --------------------------------------------------------------------------- #
# scalar_metrics: one device read instead of one per number
# --------------------------------------------------------------------------- #


def test_scalar_metrics_is_what_calling_float_on_each_would_have_built():
    values = {
        "loss_total": torch.tensor(1.25),
        "loss_end": torch.tensor(0.5),
        "cos_final": torch.tensor([0.1, 0.3]).mean(),
    }

    assert scalar_metrics(**values) == {
        name: float(tensor) for name, tensor in values.items()
    }


def test_scalar_metrics_keeps_the_order_it_was_given():
    """The order is what the progress bar and the per-step records read back."""
    metrics = scalar_metrics(
        loss_total=torch.tensor(3.0), loss_end=torch.tensor(1.0), loss_ctr=torch.tensor(2.0)
    )

    assert list(metrics) == ["loss_total", "loss_end", "loss_ctr"]


def test_scalar_metrics_carries_plain_numbers_through():
    """A criterion may report a constant it already knows next to a tensor."""
    metrics = scalar_metrics(loss_total=torch.tensor(2.0), skipped=0, weight=0.5)

    assert metrics == {"loss_total": 2.0, "skipped": 0.0, "weight": 0.5}
    assert all(isinstance(value, float) for value in metrics.values())


def test_scalar_metrics_reads_the_device_once():
    """The point of the helper: N diagnostics cost one synchronisation, not N. On
    CPU there is nothing to synchronise, so the count is pinned on the stack that
    would have been the read."""
    reads = []
    original = torch.Tensor.tolist

    def counting_tolist(self):
        reads.append(tuple(self.shape))
        return original(self)

    torch.Tensor.tolist = counting_tolist
    try:
        scalar_metrics(**{f"m{i}": torch.tensor(float(i)) for i in range(7)})
    finally:
        torch.Tensor.tolist = original

    assert reads == [(7,)]


def test_scalar_metrics_detaches_so_nothing_holds_the_graph():
    tensor = (torch.randn(4, requires_grad=True) ** 2).mean()

    metrics = scalar_metrics(loss=tensor)

    assert isinstance(metrics["loss"], float)


# --------------------------------------------------------------------------- #
# StepTimer: the same interval, read without stalling the queue
# --------------------------------------------------------------------------- #


def test_the_step_timer_returns_one_duration_per_step_in_order():
    timer = StepTimer()
    for _ in range(5):
        timer.start()
        timer.stop()

    durations = timer.finish()

    assert len(durations) == 5
    assert all(seconds >= 0.0 for seconds in durations)


def test_the_step_timer_measures_the_interval_it_wraps():
    timer = StepTimer()
    timer.start()
    time.sleep(0.02)
    timer.stop()

    assert timer.finish()[0] == pytest.approx(0.02, abs=0.02)


def test_take_new_hands_back_each_duration_exactly_once():
    """The epoch loop folds durations into its moving average as they complete, so
    a duration handed out twice would double-count and one never handed out would
    vanish from the average."""
    timer = StepTimer()
    seen = []
    for _ in range(4):
        timer.start()
        timer.stop()
        seen.extend(timer.take_new())
    seen.extend(timer.take_new())

    assert len(seen) == 4
    assert timer.finish() == seen


def test_stopping_without_starting_is_an_error():
    with pytest.raises(RuntimeError, match="without a matching start"):
        StepTimer().stop()


# --------------------------------------------------------------------------- #
# The cached-teacher collate: it builds what the method is going to read
# --------------------------------------------------------------------------- #


class _CharTokenizer:
    """Right-padding whitespace tokenizer that answers the flags the collate sets."""

    def __call__(self, texts, max_length=None, truncation=True, padding=True,
                 return_tensors="pt", return_special_tokens_mask=False, **_):
        rows = [[ord(c) % 97 for c in text][:max_length] or [0] for text in texts]
        width = max(len(row) for row in rows)
        input_ids = torch.zeros(len(rows), width, dtype=torch.long)
        attention_mask = torch.zeros(len(rows), width, dtype=torch.long)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row)
            attention_mask[index, : len(row)] = 1
        encoded = {"input_ids": input_ids, "attention_mask": attention_mask}
        if return_special_tokens_mask:
            encoded["special_tokens_mask"] = torch.zeros_like(input_ids)
        return encoded


def _batch(rows: int = 4, teacher_dim: int = 6, topo_dim: int = 10, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    frame = pd.DataFrame(
        {
            "premise": [f"premise {index}" for index in range(rows)],
            "hypothesis": [f"hypothesis number {index}" for index in range(rows)],
        }
    )
    teacher_cls = torch.randn(rows, teacher_dim, generator=generator)
    teacher_topo = torch.randn(rows, topo_dim, generator=generator)
    dataset = TextPairWithTeacher(frame, "pair_cls", teacher_cls, teacher_topo)
    return [dataset[index] for index in range(rows)], teacher_topo


def test_the_collate_leaves_out_the_second_text_when_nothing_reads_it():
    """GeoODE's default contrastive view is two dropout passes over the *first*
    sentence, so the second one was tokenized, padded, stacked and copied to the GPU
    every step without ever being read."""
    samples, _ = _batch()

    with_second = DualTokenizerCollateWithTeacher(
        _CharTokenizer(), "pair_cls", 32, need_second_text=True
    )(samples)
    without = DualTokenizerCollateWithTeacher(
        _CharTokenizer(), "pair_cls", 32, need_second_text=False
    )(samples)

    assert "input_ids2_stu" in with_second
    assert "input_ids2_stu" not in without
    # The side that is read is untouched by the decision about the other one.
    assert torch.equal(with_second["input_ids1_stu"], without["input_ids1_stu"])
    assert torch.equal(with_second["teacher_cls"], without["teacher_cls"])


def test_the_collate_leaves_out_the_special_tokens_mask_by_default():
    """Only the token-level methods (cdm, dskd) align token strings, and neither
    reads a teacher cache -- so for everything this collate serves it was dead."""
    samples, _ = _batch()

    default = DualTokenizerCollateWithTeacher(_CharTokenizer(), "pair_cls", 32)(samples)
    asked = DualTokenizerCollateWithTeacher(
        _CharTokenizer(), "pair_cls", 32, need_special_tokens_mask=True
    )(samples)

    assert not any(key.startswith("special_tokens_mask") for key in default)
    assert "special_tokens_mask1_stu" in asked


def test_the_collate_builds_the_teacher_diagram_the_step_would_have_built():
    """Moving it here is a scheduling change: the same reduction, in a DataLoader
    worker in parallel with the previous step instead of on the GPU mid-step."""
    samples, teacher_topo = _batch()

    batch = DualTokenizerCollateWithTeacher(
        _CharTokenizer(), "pair_cls", 32, topo_metric="chord"
    )(samples)

    assert "teacher_deaths" in batch
    assert "teacher_topo" not in batch  # a [B-1] copy replaces the [B, d_T] one
    assert torch.allclose(
        batch["teacher_deaths"],
        h0_death_times(teacher_topo.float(), metric="chord", sort=True),
        atol=1e-7,
    )


def test_without_a_metric_the_collate_still_ships_the_raw_cache():
    samples, teacher_topo = _batch()

    batch = DualTokenizerCollateWithTeacher(_CharTokenizer(), "pair_cls", 32)(samples)

    assert torch.equal(batch["teacher_topo"], teacher_topo)
    assert "teacher_deaths" not in batch


def test_a_one_row_tail_batch_has_no_diagram_to_build():
    """B - 1 death times of a batch of one is an empty diagram, and the MST of a
    single point does not exist; the criterion drops the term for that batch."""
    samples, teacher_topo = _batch(rows=1)

    batch = DualTokenizerCollateWithTeacher(
        _CharTokenizer(), "pair_cls", 32, topo_metric="chord"
    )(samples)

    assert "teacher_deaths" not in batch
    assert torch.equal(batch["teacher_topo"], teacher_topo)


@pytest.mark.parametrize("metric", ["chord", "angular", "cosine"])
def test_the_collate_follows_the_configured_ground_metric(metric):
    samples, teacher_topo = _batch()

    batch = DualTokenizerCollateWithTeacher(
        _CharTokenizer(), "pair_cls", 32, topo_metric=metric
    )(samples)

    assert torch.allclose(
        batch["teacher_deaths"],
        h0_death_times(teacher_topo.float(), metric=metric, sort=True),
        atol=1e-7,
    )


# --------------------------------------------------------------------------- #
# Which of those switches each method asks for
# --------------------------------------------------------------------------- #


def _stub(config) -> KnowledgeDistiller:
    stub = object.__new__(KnowledgeDistiller)
    stub.config = config
    return stub


def test_geoode_asks_for_the_second_text_only_when_its_view_is_the_pair():
    dropout = _stub(GeoODEConfig(contrastive_view="dropout", lambda_ctr=0.5))
    pair = _stub(GeoODEConfig(contrastive_view="pair", lambda_ctr=0.5))
    no_contrastive = _stub(GeoODEConfig(contrastive_view="pair", lambda_ctr=0.0))

    assert not dropout._needs_second_text()
    assert pair._needs_second_text()
    # With no contrastive term there is no second view at all.
    assert not no_contrastive._needs_second_text()


def test_the_other_cached_methods_keep_their_paired_sentence():
    """TALAS and RKD take the paired sentence as the in-batch positive, so for them
    the second text is the objective, not overhead."""
    assert _stub(RKDConfig())._needs_second_text()


def test_a_single_text_task_never_has_a_second_text():
    assert not _stub(GeoODEConfig(task_type="single_cls"))._needs_second_text()


# --------------------------------------------------------------------------- #
# Fusing the two dropout views into one forward
# --------------------------------------------------------------------------- #


class _TinyEncoder(nn.Module):
    """A stand-in student: an embedding, a linear layer and dropout between them."""

    def __init__(self, vocab: int = 128, dim: int = 8, dropout: float = 0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.config = SimpleNamespace(hidden_size=dim)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False,
                return_dict=True, **_):
        first = self.dropout(self.embedding(input_ids))
        last = self.linear(first)
        return SimpleNamespace(
            last_hidden_state=last,
            hidden_states=(first, last) if output_hidden_states else None,
        )


def _fusion_stub(dropout: float = 0.5, **config_kwargs) -> KnowledgeDistiller:
    stub = _stub(GeoODEConfig(**config_kwargs))
    stub.model_student = _TinyEncoder(dropout=dropout)
    return stub


def test_with_dropout_off_the_fused_pass_is_the_two_passes_exactly():
    """Dropout is the only thing that makes the two views differ, so with it off the
    fused forward has to reproduce both halves bit for bit -- that is the check that
    the doubled batch is being sliced back correctly."""
    stub = _fusion_stub(dropout=0.0)
    stub.model_student.eval()
    input_ids = torch.randint(0, 128, (5, 7))
    attention_mask = torch.ones(5, 7, dtype=torch.long)

    fused, second = stub._dropout_pair_forward(
        input_ids, attention_mask, hidden_states=True
    )
    reference = stub.model_student(
        input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
    )

    assert torch.equal(fused.last_hidden_state, reference.last_hidden_state)
    assert len(fused.hidden_states) == len(reference.hidden_states)
    for got, want in zip(fused.hidden_states, reference.hidden_states):
        assert torch.equal(got, want)
    assert torch.equal(second, stub._pool_student(reference.last_hidden_state, attention_mask))


def test_the_fused_halves_keep_the_batch_shape_they_were_given():
    stub = _fusion_stub()
    input_ids = torch.randint(0, 128, (6, 4))
    attention_mask = torch.ones(6, 4, dtype=torch.long)

    fused, second = stub._dropout_pair_forward(
        input_ids, attention_mask, hidden_states=True
    )

    assert fused.last_hidden_state.shape == (6, 4, 8)
    assert all(state.shape == (6, 4, 8) for state in fused.hidden_states)
    assert second.shape == (6, 8)


def test_the_two_fused_views_really_are_two_draws_of_the_mask():
    """The pair the objective asks for is the same sentence under independent
    dropout. Stacking the batch on itself gives that -- dropout samples per element,
    not per call -- and a fused pass that reused one mask would make the contrastive
    term trivially zero."""
    stub = _fusion_stub(dropout=0.5)
    stub.model_student.train()
    torch.manual_seed(0)
    input_ids = torch.randint(0, 128, (8, 6))
    attention_mask = torch.ones(8, 6, dtype=torch.long)

    fused, second = stub._dropout_pair_forward(
        input_ids, attention_mask, hidden_states=False
    )
    first = stub._pool_student(fused.last_hidden_state, attention_mask)

    assert not torch.allclose(first, second)
    # ... but of the same sentences, so each row still tracks its own text.
    assert torch.nn.functional.cosine_similarity(first, second).mean() > 0.0


def test_hidden_states_are_only_carried_when_asked_for():
    """SimCSE reads a pooled vector and nothing else; GeoODE needs the stack."""
    stub = _fusion_stub()

    without, _ = stub._dropout_pair_forward(
        torch.randint(0, 128, (3, 4)), torch.ones(3, 4, dtype=torch.long),
        hidden_states=False,
    )

    assert without.hidden_states is None


def test_fusion_is_only_for_the_dropout_view():
    """The "pair" view encodes a *different* sentence, whose padded width need not
    match; fusing it would mean re-padding both sides to the longer of the two."""
    stub = _fusion_stub(fused_views=True)

    assert stub._fuses_dropout_views("dropout")
    assert not stub._fuses_dropout_views("pair")


def test_the_flag_restores_the_two_pass_order():
    """One RNG draw of 2B masks consumes the generator differently from two draws of
    B, so a seeded run recorded before fusion needs the old order to reproduce."""
    assert not _fusion_stub(fused_views=False)._fuses_dropout_views("dropout")
    assert _stub(SimCSEConfig())._fuses_dropout_views("dropout")


# --------------------------------------------------------------------------- #
# End to end: the wiring these switches run through
# --------------------------------------------------------------------------- #


class _StubTeacher(nn.Module):
    """Stands in for the frozen teacher during the one-off caching pass."""

    def __init__(self, dim: int = 16):
        super().__init__()
        # Spread frequencies, so two different sentences really do land in different
        # places. A constant scale would map every row to a multiple of the all-ones
        # vector, i.e. to one point on the sphere, and no geometry test could say
        # anything about a cloud that has collapsed before the projection sees it.
        self.register_buffer("scale", torch.linspace(0.1, 2.0, dim))
        self.config = SimpleNamespace(hidden_size=dim)

    def forward(self, input_ids, attention_mask=None, return_dict=True, **_):
        # Each position carries a running total of the sentence so far, so the state
        # the pooling reads is a summary of the whole row rather than of its last
        # token -- which is what a real encoder gives, and what stops rows that
        # happen to end on the same character from sharing an embedding.
        mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
        running = torch.cumsum(input_ids.float() * mask, dim=1)
        return SimpleNamespace(last_hidden_state=torch.sin(running.unsqueeze(-1) * self.scale))


class _StubStudent(_TinyEncoder):
    """A student the optimizer can actually step, over a stack of hidden states."""

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False,
                return_dict=True, **_):
        first = self.dropout(self.embedding(input_ids))
        middle = self.linear(first)
        last = self.linear(middle)
        return SimpleNamespace(
            last_hidden_state=last,
            hidden_states=(first, middle, last) if output_hidden_states else None,
        )


class _BatchTokenizer(_CharTokenizer):
    """`_CharTokenizer`, but returning something with the `.to(device)` the
    distiller calls on it."""

    def __call__(self, texts, **kwargs):
        from transformers import BatchEncoding

        return BatchEncoding(super().__call__(texts, **kwargs))


def _run_distiller(tmp_path, monkeypatch, **config_kwargs) -> KnowledgeDistiller:
    corpus = tmp_path / "train.csv"
    pd.DataFrame(
        {
            # The varying part goes first: max_length truncates, and a corpus whose
            # rows share a long prefix would be 64 copies of one sentence by the
            # time the teacher sees it.
            "premise": [f"row {index} of the training corpus" for index in range(64)],
            "hypothesis": [f"pair {index} of the same corpus" for index in range(64)],
        }
    ).to_csv(corpus, index=False)

    config = GeoODEConfig(
        train_data_path=str(corpus),
        cache_dir=str(tmp_path / "cache"),
        save_dir=str(tmp_path / "ckpt"),
        epochs=1,
        batch_size=8,
        max_length=16,
        num_workers=0,
        gauge_align=False,
        **config_kwargs,
    )

    def stub_models(self):
        self.tok_student = _BatchTokenizer()
        self.tok_teacher = _BatchTokenizer()
        self.model_student = _StubStudent(dim=8)
        self.model_teacher = _StubTeacher(dim=16)
        self.model_student.to(self.device_s)
        self.model_teacher.to(self.device_t)

    monkeypatch.setattr(KnowledgeDistiller, "setup_models", stub_models)
    return KnowledgeDistiller(config)


def test_a_geoode_epoch_runs_end_to_end_through_the_new_wiring(tmp_path, monkeypatch):
    """Caching, the projection, the trimmed collate, the fused views and the event
    timer are separately covered above; this is the one test that runs them in the
    order a real run does."""
    distiller = _run_distiller(tmp_path, monkeypatch)

    before = distiller.model_student.linear.weight.detach().clone()
    loss = distiller.train_epoch(0)

    assert loss > 0.0 and torch.isfinite(torch.tensor(loss))
    # The student actually moved.
    assert not torch.equal(before, distiller.model_student.linear.weight.detach())
    # Every step got a duration, and they came from the event timer, not from a
    # placeholder left behind when the record was buffered.
    metrics = distiller.last_epoch_metrics
    assert metrics["mean_step_seconds"] > 0.0
    assert set(metrics) >= {"loss", "loss_end", "loss_ctr", "cos_final"}


def test_the_cache_is_written_once_and_reused_by_the_next_run(tmp_path, monkeypatch):
    """Re-encoding the corpus with a 4B-parameter teacher is the most expensive
    thing in the pipeline, so a second run of the same pair must not do it."""
    first = _run_distiller(tmp_path, monkeypatch)
    caches = list((tmp_path / "cache").glob("*.pt"))
    assert len(caches) == 1

    encoded = []
    original = _StubTeacher.forward

    def counting_forward(self, *args, **kwargs):
        encoded.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(_StubTeacher, "forward", counting_forward)
    second = _run_distiller(tmp_path, monkeypatch)

    assert encoded == []  # the teacher never ran the second time
    assert torch.equal(first.teacher_cls_all, second.teacher_cls_all)


def test_the_dropout_run_never_tokenizes_the_second_sentence(tmp_path, monkeypatch):
    """The saving this is here to protect: with the default contrastive view the
    paired sentence is not part of the objective, so it should not reach the GPU."""
    distiller = _run_distiller(tmp_path, monkeypatch)

    batch = next(iter(distiller.train_loader))

    assert "input_ids1_stu" in batch
    assert "input_ids2_stu" not in batch
    assert not any(key.startswith("special_tokens_mask") for key in batch)


def test_the_pair_view_still_gets_its_second_sentence(tmp_path, monkeypatch):
    distiller = _run_distiller(tmp_path, monkeypatch, contrastive_view="pair")

    batch = next(iter(distiller.train_loader))

    assert "input_ids2_stu" in batch
    assert torch.isfinite(torch.tensor(distiller.train_epoch(0)))


def test_the_topology_arm_ships_a_diagram_instead_of_the_teacher_cache(
    tmp_path, monkeypatch
):
    """With the H0 term on, the batch carries B-1 death times rather than a
    [B, d_T] copy of the teacher cache, and the term is still computed."""
    distiller = _run_distiller(tmp_path, monkeypatch, lambda_topo=0.5)

    batch = next(iter(distiller.train_loader))
    assert "teacher_deaths" in batch
    assert "teacher_topo" not in batch
    assert batch["teacher_deaths"].shape == (batch["teacher_cls"].shape[0] - 1,)

    distiller.train_epoch(0)
    assert distiller.last_epoch_metrics["loss_topo"] > 0.0


def test_the_topology_arm_reads_the_teachers_own_dimension(tmp_path, monkeypatch):
    """The H0 term is the one signal P_T cannot colour, so its diagram has to come
    from the unprojected d_T cache -- not from the d_S targets."""
    distiller = _run_distiller(tmp_path, monkeypatch, lambda_topo=0.5)
    batch = next(iter(distiller.train_loader))

    rows = batch["teacher_cls"].shape[0]
    projected = h0_death_times(batch["teacher_cls"].float(), metric="chord", sort=True)

    assert batch["teacher_cls"].shape[1] == 8  # d_S, after P_T
    assert not torch.allclose(batch["teacher_deaths"], projected, atol=1e-5)
    assert batch["teacher_deaths"].shape == (rows - 1,)


def test_the_h1_arm_ships_a_diagram_too(tmp_path, monkeypatch):
    """With lambda_h1 on, the collate also reduces the teacher's cache to its H1
    diagram, and the epoch's L_topo carries both halves."""
    pytest.importorskip("gudhi")
    distiller = _run_distiller(
        tmp_path, monkeypatch, lambda_topo=0.5, lambda_h1=0.25
    )

    batch = next(iter(distiller.train_loader))
    assert "teacher_topo" not in batch
    assert batch["teacher_deaths"].shape == (batch["teacher_cls"].shape[0] - 1,)
    # An empty diagram is a legitimate outcome for a small batch, so only the shape
    # is pinned here; the term's own tests cover what it contains.
    assert batch["teacher_h1"].ndim == 2 and batch["teacher_h1"].shape[1] == 2

    distiller.train_epoch(0)
    metrics = distiller.last_epoch_metrics
    assert metrics["loss_topo"] > 0.0
    assert metrics["loss_topo"] == pytest.approx(
        metrics["loss_h0"] + 0.25 * metrics["loss_h1"], rel=1e-4
    )


def test_the_h1_arm_stays_off_by_default(tmp_path, monkeypatch):
    """The H0 arm must not start paying for the 2-skeleton it never asked for."""
    distiller = _run_distiller(tmp_path, monkeypatch, lambda_topo=0.5)
    assert "teacher_h1" not in next(iter(distiller.train_loader))


def test_turning_fusion_off_changes_nothing_but_the_trajectory(tmp_path, monkeypatch):
    """Both orders train; they simply consume the RNG differently."""
    other = tmp_path / "b"
    other.mkdir()
    fused = _run_distiller(tmp_path, monkeypatch, seed=7)
    unfused = _run_distiller(other, monkeypatch, seed=7, fused_views=False)

    assert torch.isfinite(torch.tensor(fused.train_epoch(0)))
    assert torch.isfinite(torch.tensor(unfused.train_epoch(0)))
    assert unfused.config.fused_views is False
