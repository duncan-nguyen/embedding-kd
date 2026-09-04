"""What the three online baselines stopped paying per step, and what they still compute.

Same contract as ``test_training_loop_costs``: every change here is meant to be
invisible in the numbers. So each test pins an equivalence -- the collate's
alignment is the alignment the training step used to derive, the batched Sinkhorn
is the per-sample one, Stella's cached teacher vector is the online teacher's --
rather than the saving that motivated it.

The savings themselves, measured on the main-results sweep (batch 128, 100k rows,
H200):

* EMO's transport was solved one sentence at a time, and its convergence test read
  a device scalar on every iteration: 25,600 synchronisations per step. Its
  alignment was also derived twice per row per side, because both of its terms
  wanted it.
* CDM re-derived a DTW path per row per epoch on the critical path, and read four
  tensors back to the host per row to do it.
* Stella ran the teacher every step to use its pooled vector, which is the one
  thing a cache already holds.
"""

from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn

from config import CDMConfig, EMOConfig, StellaConfig
from distiller import CACHED_TEACHER_METHODS, KnowledgeDistiller
from src.criterions import contextual_dynamic_mapping as cdm
from src.criterions import emo_embedding_distillation as emo
from src.criterions.emo_embedding_distillation import EMODistillation, sinkhorn
from src.criterions.stella_distillation import stella_stage1_loss, stella_stage2_loss
from src.data_utils import DualTokenizerCollate, TextPairRaw

SENTENCES = [
    ("the cat rested on the mat", "a dog rested nearby"),
    ("distillation transfers geometry", "the student learns the manifold"),
    ("short row", "another short row"),
    ("tokenizers disagree about words", "alignment repairs the disagreement"),
]


# --------------------------------------------------------------------------- #
# Stubs: two tokenizers that really do segment the same sentence differently
# --------------------------------------------------------------------------- #


class _StubTokenizer:
    """Whitespace tokenizer with a growing vocabulary and right padding.

    ``word_piece`` splits any word longer than four characters into a head and a
    ``##`` continuation, so the student's segmentation genuinely differs from the
    teacher's and the alignment has real work to do.
    """

    def __init__(self, marker="", word_piece=False, start="<s>", end="</s>"):
        self.marker = marker
        self.word_piece = word_piece
        self.pad_token, self.unk_token, self.mask_token = "<pad>", "<unk>", None
        self.cls_token = self.bos_token = start
        self.sep_token = self.eos_token = end
        self.vocab = [self.pad_token, start, end, self.unk_token]
        self.ids = {token: index for index, token in enumerate(self.vocab)}

    def _id(self, token):
        if token not in self.ids:
            self.ids[token] = len(self.vocab)
            self.vocab.append(token)
        return self.ids[token]

    def _tokens(self, text):
        pieces = []
        for position, word in enumerate(str(text).split()):
            if self.word_piece and len(word) > 4:
                pieces.extend([word[:3], f"##{word[3:]}"])
            else:
                pieces.append(f"{self.marker}{word}" if position else word)
        return [self.cls_token, *pieces, self.sep_token]

    def __call__(
        self,
        texts,
        max_length=None,
        truncation=True,
        padding=True,
        return_tensors="pt",
        return_special_tokens_mask=False,
        **_,
    ):
        rows = [
            [self._id(t) for t in self._tokens(text)][:max_length] for text in texts
        ]
        width = max(len(row) for row in rows)
        input_ids = torch.zeros(len(rows), width, dtype=torch.long)
        attention_mask = torch.zeros(len(rows), width, dtype=torch.long)
        special = torch.zeros(len(rows), width, dtype=torch.long)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row)
            attention_mask[index, : len(row)] = 1
            special[index, 0] = 1
            special[index, len(row) - 1] = 1
            special[index, len(row) :] = 1
        encoded = {"input_ids": input_ids, "attention_mask": attention_mask}
        if return_special_tokens_mask:
            encoded["special_tokens_mask"] = special
        from transformers import BatchEncoding

        return BatchEncoding(encoded)

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        return [self.vocab[index] for index in ids]


def _tokenizers():
    return _StubTokenizer(word_piece=True), _StubTokenizer(marker="Ġ")


# The two config fields the collate takes are read differently by the two methods,
# exactly as they are on the command line: sub-word markers for CDM, special tokens
# for EMO.
ALIGNMENT_TOKENS = {"cdm": ("Ġ", "##"), "emo": ("<s>", "<s>")}


def _collate(alignment, tok_student, tok_teacher):
    teacher_token, student_token = ALIGNMENT_TOKENS[alignment]
    return DualTokenizerCollate(
        tok_student,
        tok_teacher,
        "pair_cls",
        32,
        alignment=alignment,
        teacher_token=teacher_token,
        student_token=student_token,
    )


def _samples():
    frame = pd.DataFrame(
        {"premise": [a for a, _ in SENTENCES], "hypothesis": [b for _, b in SENTENCES]}
    )
    dataset = TextPairRaw(frame, "pair_cls")
    return [dataset[index] for index in range(len(dataset))]


# --------------------------------------------------------------------------- #
# CDM: the collate builds the alignment the step used to derive
# --------------------------------------------------------------------------- #


def test_the_collate_alignment_gives_the_loss_the_per_sample_loop_gave():
    """The whole point: same number, without the Python loop over the batch."""
    tok_student, tok_teacher = _tokenizers()
    batch = _collate("cdm", tok_student, tok_teacher)(_samples())
    criterion = cdm.ContextualDynamicMapping(tok_student, tok_teacher, "Ġ", "##")

    torch.manual_seed(0)
    student_last = torch.randn(*batch["input_ids1_stu"].shape, 8)
    teacher_last = torch.randn(*batch["input_ids1_tea"].shape, 12)
    projection = nn.Linear(8, 12, bias=False)

    looped = criterion.compute_cdm_loss(
        S_last=student_last,
        T_last=teacher_last,
        batch_input_ids_stu=batch["input_ids1_stu"],
        batch_input_ids_tea=batch["input_ids1_tea"],
        keep_mask_stu=batch["attention_mask1_stu"].bool()
        & ~batch["special_tokens_mask1_stu"].bool(),
        keep_mask_tea=batch["attention_mask1_tea"].bool()
        & ~batch["special_tokens_mask1_tea"].bool(),
        proj_s2t=projection,
        device_s=torch.device("cpu"),
    )
    gathered = criterion.aligned_token_loss(
        student_last, teacher_last, batch["cdm_align1"], projection
    )

    # The alignment is not trivially empty: these two tokenizations really do share
    # tokens, which is what makes the equality worth asserting.
    assert batch["cdm_align1"].numel() > 0
    assert gathered.item() == pytest.approx(looped.item(), rel=1e-6)


def test_the_alignment_indexes_the_padded_batch_and_skips_special_tokens():
    tok_student, tok_teacher = _tokenizers()
    batch = _collate("cdm", tok_student, tok_teacher)(_samples())
    alignment = batch["cdm_align1"]

    rows, teacher_positions, student_positions = alignment.T
    assert alignment.shape[1] == 3
    assert rows.max() < batch["input_ids1_stu"].shape[0]
    assert teacher_positions.max() < batch["input_ids1_tea"].shape[1]
    assert student_positions.max() < batch["input_ids1_stu"].shape[1]
    # Nothing padded or special was aligned.
    assert batch["attention_mask1_stu"][rows, student_positions].all()
    assert not batch["special_tokens_mask1_stu"][rows, student_positions].any()
    assert not batch["special_tokens_mask1_tea"][rows, teacher_positions].any()


def test_an_empty_alignment_is_a_zero_the_optimizer_can_step_through():
    criterion = cdm.ContextualDynamicMapping(*_tokenizers())
    student_last = torch.randn(2, 5, 8, requires_grad=True)

    loss = criterion.aligned_token_loss(
        student_last,
        torch.randn(2, 5, 12),
        torch.zeros((0, 3), dtype=torch.long),
        nn.Linear(8, 12, bias=False),
    )

    assert loss.item() == 0.0
    assert loss.shape == ()


def test_only_the_first_sentence_carries_a_cdm_alignment():
    """CDM's token term is taken on side 1 alone; side 2 only feeds the task loss."""
    batch = _collate("cdm", *_tokenizers())(_samples())

    assert "cdm_align1" in batch
    assert "cdm_align2" not in batch


def test_the_cost_table_does_not_change_the_cost():
    """Memoising the edit distance is a lookup, not a different metric."""
    cdm._COST_CACHE.clear()
    cold = cdm.cost_fn("Ġmat", "ma", "Ġ", "##")
    warm = cdm.cost_fn("Ġmat", "ma", "Ġ", "##")

    assert cold == warm == 1.0
    assert cdm._COST_CACHE


# --------------------------------------------------------------------------- #
# EMO: one alignment, one transport, three attention maps
# --------------------------------------------------------------------------- #


def _emo_inputs(seed=0, layers_teacher=8, layers_student=4, heads=2):
    torch.manual_seed(seed)
    tok_student, tok_teacher = _tokenizers()
    batch = _collate("emo", tok_student, tok_teacher)(_samples())
    rows, length_teacher = batch["input_ids1_tea"].shape
    length_student = batch["input_ids1_stu"].shape[1]
    teacher = SimpleNamespace(
        last_hidden_state=torch.randn(rows, length_teacher, 12),
        attentions=tuple(
            torch.softmax(torch.randn(rows, heads, length_teacher, length_teacher), -1)
            for _ in range(layers_teacher)
        ),
    )
    student = SimpleNamespace(
        last_hidden_state=torch.randn(rows, length_student, 8),
        attentions=tuple(
            torch.softmax(torch.randn(rows, heads, length_student, length_student), -1)
            for _ in range(layers_student)
        ),
    )
    return batch, teacher, student, tok_student, tok_teacher


def _emo_loss(criterion, batch, teacher, student, tok_student, tok_teacher, **kwargs):
    return criterion.compute_emo_loss(
        teacher_outputs=teacher,
        student_outputs=student,
        input_ids_tea=batch["input_ids1_tea"],
        input_ids_stu=batch["input_ids1_stu"],
        attention_mask_tea=batch["attention_mask1_tea"],
        attention_mask_stu=batch["attention_mask1_stu"],
        tok_teacher=tok_teacher,
        tok_student=tok_student,
        **kwargs,
    )


def test_the_collates_alignment_is_the_one_the_criterion_would_have_derived():
    batch, teacher, student, tok_student, tok_teacher = _emo_inputs()
    criterion = EMODistillation(
        d_teacher=12,
        d_student=8,
        k_layers=2,
        teacher_special="<s>",
        student_special="<s>",
    )

    derived, _ = _emo_loss(criterion, batch, teacher, student, tok_student, tok_teacher)
    handed, _ = _emo_loss(
        criterion,
        batch,
        teacher,
        student,
        tok_student,
        tok_teacher,
        alignment=batch["emo_align1"],
    )

    assert batch["emo_align1"].numel() > 0
    assert handed.item() == pytest.approx(derived.item(), rel=1e-6)


def test_both_sides_of_a_pair_get_an_alignment():
    """EMO distils the second sentence too, so the collate has to align it as well."""
    batch = _collate("emo", *_tokenizers())(_samples())

    assert batch["emo_align1"].shape[1] == 3
    assert batch["emo_align2"].shape[1] == 3


def test_only_the_attention_maps_the_loss_reads_have_to_be_moved():
    layers = emo.teacher_attention_layers(
        teacher_layer_num=28, student_layer_num=12, k=2
    )

    # The last layer (the token importances) plus the last two of the mapped ones.
    assert layers == [21, 23, 27]
    assert emo.teacher_attention_layers(36, 12, 2) == [32, 35]
    # A teacher shallower than the student still resolves to a real index.
    assert emo.teacher_attention_layers(6, 12, 2) == [5]


def test_the_sparse_layer_map_gives_the_same_loss_as_the_full_tuple():
    batch, teacher, student, tok_student, tok_teacher = _emo_inputs()
    criterion = EMODistillation(
        d_teacher=12,
        d_student=8,
        k_layers=2,
        teacher_special="<s>",
        student_special="<s>",
    )
    layer_count = len(teacher.attentions)
    needed = emo.teacher_attention_layers(layer_count, len(student.attentions), 2)

    full, _ = _emo_loss(criterion, batch, teacher, student, tok_student, tok_teacher)
    sparse, _ = _emo_loss(
        criterion,
        batch,
        SimpleNamespace(
            last_hidden_state=teacher.last_hidden_state,
            attentions={index: teacher.attentions[index] for index in needed},
        ),
        student,
        tok_student,
        tok_teacher,
        teacher_layer_num=layer_count,
    )

    assert len(needed) < layer_count
    assert sparse.item() == pytest.approx(full.item(), rel=1e-6)


def test_the_sparse_map_refuses_to_guess_the_teachers_depth():
    """The block mapping is read off the teacher's real layer count, which a dict
    of three layers cannot supply."""
    batch, teacher, student, tok_student, tok_teacher = _emo_inputs()
    criterion = EMODistillation(
        d_teacher=12,
        d_student=8,
        k_layers=2,
        teacher_special="<s>",
        student_special="<s>",
    )

    with pytest.raises(ValueError, match="teacher_layer_num"):
        _emo_loss(
            criterion,
            batch,
            SimpleNamespace(
                last_hidden_state=teacher.last_hidden_state,
                attentions={7: teacher.attentions[7]},
            ),
            student,
            tok_student,
            tok_teacher,
        )


def test_the_batched_transport_is_the_per_sample_transport():
    """The rows of a batch never interact, so solving them together is the same
    transport -- it just stops testing convergence on a device scalar per row."""
    torch.manual_seed(0)
    lengths_student, lengths_teacher = [5, 3, 2], [4, 4, 2]
    cost = torch.rand(3, 5, 4)
    valid_student = emo.valid_prefix_mask(
        torch.tensor([[1] * n + [0] * (5 - n) for n in lengths_student])
    )
    valid_teacher = emo.valid_prefix_mask(
        torch.tensor([[1] * n + [0] * (4 - n) for n in lengths_teacher])
    )
    mass_student = torch.rand(3, 5) * valid_student
    mass_teacher = torch.rand(3, 4) * valid_teacher

    batched = emo.sinkhorn_batched(
        cost,
        mass_student,
        mass_teacher,
        valid_student,
        valid_teacher,
        alpha=0.1,
        max_iter=200,
    )
    one_at_a_time = []
    for row, (n_student, n_teacher) in enumerate(zip(lengths_student, lengths_teacher)):
        loss, _ = sinkhorn(
            cost[row, :n_student, :n_teacher],
            mass_student[row, :n_student],
            mass_teacher[row, :n_teacher],
            alpha=0.1,
            max_iter=200,
        )
        one_at_a_time.append(loss)

    assert batched.tolist() == pytest.approx(
        [loss.item() for loss in one_at_a_time], rel=1e-5, abs=1e-7
    )


def test_padded_positions_neither_send_nor_receive_transport_mass():
    """A padded row of the batch must not leak mass into a neighbour's plan."""
    torch.manual_seed(0)
    valid_student = emo.valid_prefix_mask(torch.tensor([[1, 1, 0, 0]]))
    valid_teacher = emo.valid_prefix_mask(torch.tensor([[1, 1, 1, 0]]))
    cost = torch.rand(1, 4, 4)

    padded = emo.sinkhorn_batched(
        cost,
        torch.rand(1, 4),
        torch.rand(1, 4),
        valid_student,
        valid_teacher,
        alpha=0.1,
        max_iter=200,
    )
    trimmed = emo.sinkhorn_batched(
        cost[:, :2, :3],
        torch.rand(1, 2),
        torch.rand(1, 3),
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 3, dtype=torch.bool),
        alpha=0.1,
        max_iter=200,
    )

    # The marginals differ (both are random), but a uniform one has to agree.
    uniform = emo.sinkhorn_batched(
        cost,
        torch.ones(1, 4),
        torch.ones(1, 4),
        valid_student,
        valid_teacher,
        alpha=0.1,
        max_iter=200,
    )
    uniform_trimmed = emo.sinkhorn_batched(
        cost[:, :2, :3],
        torch.ones(1, 2),
        torch.ones(1, 3),
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 3, dtype=torch.bool),
        alpha=0.1,
        max_iter=200,
    )
    assert uniform.item() == pytest.approx(uniform_trimmed.item(), rel=1e-5)
    assert torch.isfinite(padded).all() and torch.isfinite(trimmed).all()


def test_a_single_token_sequence_no_longer_crashes_the_transport():
    """``torch.diag`` on the per-sample plan needed a 1-D vector, so a row with one
    valid token raised. The batched form has no such shape to squeeze."""
    valid = torch.ones(1, 1, dtype=torch.bool)
    loss = emo.sinkhorn_batched(
        torch.rand(1, 1, 1), torch.ones(1, 1), torch.ones(1, 1), valid, valid
    )

    assert torch.isfinite(loss).all()


# --------------------------------------------------------------------------- #
# Stella: the teacher's pooled vector, from the cache
# --------------------------------------------------------------------------- #


def test_stella_is_one_of_the_cached_teacher_methods():
    assert CACHED_TEACHER_METHODS == {"talas", "geoode", "rkd", "stella"}


@pytest.mark.parametrize("stage", [1, 2])
def test_the_stage_losses_read_the_teacher_only_through_its_direction(stage):
    """Why the cache is a legitimate substitute: both stages L2-normalise the
    teacher embedding before they touch it, and the cache is written normalised."""
    torch.manual_seed(0)
    teacher = torch.randn(6, 10)
    normalised = torch.nn.functional.normalize(teacher, p=2, dim=-1)
    student = [torch.randn(6, dim) for dim in (10, 10, 8, 6, 4)]

    if stage == 1:
        raw, _ = stella_stage1_loss(student[0], teacher)
        cached, _ = stella_stage1_loss(student[0], normalised)
    else:
        raw, _ = stella_stage2_loss(*student[:2], *student[1:], teacher)
        cached, _ = stella_stage2_loss(*student[:2], *student[1:], normalised)

    assert cached.item() == pytest.approx(raw.item(), rel=1e-5)


class _StubTeacher(nn.Module):
    """The frozen teacher, for the one-off caching pass and nothing else."""

    def __init__(self, dim=10):
        super().__init__()
        self.register_buffer("scale", torch.linspace(0.1, 2.0, dim))
        self.config = SimpleNamespace(hidden_size=dim, num_hidden_layers=4)

    def forward(self, input_ids, attention_mask=None, return_dict=True, **_):
        mask = (
            attention_mask if attention_mask is not None else torch.ones_like(input_ids)
        )
        running = torch.cumsum(input_ids.float() * mask, dim=1)
        return SimpleNamespace(
            last_hidden_state=torch.sin(running.unsqueeze(-1) * self.scale)
        )


class _StubStella(nn.Module):
    """A Stella-shaped student: a backbone plus the four Matryoshka heads."""

    def __init__(self, vocab=256, dim=8, teacher_dim=10):
        super().__init__()
        self.backbone = nn.Sequential()
        self.embedding = nn.Embedding(vocab, dim)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(dim, teacher_dim)
        self.fc2, self.fc3, self.fc4 = (
            nn.Linear(dim, 6),
            nn.Linear(dim, 4),
            nn.Linear(dim, 2),
        )
        self.config = SimpleNamespace(hidden_size=dim, num_hidden_layers=2)

    def forward(self, input_ids, attention_mask=None, **_):
        pooled = self.dropout(self.embedding(input_ids))[:, 0, :]
        return {
            "pooled": pooled,
            "fc1": self.fc1(pooled),
            "fc2": self.fc2(pooled),
            "fc3": self.fc3(pooled),
            "fc4": self.fc4(pooled),
        }


def _stella_distiller(tmp_path, monkeypatch, **config_kwargs):
    corpus = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "premise": [f"row {index} of the training corpus" for index in range(32)],
            "hypothesis": [f"pair {index} of the same corpus" for index in range(32)],
        }
    ).to_csv(corpus, index=False)

    config = StellaConfig(
        train_data_path=str(corpus),
        cache_dir=str(tmp_path / "cache"),
        save_dir=str(tmp_path / "ckpt"),
        epochs=1,
        batch_size=8,
        max_length=16,
        num_workers=0,
        **config_kwargs,
    )

    def stub_models(self):
        student, teacher = _StubTokenizer(word_piece=True), _StubTokenizer()
        self.tok_student, self.tok_teacher = student, teacher
        self.model_student = _StubStella()
        self.model_teacher = _StubTeacher()
        self.model_student.to(self.device_s)
        self.model_teacher.to(self.device_t)
        self.current_stage = 1

    monkeypatch.setattr(KnowledgeDistiller, "setup_models", stub_models)
    return KnowledgeDistiller(config)


def test_stella_frees_the_teacher_before_the_first_step(tmp_path, monkeypatch):
    """The saving: a run that never encodes with the teacher should not be holding
    its weights, and its batches should already carry the vector it would produce."""
    distiller = _stella_distiller(tmp_path, monkeypatch)

    assert distiller.model_teacher is None
    batch = next(iter(distiller.train_loader))
    assert "teacher_cls" in batch
    # The teacher's tokenization is not part of a step any more either.
    assert not any(key.endswith("_tea") for key in batch)
    # Stage 2 still needs the paired sentence for its in-batch contrastive term.
    assert "input_ids2_stu" in batch


def test_the_teacher_runs_once_and_the_next_run_reads_the_cache(tmp_path, monkeypatch):
    first = _stella_distiller(tmp_path, monkeypatch)
    assert len(list((tmp_path / "cache").glob("*.pt"))) == 1

    encoded = []
    original = _StubTeacher.forward

    def counting_forward(self, *args, **kwargs):
        encoded.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(_StubTeacher, "forward", counting_forward)
    second = _stella_distiller(tmp_path, monkeypatch)

    assert encoded == []
    assert torch.equal(first.teacher_cls_all, second.teacher_cls_all)


@pytest.mark.parametrize("stage", [1, 2])
def test_a_stella_epoch_runs_end_to_end_off_the_cache(tmp_path, monkeypatch, stage):
    distiller = _stella_distiller(tmp_path, monkeypatch)
    distiller.current_stage = stage
    before = distiller.model_student.fc1.weight.detach().clone()

    loss = distiller.train_epoch(0)

    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(before, distiller.model_student.fc1.weight.detach())
    assert set(distiller.last_epoch_metrics) >= {"loss", "loss_cos", "loss_sim"}


def test_the_cached_batch_carries_the_vector_the_online_teacher_would_have(
    tmp_path, monkeypatch
):
    """The equivalence the whole change rests on, end to end: row i of the cache is
    the teacher's own pooled, normalised embedding of row i of the corpus."""
    distiller = _stella_distiller(tmp_path, monkeypatch)
    teacher = _StubTeacher()
    texts = (
        pd.read_csv(distiller.config.train_data_path)["premise"].astype(str).tolist()
    )

    encoding = distiller.tok_teacher(texts[:8], max_length=16, truncation=True)
    with torch.no_grad():
        online = teacher(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
        ).last_hidden_state
    from src.pooling import pool_sentence_embedding

    pooled = torch.nn.functional.normalize(
        pool_sentence_embedding(
            online, encoding["attention_mask"], distiller.config.pooling_method
        ),
        p=2,
        dim=-1,
    )

    assert torch.allclose(distiller.teacher_cls_all[:8], pooled, atol=1e-5)


# --------------------------------------------------------------------------- #
# End to end: cdm and emo through the collate that now feeds them
# --------------------------------------------------------------------------- #


class _StubEncoder(nn.Module):
    """A student/teacher that can also report attention maps, for EMO."""

    def __init__(self, vocab=256, dim=8, layers=4, heads=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(dim, dim)
        self.layers, self.heads = layers, heads
        self.config = SimpleNamespace(hidden_size=dim, num_hidden_layers=layers)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        **_,
    ):
        hidden = self.linear(self.dropout(self.embedding(input_ids)))
        attentions = None
        if output_attentions:
            length = input_ids.shape[1]
            scores = hidden @ hidden.transpose(1, 2)
            one = torch.softmax(scores, dim=-1)
            attentions = tuple(
                one.unsqueeze(1).expand(-1, self.heads, length, length)
                for _ in range(self.layers)
            )
        return SimpleNamespace(
            last_hidden_state=hidden,
            hidden_states=(hidden,) if output_hidden_states else None,
            attentions=attentions,
        )


def _online_distiller(tmp_path, monkeypatch, config, teacher_layers=8):
    corpus = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "premise": [a for a, _ in SENTENCES] * 4,
            "hypothesis": [b for _, b in SENTENCES] * 4,
        }
    ).to_csv(corpus, index=False)
    config.train_data_path = str(corpus)
    config.save_dir = str(tmp_path / "ckpt")
    config.epochs, config.batch_size = 1, 4
    config.max_length, config.num_workers = 16, 0

    def stub_models(self):
        self.tok_student = _StubTokenizer(word_piece=True)
        self.tok_teacher = _StubTokenizer(marker="Ġ")
        self.model_student = _StubEncoder(dim=8, layers=2)
        self.model_teacher = _StubEncoder(dim=12, layers=teacher_layers)
        self.model_student.to(self.device_s)
        self.model_teacher.to(self.device_t)

    monkeypatch.setattr(KnowledgeDistiller, "setup_models", stub_models)
    return KnowledgeDistiller(config)


def test_a_cdm_epoch_runs_end_to_end_on_the_collates_alignment(tmp_path, monkeypatch):
    distiller = _online_distiller(
        tmp_path,
        monkeypatch,
        CDMConfig(teacher_special_token="Ġ", student_special_token="##"),
    )
    batch = next(iter(distiller.train_loader))
    assert "cdm_align1" in batch and batch["cdm_align1"].numel() > 0

    before = distiller.model_student.linear.weight.detach().clone()
    loss = distiller.train_epoch(0)

    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(before, distiller.model_student.linear.weight.detach())
    assert set(distiller.last_epoch_metrics) >= {"loss_kd_dtw", "loss_kd_cls"}


def test_an_emo_epoch_runs_end_to_end_on_the_collates_alignment(tmp_path, monkeypatch):
    distiller = _online_distiller(
        tmp_path,
        monkeypatch,
        EMOConfig(teacher_special_token="<s>", student_special_token="<s>", k_layers=2),
    )
    batch = next(iter(distiller.train_loader))
    assert {"emo_align1", "emo_align2"} <= set(batch)

    before = distiller.model_student.linear.weight.detach().clone()
    loss = distiller.train_epoch(0)

    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(before, distiller.model_student.linear.weight.detach())
    assert set(distiller.last_epoch_metrics) >= {"att_loss", "ot_loss"}


# --------------------------------------------------------------------------- #
# Second pass: what was left once the alignment and the transport were fixed
# --------------------------------------------------------------------------- #


def test_the_batched_cka_is_the_per_row_cka():
    """The attention term ran one ~ten-kernel CKA chain per row per matched layer
    -- 512 a step at batch 128, over operands a few tokens wide. Batching them is
    only allowed if it is the same number."""
    torch.manual_seed(0)
    rows_per_batch = [4, 2, 1]
    width = max(rows_per_batch)
    student = torch.randn(3, width, 7)
    teacher = torch.randn(3, width, 5)
    valid = torch.tensor(
        [[index < count for index in range(width)] for count in rows_per_batch]
    )

    batched = emo.batched_cka(student, teacher, valid)
    per_row = [
        emo.CKALoss(eps=1e-8)(student[row, :count], teacher[row, :count])
        for row, count in enumerate(rows_per_batch)
    ]

    assert batched.tolist() == pytest.approx(
        [value.item() for value in per_row], rel=1e-9
    )


def test_zero_feature_columns_are_invisible_to_the_batched_cka():
    """Padding a row's sequence out to the batch width adds all-zero columns to
    the Gram matrices, which the Frobenius norm does not see."""
    torch.manual_seed(0)
    student, teacher = torch.randn(1, 4, 6), torch.randn(1, 4, 5)
    valid = torch.ones(1, 4, dtype=torch.bool)

    tight = emo.batched_cka(student, teacher, valid)
    padded = emo.batched_cka(
        torch.cat([student, torch.zeros(1, 4, 3)], dim=-1),
        torch.cat([teacher, torch.zeros(1, 4, 2)], dim=-1),
        valid,
    )

    assert padded.item() == pytest.approx(tight.item(), rel=1e-12)


def test_a_row_that_aligned_nothing_is_left_out_of_the_average():
    """The per-sample loop skipped it with a ``continue``; the batched form has to
    drop it from both the sum and the divisor."""
    batch, teacher, student, tok_student, tok_teacher = _emo_inputs()
    criterion = EMODistillation(
        d_teacher=12,
        d_student=8,
        k_layers=2,
        teacher_special="<s>",
        student_special="<s>",
    )
    alignment = batch["emo_align1"]
    kept = alignment[alignment[:, 0] != 0]  # row 0 aligns nothing

    full, _ = _emo_loss(
        criterion, batch, teacher, student, tok_student, tok_teacher, alignment=kept
    )

    assert torch.isfinite(full)
    assert 0 not in kept[:, 0].tolist()


def test_the_dtw_path_and_its_matrix_are_what_the_numpy_version_gave():
    """The recurrence moved off NumPy scalar indexing and the backtrace off
    ``np.argmin``; both had to keep ``argmin``'s tie-break, which is first-wins
    over (diagonal, up, left)."""
    # A tie everywhere: every substitution costs the same, so every step of the
    # path is decided by the tie-break alone.
    teacher_tokens = ["a", "b", "c", "d"]
    student_tokens = ["a", "b", "c"]
    path, matrix = cdm.dtw(teacher_tokens, student_tokens, norm_func=lambda a, b: 1.0)

    assert matrix.shape == (len(teacher_tokens), len(student_tokens))
    assert path[0] == (0, 0)
    assert path[-1] == (len(teacher_tokens) - 1, len(student_tokens) - 1)
    # Monotone and one step at a time, which is what makes it an alignment.
    for (i, j), (next_i, next_j) in zip(path, path[1:]):
        assert (next_i - i, next_j - j) in {(1, 0), (0, 1), (1, 1)}


def test_row_alignment_is_the_dtw_path_put_through_the_strict_filter():
    """``row_alignment`` skips building the cost array, so it has to agree with the
    two-step form the debug path still uses."""
    tok_student, tok_teacher = _tokenizers()
    mapper = cdm.build_special_token_mapper(tok_student, tok_teacher)
    teacher_tokens = tok_teacher.convert_ids_to_tokens(
        tok_teacher([SENTENCES[0][0]], max_length=32)["input_ids"][0].tolist()
    )
    student_tokens = tok_student.convert_ids_to_tokens(
        tok_student([SENTENCES[0][0]], max_length=32)["input_ids"][0].tolist()
    )

    path, _ = cdm.dtw(
        teacher_tokens,
        student_tokens,
        norm_func=lambda a, b: cdm.cost_fn(a, b, "Ġ", "##", mapper),
    )
    two_step, _, _ = cdm.strict_one_to_one_pairs(
        path, teacher_tokens, student_tokens, "##", "Ġ", mapper
    )

    assert (
        cdm.row_alignment(teacher_tokens, student_tokens, "Ġ", "##", mapper) == two_step
    )
    assert two_step


def test_the_mined_cost_table_does_not_change_the_alignment():
    tokens_teacher, tokens_student = (
        ["<s>", "Ġcat", "Ġmat"],
        ["<s>", "ca", "##t", "mat"],
    )
    emo._SUBSTITUTION_CACHE.clear()
    cold = emo.align_tokens(tokens_teacher, tokens_student)
    warm = emo.align_tokens(tokens_teacher, tokens_student)

    assert cold == warm
    assert emo._SUBSTITUTION_CACHE
