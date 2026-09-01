"""Teacher pooling dispatch, tagged teacher caches and the special-token flags.

These are the three places a change of teacher family (Qwen3-Embedding -> BGE-M3)
or of cache path can go wrong silently: the pooling has to follow the family for
the online methods too, a cache built for another teacher has to be refused, and
the tokenizer marker has to be settable from the CLI.
"""

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from config import CDMConfig, GeoODEConfig
from distiller import KnowledgeDistiller
from main import get_config, parse_args
from src.cache_teacher import (
    _length_sorted_batches,
    cache_filename,
    cache_teacher_embeddings,
    corpus_digest,
    load_cached_embeddings,
    save_cached_embeddings,
    validate_cached_embeddings,
)
from src.pooling import POOLING_METHODS, pool_sentence_embedding


def _config(*argv):
    original = sys.argv
    sys.argv = ["main.py", *argv]
    try:
        args = parse_args()
        return get_config(args.method, args)
    finally:
        sys.argv = original


# --- pooling -------------------------------------------------------------------


def _hidden_and_mask():
    torch.manual_seed(0)
    hidden = torch.randn(2, 4, 3)
    # Row 0 is full length, row 1 is right-padded to two tokens.
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    return hidden, mask


def test_pool_sentence_embedding_reads_each_family_at_its_position():
    hidden, mask = _hidden_and_mask()

    cls = pool_sentence_embedding(hidden, mask, "cls")
    last = pool_sentence_embedding(hidden, mask, "last_token")
    mean = pool_sentence_embedding(hidden, mask, "mean")

    assert torch.equal(cls, hidden[:, 0, :])
    assert torch.equal(last[0], hidden[0, 3]) and torch.equal(last[1], hidden[1, 1])
    assert torch.allclose(mean[1], hidden[1, :2].mean(dim=0))


def test_pool_sentence_embedding_rejects_unknown_methods():
    hidden, mask = _hidden_and_mask()
    with pytest.raises(ValueError, match="Unknown pooling method"):
        pool_sentence_embedding(hidden, mask, "first_token")
    assert set(POOLING_METHODS) == {"last_token", "mean", "cls"}


def test_teacher_pooling_flag_reaches_online_methods_too():
    # The online methods used to hardcode last-token pooling; a BGE-style
    # teacher needs the flag to land on their config as well.
    assert _config("--method", "cdm").pooling_method == "last_token"
    assert (
        _config("--method", "cdm", "--teacher_pooling", "cls").pooling_method == "cls"
    )
    assert (
        _config("--method", "stella", "--teacher_pooling", "mean").pooling_method
        == "mean"
    )


# --- tagged cache --------------------------------------------------------------


def _metadata(**overrides):
    metadata = {
        "teacher_model_name": "Qwen/Qwen3-Embedding-0.6B",
        "pooling_method": "last_token",
        "normalize": True,
        "train_data_path": "data/train_set/merged_3_data_5k_each.csv",
    }
    metadata.update(overrides)
    return metadata


def test_cache_round_trips_embeddings_and_metadata(tmp_path):
    path = tmp_path / "cache" / "teacher_train.pt"
    embeddings = torch.randn(5, 8)

    save_cached_embeddings(str(path), embeddings, _metadata())
    loaded, metadata = load_cached_embeddings(str(path))

    assert torch.equal(loaded, embeddings)
    assert metadata["teacher_model_name"] == "Qwen/Qwen3-Embedding-0.6B"
    assert metadata["pooling_method"] == "last_token"
    assert metadata["rows"] == 5 and metadata["dim"] == 8


def test_legacy_bare_tensor_cache_still_loads(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save(torch.zeros(3, 4), path)

    loaded, metadata = load_cached_embeddings(str(path))

    assert loaded.shape == (3, 4)
    assert metadata == {}


def test_validate_refuses_a_cache_built_for_another_teacher_of_the_same_width():
    # Qwen3-Embedding-0.6B and BGE-M3 are both 1024-d: only the metadata can tell
    # the two caches apart.
    embeddings = torch.zeros(10, 1024)
    with pytest.raises(
        ValueError, match="teacher_model_name: cache has 'Qwen/Qwen3-Embedding-0.6B'"
    ):
        validate_cached_embeddings(
            embeddings,
            _metadata(),
            "cache/teacher_train.pt",
            teacher_model_name="BAAI/bge-m3",
            pooling_method="cls",
            normalize=True,
            teacher_dim=1024,
            rows=10,
        )


def test_validate_refuses_a_pooling_or_shape_mismatch():
    embeddings = torch.zeros(10, 1024)
    with pytest.raises(ValueError, match="pooling_method: cache has 'last_token'"):
        validate_cached_embeddings(
            embeddings,
            _metadata(),
            "x.pt",
            teacher_model_name="Qwen/Qwen3-Embedding-0.6B",
            pooling_method="cls",
            normalize=True,
        )
    with pytest.raises(ValueError, match="embedding dim: cache has 1024"):
        validate_cached_embeddings(
            embeddings,
            _metadata(),
            "x.pt",
            teacher_model_name="Qwen/Qwen3-Embedding-0.6B",
            pooling_method="last_token",
            normalize=True,
            teacher_dim=2560,
        )
    with pytest.raises(ValueError, match="rows: cache has 10"):
        validate_cached_embeddings(
            embeddings,
            _metadata(),
            "x.pt",
            teacher_model_name="Qwen/Qwen3-Embedding-0.6B",
            pooling_method="last_token",
            normalize=True,
            rows=11,
        )


def test_validate_accepts_a_matching_cache_and_a_legacy_one_by_shape(capsys):
    embeddings = torch.zeros(10, 1024)
    kwargs = dict(
        teacher_model_name="Qwen/Qwen3-Embedding-0.6B",
        pooling_method="last_token",
        normalize=True,
        teacher_dim=1024,
        rows=10,
    )
    validate_cached_embeddings(embeddings, _metadata(), "x.pt", **kwargs)
    assert "WARN" not in capsys.readouterr().out

    validate_cached_embeddings(embeddings, {}, "x.pt", **kwargs)
    assert "carries no metadata" in capsys.readouterr().out


# --- special-token flags ---------------------------------------------------------


def test_special_token_flags_override_the_config():
    assert (
        _config("--method", "cdm").teacher_special_token
        == CDMConfig.teacher_special_token
    )
    config = _config(
        "--method",
        "cdm",
        "--teacher_special_token",
        "▁",
        "--student_special_token",
        "##",
    )
    assert config.teacher_special_token == "▁"
    assert config.student_special_token == "##"
    # The flags are defined on the base config, so every method accepts them.
    assert (
        _config(
            "--method", "geoode", "--teacher_special_token", "Ġ"
        ).teacher_special_token
        == "Ġ"
    )
    assert (
        _config("--method", "geoode").teacher_special_token
        == GeoODEConfig.teacher_special_token
    )


# --------------------------------------------------------------------------- #
# Reuse: one shared directory, a filename that carries the cache's identity
# --------------------------------------------------------------------------- #


def _corpus(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_the_cache_name_separates_runs_that_must_not_share_one(tmp_path):
    """Everything a cache's reusability depends on has to change the filename, or a
    shared directory turns into a pile of mutual refusals."""
    corpus = _corpus(tmp_path, "train.csv", "premise,hypothesis\na,b\n")
    base = dict(
        teacher_model_name="Qwen/Qwen3-Embedding-4B",
        pooling_method="last_token",
        train_data_path=str(corpus),
        max_length=256,
        normalize=True,
    )
    reference = cache_filename(**base)

    assert cache_filename(**base) == reference  # deterministic
    for field, value in [
        ("teacher_model_name", "BAAI/bge-m3"),
        ("pooling_method", "cls"),
        ("max_length", 128),
        ("normalize", False),
    ]:
        assert cache_filename(**{**base, field: value}) != reference, field
    # Readable enough to identify by eye in a shared directory listing.
    assert reference.startswith("qwen-qwen3-embedding-4b__train__last_token__")
    assert reference.endswith(".pt")


def test_a_corpus_rebuilt_in_place_gets_a_different_cache(tmp_path):
    """The failure a shared cache directory would otherwise introduce: a corpus
    regenerated to the same path with the same row count is a different corpus, and
    nothing but its contents can say so."""
    corpus = _corpus(tmp_path, "train_150k.csv", "premise,hypothesis\na,b\nc,d\n")
    before = cache_filename(
        teacher_model_name="t",
        pooling_method="cls",
        train_data_path=str(corpus),
        max_length=256,
        normalize=True,
    )
    corpus.write_text("premise,hypothesis\nx,y\nz,w\n", encoding="utf-8")
    after = cache_filename(
        teacher_model_name="t",
        pooling_method="cls",
        train_data_path=str(corpus),
        max_length=256,
        normalize=True,
    )

    assert before != after


def test_the_corpus_digest_reads_contents_not_names(tmp_path):
    first = _corpus(tmp_path, "a.csv", "same\n")
    second = _corpus(tmp_path, "b.csv", "same\n")

    assert corpus_digest(first) == corpus_digest(second)
    assert corpus_digest(first) != corpus_digest(_corpus(tmp_path, "c.csv", "other\n"))


def test_a_stale_corpus_is_refused_even_behind_an_explicit_cache_path():
    """--cache_path names one file forever, so the digest is the only thing that
    can notice the corpus underneath it changed."""
    with pytest.raises(ValueError, match="train_data_digest"):
        validate_cached_embeddings(
            torch.randn(4, 8),
            {
                "teacher_model_name": "t",
                "pooling_method": "cls",
                "normalize": True,
                "train_data_digest": "0123456789abcdef",
            },
            "cache.pt",
            teacher_model_name="t",
            pooling_method="cls",
            normalize=True,
            train_data_digest="fedcba9876543210",
        )


def test_an_older_cache_without_the_new_fields_still_loads():
    """Caches predating the digest carry fewer identity fields; a field only one
    side knows is not a clash, or every existing cache would be thrown away."""
    validate_cached_embeddings(
        torch.randn(4, 8),
        {"teacher_model_name": "t", "pooling_method": "cls", "normalize": True},
        "cache.pt",
        teacher_model_name="t",
        pooling_method="cls",
        normalize=True,
        max_length=256,
        train_data_digest="0123456789abcdef",
    )


def test_the_run_resolves_a_shared_directory_into_its_own_file(tmp_path):
    """cache_dir wins over cache_path, and an explicitly set cache_path still wins
    when no directory is given."""
    config = GeoODEConfig(
        cache_dir=str(tmp_path),
        cache_path="cache/teacher_train.pt",
        train_data_path="data/train_set/train_100k.csv",
        teacher_model_name="Qwen/Qwen3-Embedding-4B",
        pooling_method="last_token",
        max_length=256,
        normalize_cache=True,
    )
    stub = object.__new__(KnowledgeDistiller)
    stub.config = config
    digest = corpus_digest(config.train_data_path)

    resolved = stub._resolve_cache_path(digest)
    assert resolved.parent == tmp_path
    assert resolved.name == cache_filename(
        teacher_model_name=config.teacher_model_name,
        pooling_method=config.pooling_method,
        train_data_path=config.train_data_path,
        max_length=config.max_length,
        normalize=config.normalize_cache,
        train_data_digest=digest,
    )

    config.cache_dir = None
    assert stub._resolve_cache_path(digest) == Path("cache/teacher_train.pt")


def test_two_pairs_land_on_different_files_in_one_directory(tmp_path):
    """The point of the shared directory: the second pair misses instead of loading
    the first pair's cache and being refused."""
    stub = object.__new__(KnowledgeDistiller)
    paths = set()
    for teacher, pooling in [
        ("Qwen/Qwen3-Embedding-0.6B", "last_token"),
        ("BAAI/bge-m3", "cls"),
        ("Qwen/Qwen3-Embedding-4B", "last_token"),
    ]:
        stub.config = GeoODEConfig(
            cache_dir=str(tmp_path),
            train_data_path="data/train_set/train_100k.csv",
            teacher_model_name=teacher,
            pooling_method=pooling,
        )
        paths.add(stub._resolve_cache_path("digest"))
    assert len(paths) == 3


# --------------------------------------------------------------------------- #
# The caching pass itself: length-sorted batching, teacher-only tokenization
# --------------------------------------------------------------------------- #


class _EchoTeacher(torch.nn.Module):
    """A teacher whose embedding of a row is a function of that row's tokens only.

    Every real encoder has this property (padding is masked out), and it is the
    property the length sort relies on: it is what makes "which rows share a batch"
    a scheduling decision rather than a modelling one. Making it exact here lets
    the test assert equality instead of a tolerance.
    """

    def __init__(self, dim: int = 4):
        super().__init__()
        self.dim = dim
        self.config = SimpleNamespace(hidden_size=dim)
        self.seen_batch_widths: list[int] = []

    def forward(self, input_ids, attention_mask, return_dict=True, **_):
        self.seen_batch_widths.append(int(input_ids.shape[1]))
        rows, length = input_ids.shape
        hidden = torch.zeros(rows, length, self.dim)
        for row in range(rows):
            valid = input_ids[row][attention_mask[row].bool()]
            # A per-row summary that no amount of padding can change.
            hidden[row, :, 0] = float(valid.sum())
            hidden[row, :, 1] = float(len(valid))
            hidden[row, :, 2] = float(valid[0]) if len(valid) else 0.0
            hidden[row, :, 3] = float(valid[-1]) if len(valid) else 0.0
        return SimpleNamespace(last_hidden_state=hidden)


class _WordTokenizer:
    """Whitespace tokenizer with right padding, standing in for a fast tokenizer."""

    def __init__(self):
        self.calls = 0

    def __call__(
        self,
        texts,
        max_length=None,
        truncation=True,
        padding=True,
        return_tensors="pt",
        **_,
    ):
        self.calls += 1
        rows = [
            [len(word) for word in text.split()][:max_length] or [0] for text in texts
        ]
        width = max(len(row) for row in rows)
        input_ids = torch.zeros(len(rows), width, dtype=torch.long)
        attention_mask = torch.zeros(len(rows), width, dtype=torch.long)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row)
            attention_mask[index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _corpus_texts(count: int = 37) -> list[str]:
    """Deliberately unsorted lengths, the way a merged corpus arrives."""
    rng = random.Random(0)
    return [
        " ".join(
            f"w{rng.randrange(1, 9)}" * rng.randrange(1, 4)
            for _ in range(rng.randrange(1, 14))
        )
        for _ in range(count)
    ]


def test_length_sorted_batches_cover_every_row_exactly_once():
    texts = _corpus_texts()
    batches = _length_sorted_batches(texts, batch_size=8)

    flat = [index for batch in batches for index in batch]
    assert sorted(flat) == list(range(len(texts)))
    assert len(batches) == 5 and [len(b) for b in batches] == [8, 8, 8, 8, 5]


def test_length_sorted_batches_group_similar_lengths():
    """The point of the sort: a batch pays for its longest row on every row it
    holds, so the padded token count is what has to come down."""
    texts = _corpus_texts(200)
    lengths = [len(text) for text in texts]

    def padded_tokens(batches):
        return sum(max(lengths[i] for i in batch) * len(batch) for batch in batches)

    corpus_order = [list(range(s, min(s + 16, 200))) for s in range(0, 200, 16)]
    sorted_order = _length_sorted_batches(texts, batch_size=16)

    assert padded_tokens(sorted_order) < padded_tokens(corpus_order)
    # Within a batch the rows really are neighbours in length.
    for batch in sorted_order:
        assert sorted(lengths[i] for i in batch) == [lengths[i] for i in batch]


def test_cached_row_i_is_the_teacher_embedding_of_corpus_row_i():
    """The sort reorders the *work*, never the result: the cache is indexed by
    corpus row, because that is the contract the training dataset relies on."""
    texts = _corpus_texts()
    teacher, tokenizer = _EchoTeacher(), _WordTokenizer()

    cached = cache_teacher_embeddings(
        model_teacher=teacher,
        texts=texts,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        max_length=64,
        batch_size=8,
        pooling_method="cls",
        use_amp=False,
    )

    assert cached.shape == (len(texts), 4)
    for index, text in enumerate(texts):
        one = cache_teacher_embeddings(
            model_teacher=_EchoTeacher(),
            texts=[text],
            tokenizer=_WordTokenizer(),
            device=torch.device("cpu"),
            max_length=64,
            batch_size=8,
            pooling_method="cls",
            use_amp=False,
        )
        assert torch.equal(cached[index], one[0]), f"row {index}"


def test_the_cache_is_not_an_inference_tensor():
    """The targets are read by a loss that saves them for backward, and a tensor
    born inside inference mode refuses that -- silently until the first step."""
    cached = cache_teacher_embeddings(
        model_teacher=_EchoTeacher(),
        texts=_corpus_texts(9),
        tokenizer=_WordTokenizer(),
        device=torch.device("cpu"),
        max_length=64,
        batch_size=4,
        pooling_method="cls",
        use_amp=False,
    )

    assert not cached.is_inference()
    weight = torch.randn(4, 2, requires_grad=True)
    (cached @ weight).sum().backward()  # would raise on an inference tensor
    assert weight.grad is not None


def test_the_caching_pass_tokenizes_each_row_once_for_the_teacher_alone():
    """It used to go through the training collate, which encodes both sides of the
    pair for both models -- four encodings of two identical strings -- and then
    read one of them."""
    texts = _corpus_texts(24)
    tokenizer = _WordTokenizer()

    cache_teacher_embeddings(
        model_teacher=_EchoTeacher(),
        texts=texts,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        max_length=64,
        batch_size=8,
        pooling_method="cls",
        use_amp=False,
    )

    assert tokenizer.calls == 3  # one per batch, not four


def test_every_pooling_method_survives_the_reordering():
    for pooling in POOLING_METHODS:
        texts = _corpus_texts(20)
        cached = cache_teacher_embeddings(
            model_teacher=_EchoTeacher(),
            texts=texts,
            tokenizer=_WordTokenizer(),
            device=torch.device("cpu"),
            max_length=64,
            batch_size=6,
            pooling_method=pooling,
            use_amp=False,
        )
        for index in [0, 7, 19]:
            one = cache_teacher_embeddings(
                model_teacher=_EchoTeacher(),
                texts=[texts[index]],
                tokenizer=_WordTokenizer(),
                device=torch.device("cpu"),
                max_length=64,
                batch_size=6,
                pooling_method=pooling,
                use_amp=False,
            )
            assert torch.equal(cached[index], one[0]), f"{pooling} row {index}"


def test_an_existing_cache_short_circuits_the_teacher(tmp_path):
    path = tmp_path / "cache.pt"
    save_cached_embeddings(str(path), torch.arange(12.0).reshape(3, 4), _metadata())
    teacher = _EchoTeacher()

    cached = cache_teacher_embeddings(
        model_teacher=teacher,
        texts=["a", "b", "c"],
        tokenizer=_WordTokenizer(),
        device=torch.device("cpu"),
        max_length=64,
        batch_size=2,
        cache_path=str(path),
        use_amp=False,
    )

    assert torch.equal(cached, torch.arange(12.0).reshape(3, 4))
    assert teacher.seen_batch_widths == []  # the teacher never ran


def test_the_cache_texts_column_follows_the_task_shape():
    frame = pd.DataFrame({"premise": ["p"], "hypothesis": ["h"]})
    assert KnowledgeDistiller._cache_texts(frame) == ["p"]
    assert KnowledgeDistiller._cache_texts(
        pd.DataFrame({"sentence1": ["s1"], "sentence2": ["s2"]})
    ) == ["s1"]
    assert KnowledgeDistiller._cache_texts(pd.DataFrame({"text": ["t"]})) == ["t"]
    with pytest.raises(ValueError, match="'premise', 'sentence1' or 'text'"):
        KnowledgeDistiller._cache_texts(pd.DataFrame({"other": ["x"]}))


def test_the_caching_batch_is_sized_apart_from_the_training_batch():
    stub = object.__new__(KnowledgeDistiller)
    stub.config = GeoODEConfig(batch_size=32, cache_batch_size=256)
    assert stub._cache_batch_size() == 256
    # 0 is the escape hatch back to the training batch, for a small card.
    stub.config = GeoODEConfig(batch_size=32, cache_batch_size=0)
    assert stub._cache_batch_size() == 32
