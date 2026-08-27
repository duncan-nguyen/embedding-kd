"""Teacher pooling dispatch, tagged teacher caches and the special-token flags.

These are the three places a change of teacher family (Qwen3-Embedding -> BGE-M3)
or of cache path can go wrong silently: the pooling has to follow the family for
the online methods too, a cache built for another teacher has to be refused, and
the tokenizer marker has to be settable from the CLI.
"""

import sys

import pytest
import torch

from config import CDMConfig, GeoODEConfig
from main import get_config, parse_args
from src.cache_teacher import (
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
    assert _config("--method", "cdm", "--teacher_pooling", "cls").pooling_method == "cls"
    assert _config("--method", "stella", "--teacher_pooling", "mean").pooling_method == "mean"


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
    with pytest.raises(ValueError, match="teacher_model_name: cache has 'Qwen/Qwen3-Embedding-0.6B'"):
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
    assert _config("--method", "cdm").teacher_special_token == CDMConfig.teacher_special_token
    config = _config(
        "--method", "cdm", "--teacher_special_token", "▁", "--student_special_token", "##"
    )
    assert config.teacher_special_token == "▁"
    assert config.student_special_token == "##"
    # The flags are defined on the base config, so every method accepts them.
    assert _config("--method", "geoode", "--teacher_special_token", "Ġ").teacher_special_token == "Ġ"
    assert _config("--method", "geoode").teacher_special_token == GeoODEConfig.teacher_special_token
