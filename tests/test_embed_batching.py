"""The batching layer under `_embed_texts`, which every probe shares.

Nothing here touches a model: the claim being pinned is that the ids and masks
handed to the encoder are the ones `tokenizer(..., padding=True)` used to build,
and that the batches they are grouped into cover every text exactly once while
staying inside the token budget. Those two together are what make the speedup a
pure rearrangement rather than a change in what gets scored.
"""

import numpy as np
import pytest
import torch

from src.evaluation import evaluation_automodel as ev


class FakeTokenizer:
    """Whitespace tokenizer with a BERT-shaped interface.

    A real tokenizer would make these tests a download and a second of work each;
    what is under test is the padding and grouping around it, not WordPiece.
    """

    def __init__(self, padding_side="right", pad_token_id=0):
        self.padding_side = padding_side
        self.pad_token_id = pad_token_id
        self.name_or_path = "fake"
        self.calls = 0

    def __call__(self, texts, truncation=True, max_length=None, **kwargs):
        self.calls += 1
        ids = [[len(word) for word in text.split()][:max_length] for text in texts]
        return {"input_ids": ids}


def _texts(lengths):
    return [
        " ".join("w" * (index % 9 + 1) for _ in range(n))
        for index, n in enumerate(lengths)
    ]


def test_batches_cover_every_text_exactly_once():
    lengths = np.array([5, 1, 300, 7, 512, 2, 64, 64], dtype=np.int64)

    batches = ev._plan_batches(lengths)

    covered = np.concatenate(batches)
    assert sorted(covered.tolist()) == list(range(lengths.size))


def test_batches_are_grouped_by_token_length():
    """Sorting on token length is the whole padding win; character order is not it."""
    lengths = np.array([500, 3, 499, 4, 501, 5], dtype=np.int64)

    batches = ev._plan_batches(lengths)

    visited = lengths[np.concatenate(batches)].tolist()
    assert visited == sorted(visited)


def test_a_batch_stays_inside_the_token_budget(monkeypatch):
    monkeypatch.setattr(ev, "EVAL_TOKEN_BUDGET", 1000)
    monkeypatch.setattr(ev, "_MAX_BATCH_SEQUENCES", 10_000)
    lengths = np.full(50, 300, dtype=np.int64)

    batches = ev._plan_batches(lengths)

    for batch in batches:
        assert batch.size * int(lengths[batch].max()) <= 1000
    assert all(batch.size == 3 for batch in batches[:-1])


def test_short_texts_get_more_sequences_than_the_reference_batch(monkeypatch):
    """The point of the budget: 10-token queries must not be forwarded 256 at a time."""
    monkeypatch.setattr(ev, "EVAL_TOKEN_BUDGET", 256 * 512)
    monkeypatch.setattr(ev, "_MAX_BATCH_SEQUENCES", 2048)
    lengths = np.full(4096, 10, dtype=np.int64)

    batches = ev._plan_batches(lengths)

    assert [batch.size for batch in batches] == [2048, 2048]


def test_one_sequence_wider_than_the_budget_still_gets_a_batch(monkeypatch):
    monkeypatch.setattr(ev, "EVAL_TOKEN_BUDGET", 100)
    lengths = np.array([512, 4], dtype=np.int64)

    batches = ev._plan_batches(lengths)

    assert sorted(np.concatenate(batches).tolist()) == [0, 1]
    assert all(batch.size >= 1 for batch in batches)


def test_zero_budget_restores_fixed_size_batching(monkeypatch):
    monkeypatch.setattr(ev, "EVAL_TOKEN_BUDGET", 0)
    monkeypatch.setattr(ev, "EVAL_BATCH_SIZE", 4)
    lengths = np.arange(10, dtype=np.int64)

    batches = ev._plan_batches(lengths)

    assert [batch.size for batch in batches] == [4, 4, 2]


@pytest.mark.parametrize("padding_side", ["right", "left"])
def test_padding_reproduces_the_tokenizer(padding_side):
    """`_pad_batch` replaces `padding=True`; it must produce the same tensors."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.padding_side = padding_side
    texts = [
        "a short one",
        "a considerably longer sentence than the first",
        "mid length here",
    ]

    tokenized = ev._tokenize_all(tokenizer, texts, 512)
    ids, mask = ev._pad_batch(
        tokenized, np.arange(3), tokenizer.pad_token_id, padding_side == "left"
    )
    expected = tokenizer(
        texts, truncation=True, padding=True, max_length=512, return_tensors="pt"
    )

    assert torch.equal(ids, expected["input_ids"])
    assert torch.equal(mask, expected["attention_mask"])


def test_padding_reproduces_the_tokenizer_out_of_order():
    """Batches are index lists in sorted order, not slices of the original list."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    texts = ["one", "a much longer sentence indeed", "two words"]
    indices = np.array([2, 0, 1])

    tokenized = ev._tokenize_all(tokenizer, texts, 512)
    ids, mask = ev._pad_batch(tokenized, indices, tokenizer.pad_token_id, False)
    expected = tokenizer(
        [texts[index] for index in indices],
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt",
    )

    assert torch.equal(ids, expected["input_ids"])
    assert torch.equal(mask, expected["attention_mask"])


def test_truncation_is_still_applied():
    tokenizer = FakeTokenizer()
    tokenized = ev._tokenize_all(tokenizer, [" ".join(["w"] * 100)], 16)

    assert tokenized.lengths.tolist() == [16]


def test_the_same_split_is_tokenised_once_per_run():
    """Retrieval re-embeds ~101k documents every epoch; the ids never change."""
    tokenizer = FakeTokenizer()
    texts = _texts([3, 4, 5])

    first = ev._tokenized(tokenizer, texts, 128)
    second = ev._tokenized(tokenizer, list(texts), 128)

    assert first is second
    assert tokenizer.calls == 1


def test_the_cache_keys_on_content_and_length():
    tokenizer = FakeTokenizer()
    texts = _texts([3, 4, 5])

    ev._tokenized(tokenizer, texts, 128)
    ev._tokenized(tokenizer, texts, 64)
    ev._tokenized(tokenizer, texts + ["another"], 128)

    assert tokenizer.calls == 3


def test_the_cache_does_not_leak_across_tokenizers():
    """Two tokenizers over the same texts must not share ids.

    Keying on `id(tokenizer)` passed this by luck and failed once a collected
    tokenizer's address was reused, which is why the cache holds weak keys.
    """
    texts = _texts([3, 4, 5])
    first = FakeTokenizer()
    ev._tokenized(first, texts, 128)
    del first

    second = FakeTokenizer()
    ev._tokenized(second, texts, 128)

    assert second.calls == 1


def test_the_cache_stops_growing_at_its_limit(monkeypatch):
    monkeypatch.setattr(ev, "_TOKEN_CACHE_LIMIT", 0)
    monkeypatch.setattr(ev, "_TOKEN_CACHE", {})
    monkeypatch.setattr(ev, "_TOKEN_CACHE_SIZE", 0)
    tokenizer = FakeTokenizer()
    texts = _texts([3, 4, 5])

    ev._tokenized(tokenizer, texts, 128)
    ev._tokenized(tokenizer, texts, 128)

    assert tokenizer.calls == 2


def test_token_order_pads_less_than_character_order():
    """The regression this guards: fiqa pads 1.7x on character order, 1.01x on token order."""
    import pandas as pd
    from transformers import AutoTokenizer

    frame = pd.read_csv(
        "data/test_set/retrieval/nfcorpus/corpus.csv", dtype=str
    ).fillna("")
    texts = (frame["title"] + " " + frame["text"]).str.strip().tolist()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenized = ev._tokenize_all(tokenizer, texts, 512)
    lengths = tokenized.lengths

    def padded(order):
        return sum(
            len(order[start : start + 256])
            * int(lengths[order[start : start + 256]].max())
            for start in range(0, len(order), 256)
        )

    by_character = np.argsort([len(text) for text in texts], kind="stable")
    real = int(lengths.sum())

    assert padded(np.concatenate(ev._plan_batches(lengths))) < padded(by_character)
    assert padded(np.concatenate(ev._plan_batches(lengths))) < 1.1 * real
