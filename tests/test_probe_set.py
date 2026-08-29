"""Corpus deduplication and the fixed probe set (protocol §0.1–0.2)."""

import json

import numpy as np
import pandas as pd
import pytest

from src.probe_set import (
    build_probe_set,
    deduplicate_corpus,
    eval_pairs_in_probe,
    evaluation_sentences,
    normalize_text,
    probe_digest,
)


@pytest.fixture
def project(tmp_path):
    """A miniature repo layout: one task per family plus one retrieval benchmark."""
    for split in ("val_set", "test_set", "train_set"):
        (tmp_path / "data" / split).mkdir(parents=True)
    pd.DataFrame(
        {"text": ["I feel great", "so sad today"], "label": [1, 0], "label_text": ["joy", "sadness"]}
    ).to_csv(tmp_path / "data" / "test_set" / "emotion_test.csv", index=False)
    pd.DataFrame(
        {"sentence1": ["a cat sleeps", "a dog runs"], "sentence2": ["a cat is asleep", "a bird flies"],
         "score": [4.5, 0.5]}
    ).to_csv(tmp_path / "data" / "test_set" / "stsb_test.csv", index=False)
    pd.DataFrame(
        {"sentence1": ["a cat sleeps"], "sentence2": ["an old cat"], "label": [0], "idx": [0]}
    ).to_csv(tmp_path / "data" / "test_set" / "mrpc_test.csv", index=False)
    pd.DataFrame(
        {"text": ["validation only sentence"], "label": [0], "label_text": ["x"]}
    ).to_csv(tmp_path / "data" / "val_set" / "emotion_validation.csv", index=False)
    retrieval = tmp_path / "data" / "test_set" / "retrieval" / "arguana"
    retrieval.mkdir(parents=True)
    pd.DataFrame({"_id": ["q1"], "text": ["should we do it"]}).to_csv(retrieval / "queries.csv", index=False)
    pd.DataFrame(
        {"_id": ["d1", "d2", "d3"], "title": ["", "T", ""], "text": ["doc one", "doc two", "doc three"]}
    ).to_csv(retrieval / "corpus.csv", index=False)
    pd.DataFrame({"query-id": ["q1"], "corpus-id": ["d1"], "score": [1]}).to_csv(retrieval / "qrels.csv", index=False)

    corpus = pd.DataFrame(
        {
            "text": ["I FEEL   great", "fresh corpus line", "a dog runs", "another fresh line",
                     "Should we do it", "fresh corpus line"],
            "source": ["EMOTION", "TWEET", "STSB", "TWEET", "MSMARCO", "TWEET"],
        }
    )
    corpus.to_csv(tmp_path / "data" / "train_set" / "train_tiny.csv")  # with the index column, like the real files
    return tmp_path


def test_normalisation_matches_the_corpus_builder():
    assert normalize_text("  I FEEL\tgreat \n") == "i feel great"


def test_evaluation_sentences_cover_every_column_and_the_queries(project):
    frame = evaluation_sentences(project)

    assert set(frame["task"]) == {"emotion_test", "stsb_test", "mrpc_test", "emotion_validation", "arguana_queries"}
    assert set(frame["column"]) == {"text", "sentence1", "sentence2"}
    assert (frame[frame["task"] == "emotion_test"]["label"] == "joy").any()
    # A sentence in two tasks appears once per occurrence: callers dedupe on text.
    assert (frame["text"] == "a cat sleeps").sum() == 2
    assert not (evaluation_sentences(project, include_retrieval_queries=False)["task"] == "arguana_queries").any()


def test_deduplication_removes_exactly_the_leaked_rows_and_writes_a_manifest(project):
    source = project / "data" / "train_set" / "train_tiny.csv"
    out = project / "data" / "train_set" / "train_tiny_dedup.csv"

    manifest = deduplicate_corpus(source, out, project)
    kept = pd.read_csv(out)

    # "I FEEL   great" (emotion, up to case/space), "a dog runs" (stsb) and
    # "Should we do it" (retrieval query) go; within-corpus duplicates stay.
    assert kept["text"].tolist() == ["fresh corpus line", "another fresh line", "fresh corpus line"]
    assert list(kept.columns) == ["text", "source"]
    assert manifest["rows_before"] == 6 and manifest["rows_after"] == 3 and manifest["rows_removed"] == 3
    assert manifest["removed_by_source"] == {"EMOTION": 1, "MSMARCO": 1, "STSB": 1}
    assert manifest["removed_by_task"] == {"arguana_queries": 1, "emotion_test": 1, "stsb_test": 1}
    assert manifest["source_digest"] != manifest["output_digest"]
    written = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert written == manifest


def test_probe_set_is_unique_ordered_seeded_and_flags_a_core(project):
    corpus = project / "data" / "train_set" / "train_tiny.csv"
    probe = build_probe_set(project, corpus, n_corpus=3, n_docs_per_retrieval=2, core_eval=2, core_retrieval=2, seed=1)
    again = build_probe_set(project, corpus, n_corpus=3, n_docs_per_retrieval=2, core_eval=2, core_retrieval=2, seed=1)
    other = build_probe_set(project, corpus, n_corpus=3, n_docs_per_retrieval=2, core_eval=2, core_retrieval=2, seed=2)

    assert probe["text"].is_unique
    assert probe["probe_id"].tolist() == list(range(len(probe)))
    assert list(probe.columns) == ["probe_id", "text", "group", "task", "label", "core"]
    assert probe["group"].tolist() == sorted(probe["group"].tolist(), key=["corpus", "eval", "retrieval_query", "retrieval_doc"].index)
    assert (probe["group"] == "corpus").sum() == 3
    assert (probe["group"] == "retrieval_doc").sum() == 2
    # Test split only by default: the validation-only sentence is not a probe row.
    assert "validation only sentence" not in set(probe["text"])
    # Every corpus row is core; the eval/retrieval core is a seeded sample.
    assert probe.loc[probe["group"] == "corpus", "core"].all()
    assert probe.loc[probe["group"] == "eval", "core"].sum() == 2
    assert probe.loc[probe["group"] != "corpus", "core"].sum() == 4
    assert probe.equals(again)
    assert probe_digest(probe) == probe_digest(again)
    assert not probe.equals(other)


def test_eval_pairs_map_onto_probe_rows_and_count_what_is_missing(project):
    corpus = project / "data" / "train_set" / "train_tiny.csv"
    probe = build_probe_set(project, corpus, n_corpus=1, n_docs_per_retrieval=1)
    pairs = eval_pairs_in_probe(
        project, probe,
        {"stsb": "data/test_set/stsb_test.csv", "mrpc": "data/test_set/mrpc_test.csv"},
    )

    assert pairs["stsb"]["kind"] == "sts" and pairs["mrpc"]["kind"] == "pair"
    assert pairs["stsb"]["missing"] == 0
    texts = probe["text"].tolist()
    assert [texts[i] for i in pairs["stsb"]["left"]] == ["a cat sleeps", "a dog runs"]
    assert np.allclose(pairs["stsb"]["target"], [4.5, 0.5])
    assert pairs["mrpc"]["target"].tolist() == [0.0]
