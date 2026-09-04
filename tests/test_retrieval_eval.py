"""The retrieval scorer, checked against hand-computed BEIR numbers.

nDCG@10 is the headline number of the five retrieval benchmarks, so the gain and
discount conventions, the rank-0 cut and ArguAna's self-match exclusion are all
pinned here rather than trusted to read correctly.
"""

import math

import numpy as np
import pytest

from src.evaluation.retrieval import _query_metrics, _rank, load_benchmark


def test_perfect_ranking_scores_one():
    ranking = ["a", "b", "c"]
    metrics = _query_metrics(ranking, {"a": 1, "b": 1}, top_k=10)

    assert metrics["ndcg_at_10"] == pytest.approx(1.0)
    assert metrics["recall_at_10"] == pytest.approx(1.0)
    assert metrics["mrr_at_10"] == pytest.approx(1.0)


def test_discount_matches_pytrec_eval():
    """One relevant document at rank 3 -> DCG = 1/log2(4), IDCG = 1/log2(2)."""
    ranking = ["x", "y", "gold", "z"]
    metrics = _query_metrics(ranking, {"gold": 1}, top_k=10)

    assert metrics["ndcg_at_10"] == pytest.approx(1.0 / math.log2(4))
    assert metrics["mrr_at_10"] == pytest.approx(1 / 3)
    assert metrics["recall_at_10"] == pytest.approx(1.0)


def test_relevance_beyond_the_cut_is_not_credited():
    """SCIDOCS averages ~5 positives per query, so the @10 truncation is live."""
    ranking = [f"miss{index}" for index in range(10)] + ["gold"]
    metrics = _query_metrics(ranking, {"gold": 1}, top_k=10)

    assert metrics == {"ndcg_at_10": 0.0, "recall_at_10": 0.0, "mrr_at_10": 0.0}


def test_ideal_ranking_is_also_truncated_at_the_cut():
    """12 positives, 10 of them retrieved perfectly: IDCG is over 10, so nDCG is 1."""
    relevant = {f"gold{index}": 1 for index in range(12)}
    ranking = [f"gold{index}" for index in range(10)]
    metrics = _query_metrics(ranking, relevant, top_k=10)

    assert metrics["ndcg_at_10"] == pytest.approx(1.0)
    assert metrics["recall_at_10"] == pytest.approx(10 / 12)


def test_gold_document_absent_from_the_corpus_scores_zero():
    """ArguAna ships 5 such queries; BEIR keeps them and they can only score 0."""
    metrics = _query_metrics(["a", "b"], {"not-in-corpus": 1}, top_k=10)

    assert metrics["ndcg_at_10"] == 0.0


def test_graded_relevance_uses_exponential_gains():
    ranking = ["low", "high"]
    metrics = _query_metrics(ranking, {"high": 2, "low": 1}, top_k=10)

    dcg = (2**1 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3)
    idcg = (2**2 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3)
    assert metrics["ndcg_at_10"] == pytest.approx(dcg / idcg)


def test_a_query_never_retrieves_its_own_document():
    """ArguAna puts each argument in the pool it is searching; BEIR drops it."""
    corpus_ids = ["q1", "gold", "other"]
    documents = np.array(
        [[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]], dtype=np.float32
    )
    queries = np.array([[1.0, 0.0]], dtype=np.float32)

    ranking = _rank(queries, documents, ["q1"], corpus_ids, top_k=2)

    assert ranking == [["gold", "other"]]


def test_self_match_is_dropped_before_the_top_k_cut():
    """Dropping it after the cut would cost the query one real candidate."""
    corpus_ids = ["q1", "gold"]
    documents = np.array([[1.0, 0.0], [0.9, 0.44]], dtype=np.float32)
    queries = np.array([[1.0, 0.0]], dtype=np.float32)

    assert _rank(queries, documents, ["q1"], corpus_ids, top_k=1) == [["gold"]]


def test_ranking_is_by_cosine_not_dot_product():
    """A long document must not outrank a better-aligned short one on norm alone."""
    corpus_ids = ["long", "aligned"]
    documents = np.array([[8.0, 6.0], [1.0, 0.0]], dtype=np.float32)
    queries = np.array([[1.0, 0.0]], dtype=np.float32)

    assert _rank(queries, documents, ["q"], corpus_ids, top_k=1) == [["aligned"]]


@pytest.mark.parametrize(
    "name,documents,queries",
    [
        ("arguana", 8674, 1406),
        ("fiqa", 57638, 648),
        ("scidocs", 25657, 1000),
        ("scifact", 5183, 300),
        ("nfcorpus", 3633, 323),
    ],
)
def test_downloaded_benchmarks_match_beir(name, documents, queries):
    benchmark = pytest.importorskip("pandas") and load_benchmark(
        f"data/test_set/retrieval/{name}"
    )

    assert len(benchmark["documents"]) == documents
    assert len(benchmark["queries"]) == queries
    # Every query the scorer averages over must have judged positives.
    judged = sum(1 for qid in benchmark["query_ids"] if qid in benchmark["relevance"])
    assert judged == queries
