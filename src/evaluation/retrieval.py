"""Zero-shot dense retrieval on ArguAna, FiQA-2018 and SCIDOCS.

The classification/pair/STS probes all score a sentence pair or fit a linear head
on top of frozen features. Retrieval is the one family that asks the embedding
space to rank a whole corpus, so it is the part of the protocol that a distilled
student cannot pass by getting local neighbourhoods roughly right.

Protocol follows BEIR exactly, so the numbers are comparable to published ones:

* a document is `title + " " + text` (FiQA has no titles, SCIDOCS has one per row);
* queries and documents are embedded by the same encoder, no instruction prefix;
* ranking is cosine similarity over the full corpus, exhaustive, no ANN index;
* a document whose id equals the query id is dropped before ranking (ArguAna puts
  each argument in its own search space; BEIR excludes it for every task);
* nDCG@10 is the primary metric, with `2^rel - 1` gains and `log2(rank + 1)`
  discounts -- identical to `pytrec_eval`'s `ndcg_cut_10` on these binary qrels.

The benchmarks are not in git (~90 MB); `scripts/data/download_retrieval_benchmarks.py`
writes them. Embedding the three corpora is ~92k forward passes, far more than the
rest of the protocol combined, so this runs on the test split only.
"""

import functools
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .evaluation_automodel import BASE_DIR, _embed_texts

# Retrieval is the one family where truncation loses judged content: ArguAna
# queries average ~1.2k characters and its documents ~1k. BEIR encodes with the
# model's own window, so the default is the student's full 512 positions rather
# than the 256 used for training.
RETRIEVAL_MAX_LEN = int(os.environ.get("EVAL_RETRIEVAL_MAX_LEN", "512"))
# Similarities are formed one query block at a time: the full matrix for FiQA is
# 648 x 57638, which fits, but the block keeps GPU scoring bounded for any corpus.
QUERY_BLOCK = int(os.environ.get("EVAL_RETRIEVAL_QUERY_BLOCK", "128"))
TOP_K = 10


@functools.cache
def load_benchmark(directory: str) -> dict:
    """Read one benchmark's three CSVs into ids, texts and a qrels mapping.

    Cached per process for the same reason the other probes cache theirs: the
    files never change during a run and re-parsing a 57k-row corpus on every
    evaluation is pure repeated work.
    """
    root = BASE_DIR / directory if not os.path.isabs(directory) else Path(directory)
    missing = [
        name
        for name in ("corpus.csv", "queries.csv", "qrels.csv")
        if not (root / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Retrieval benchmark {root} is missing {', '.join(missing)}. "
            f"Run: python scripts/data/download_retrieval_benchmarks.py"
        )

    corpus = pd.read_csv(root / "corpus.csv", dtype=str).fillna("")
    queries = pd.read_csv(root / "queries.csv", dtype=str).fillna("")
    qrels = pd.read_csv(
        root / "qrels.csv", dtype={"query-id": str, "corpus-id": str, "score": int}
    )

    # BEIR's document text; `.strip()` matters for FiQA, whose titles are empty.
    documents = (corpus["title"] + " " + corpus["text"]).str.strip().tolist()

    relevance: dict[str, dict[str, int]] = {}
    for query_id, corpus_id, score in qrels.itertuples(index=False):
        if score > 0:
            relevance.setdefault(query_id, {})[corpus_id] = int(score)

    return {
        "name": root.name,
        "corpus_ids": corpus["_id"].tolist(),
        "documents": documents,
        "query_ids": queries["_id"].tolist(),
        "queries": queries["text"].tolist(),
        "relevance": relevance,
    }


def _normalize(matrix: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(matrix)
    return tensor / tensor.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _rank(query_embeddings, document_embeddings, query_ids, corpus_ids, top_k):
    """Top-k corpus ids per query by cosine similarity, self-matches removed."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    documents = _normalize(document_embeddings).to(device)
    queries = _normalize(query_embeddings).to(device)

    # A query id that is also a corpus id must lose its own document *before* the
    # top-k cut, otherwise it would displace a real candidate out of the list.
    position_of = {corpus_id: index for index, corpus_id in enumerate(corpus_ids)}
    self_positions = torch.tensor(
        [position_of.get(query_id, -1) for query_id in query_ids], device=device
    )

    width = min(top_k, len(corpus_ids))
    ranked = []
    for start in range(0, len(query_ids), QUERY_BLOCK):
        block = queries[start : start + QUERY_BLOCK]
        scores = block @ documents.T
        block_self = self_positions[start : start + QUERY_BLOCK]
        rows = torch.nonzero(block_self >= 0, as_tuple=True)[0]
        scores[rows, block_self[rows]] = float("-inf")
        indices = torch.topk(scores, width, dim=1).indices.cpu().numpy()
        ranked.extend([corpus_ids[index] for index in row] for row in indices)
    return ranked


def _query_metrics(ranking, relevant, top_k):
    """nDCG@k, Recall@k and MRR@k for one query, on trec_eval's definitions."""
    gains = [(2 ** relevant.get(doc, 0) - 1) for doc in ranking[:top_k]]
    discounts = [1.0 / np.log2(rank + 2) for rank in range(len(gains))]
    dcg = float(np.dot(gains, discounts))

    ideal = sorted(relevant.values(), reverse=True)[:top_k]
    idcg = float(
        sum((2**score - 1) / np.log2(rank + 2) for rank, score in enumerate(ideal))
    )

    hits = [doc for doc in ranking[:top_k] if doc in relevant]
    first = next(
        (rank + 1 for rank, doc in enumerate(ranking[:top_k]) if doc in relevant), None
    )
    return {
        "ndcg_at_10": dcg / idcg if idcg > 0 else 0.0,
        "recall_at_10": len(hits) / len(relevant) if relevant else 0.0,
        "mrr_at_10": 1.0 / first if first else 0.0,
    }


def eval_retrieval(model, tokenizer, directory, top_k=TOP_K):
    benchmark = load_benchmark(directory)
    name = benchmark["name"]
    print(
        f"{name}: {len(benchmark['queries'])} queries over "
        f"{len(benchmark['documents'])} documents"
    )

    document_embeddings = _embed_texts(
        model, tokenizer, benchmark["documents"], RETRIEVAL_MAX_LEN, desc=f"{name} corpus"
    )
    query_embeddings = _embed_texts(
        model, tokenizer, benchmark["queries"], RETRIEVAL_MAX_LEN, desc=f"{name} queries"
    )
    rankings = _rank(
        query_embeddings,
        document_embeddings,
        benchmark["query_ids"],
        benchmark["corpus_ids"],
        top_k,
    )

    # A query the qrels never judge has no ground truth to score against; BEIR's
    # loaders drop them upstream and the download script keeps only judged queries,
    # so this is a guard rather than a filter that normally fires.
    per_query = [
        _query_metrics(ranking, benchmark["relevance"][query_id], top_k)
        for query_id, ranking in zip(benchmark["query_ids"], rankings)
        if query_id in benchmark["relevance"]
    ]
    metrics = {
        key: float(np.mean([row[key] for row in per_query])) for key in per_query[0]
    }
    print(metrics)
    return metrics


def eval_retrieval_task(model, path_list, tokenizer):
    model.eval()
    print(" eval_retrieval_task")
    results = {}
    for directory in path_list:
        results[directory] = eval_retrieval(model, tokenizer, directory)
    model.train()
    return results


# Retrieval is scored on the test split only: there is no validation qrel set for
# these three, and the corpora are ~92k documents to embed.
test_retrieval_tasks = [
    "data/test_set/retrieval/arguana",
    "data/test_set/retrieval/fiqa",
    "data/test_set/retrieval/scidocs",
]
