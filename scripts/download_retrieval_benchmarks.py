"""Fetch the ArguAna, FiQA-2018 and SCIDOCS test splits used for retrieval eval.

The three benchmarks are the retrieval side of the protocol: the student is
distilled on unlabelled text and then scored zero-shot on nDCG@10 over corpora
it has never seen. None of them contributes a single sentence to the training
corpus, so `scripts/build_train_corpus.py` checks its MS MARCO block against the
files written here before accepting a row.

The BEIR/MTEB copies on the Hub are plain JSONL, so they are downloaded directly
rather than through `datasets`: one HTTP GET per file, pinned to a commit sha so
a later re-run cannot silently pick up a different release. Each benchmark lands
as three CSVs under `data/test_set/retrieval/<name>/`:

    corpus.csv   _id, title, text     every document, the full search space
    queries.csv  _id, text            only the queries the test qrels judge
    qrels.csv    query-id, corpus-id, score

Row counts are asserted against the BEIR paper, so a mirror that has drifted
fails here instead of turning into an unexplained score change.

Usage:
    python scripts/download_retrieval_benchmarks.py
    python scripts/download_retrieval_benchmarks.py --only fiqa --force
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "data" / "test_set" / "retrieval"

# (hub repo, commit sha, corpus rows, test queries) -- the sha pins the release and
# the two counts are the BEIR paper's, so a drifted mirror is caught on download.
BENCHMARKS = {
    "arguana": ("mteb/arguana", "6c1bcf74b13dfd823aff056b79d4d93e702f19c7", 8674, 1406),
    "fiqa": ("mteb/fiqa", "5e59eeb3a7df6b85882112b747008547c21587ea", 57638, 648),
    "scidocs": ("mteb/scidocs", "490848228d0a9ca7a7244f5e77d8fe33e6df6974", 25657, 1000),
}

FILES = ("corpus.jsonl", "queries.jsonl", "qrels/test.jsonl")


def fetch_jsonl(repo: str, revision: str, name: str) -> list[dict]:
    """One JSONL file off the Hub, parsed into a list of records."""
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}"
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(request, timeout=300) as response:
        raw = response.read()
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def build(name: str) -> dict[str, pd.DataFrame]:
    repo, revision, expected_corpus, expected_queries = BENCHMARKS[name]
    print(f"[{name}] {repo}@{revision[:8]}")

    corpus_rows, query_rows, qrel_rows = (
        fetch_jsonl(repo, revision, file) for file in FILES
    )

    corpus = pd.DataFrame(
        {
            "_id": [str(row["_id"]) for row in corpus_rows],
            "title": [str(row.get("title") or "") for row in corpus_rows],
            "text": [str(row.get("text") or "") for row in corpus_rows],
        }
    )
    qrels = pd.DataFrame(
        {
            "query-id": [str(row["query-id"]) for row in qrel_rows],
            "corpus-id": [str(row["corpus-id"]) for row in qrel_rows],
            "score": [int(row["score"]) for row in qrel_rows],
        }
    )
    # queries.jsonl of these three holds every split's queries; the test protocol
    # only ranks the ones the test qrels judge.
    judged = set(qrels["query-id"])
    queries = pd.DataFrame(
        {
            "_id": [str(row["_id"]) for row in query_rows],
            "text": [str(row["text"]) for row in query_rows],
        }
    )
    queries = queries[queries["_id"].isin(judged)].reset_index(drop=True)

    missing = judged - set(queries["_id"])
    if missing:
        raise SystemExit(
            f"{name}: {len(missing)} judged query ids are absent from queries.jsonl"
        )
    # ArguAna ships 5 qrels whose gold document is not in its own corpus. BEIR keeps
    # them, so those queries score 0 and every published number includes that; the
    # rows stay here for the same reason.
    unknown = set(qrels["corpus-id"]) - set(corpus["_id"])
    if unknown:
        print(
            f"    {len(unknown)} judged documents are absent from the corpus; their "
            f"queries can only score 0 (kept, as BEIR does)"
        )
    if len(corpus) != expected_corpus or len(queries) != expected_queries:
        raise SystemExit(
            f"{name}: got {len(corpus)} documents / {len(queries)} test queries, "
            f"BEIR reports {expected_corpus} / {expected_queries}. Refusing to write "
            f"a benchmark that does not match the published task."
        )

    relevant = qrels[qrels["score"] > 0]
    print(
        f"    corpus {len(corpus)}  queries {len(queries)}  "
        f"qrels {len(qrels)} ({len(relevant)} positive, "
        f"{len(relevant) / len(queries):.2f} per query)"
    )
    # ArguAna ranks each argument against a pool that contains itself; BEIR drops
    # that hit before scoring, and `src/evaluation/retrieval.py` does the same.
    self_hits = len(set(queries["_id"]) & set(corpus["_id"]))
    if self_hits:
        print(f"    {self_hits} queries are themselves documents (excluded at rank time)")
    return {"corpus": corpus, "queries": queries, "qrels": qrels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=sorted(BENCHMARKS))
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--force", action="store_true", help="re-download a benchmark already on disk"
    )
    args = parser.parse_args()

    for name in args.only or sorted(BENCHMARKS):
        destination = args.out_dir / name
        if (destination / "corpus.csv").is_file() and not args.force:
            print(f"[{name}] {destination} exists; skipping (use --force)")
            continue
        frames = build(name)
        destination.mkdir(parents=True, exist_ok=True)
        for stem, frame in frames.items():
            frame.to_csv(destination / f"{stem}.csv", index=False)
        print(f"    wrote {destination.relative_to(BASE_DIR)}/{{corpus,queries,qrels}}.csv")


if __name__ == "__main__":
    sys.exit(main())
