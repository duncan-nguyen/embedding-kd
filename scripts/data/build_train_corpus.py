"""Build one rung of the unlabelled distillation ladder: 100k benchmark text + MS MARCO.

The 100k base is entirely sentence-level: short utterances, premise/hypothesis
pairs and STS sentences. Nothing in it looks like a search query or a web passage,
so a student distilled on it has never had to place a question near the paragraph
that answers it. The MS MARCO block adds exactly that shape while staying inside
the unlabelled objective:

* half the block's rows contribute their *query* and a disjoint half contribute one
  *passage* each, so no query in the corpus sits next to a passage that was
  retrieved for it;
* `is_selected`, `answers` and the qrels are never read. Nothing but raw text is
  written, so the training objective is unchanged;
* MS MARCO is a *training* source for the retrieval evaluation, never a test one:
  ArguAna, FiQA, SCIDOCS, SciFact and NFCorpus contribute nothing here, which
  keeps the retrieval numbers zero-shot cross-dataset rather than in-domain.

The base sample is identical at every size and the MS MARCO rows are accepted in a
fixed permutation order, so a larger rung *extends* a smaller one instead of
resampling it: the data-scaling ablation then varies one thing, corpus size, and
not which sentences are in the corpus.

Everything is pinned: the base sample, the MS MARCO shard and the passage picks
all come from one seed, and the Hub file is fetched at a fixed commit sha, so a
re-run reproduces the corpus byte for byte.

Every new text is checked, on normalised form, against the rows already in the
100k base, against the other new rows, and against the queries and corpora of all
five retrieval benchmarks plus every existing test and validation split. A hit is
dropped and counted.

Requires `python scripts/data/download_retrieval_benchmarks.py` first.

Usage:
    python scripts/data/build_train_corpus.py --total 150000
    python scripts/data/build_train_corpus.py --total 200000
    python scripts/data/build_train_corpus.py --total 100000        # base only, no MS MARCO
    python scripts/data/build_train_corpus.py --queries 40000 --passages 10000
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parents[2]
TRAIN_DIR = BASE_DIR / "data" / "train_set"
RETRIEVAL_DIR = BASE_DIR / "data" / "test_set" / "retrieval"
CACHE_DIR = BASE_DIR / "data" / "cache"

MSMARCO_REPO = "microsoft/ms_marco"
MSMARCO_SHA = "a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a"
# One shard of the v2.1 train split: 115,533 queries and ~1.1M passages, enough to
# draw 25k + 25k from disjoint halves without ever reusing a row. Pinning a single
# shard keeps the download at 240 MB and the sample reproducible.
MSMARCO_SHARD = "v2.1/train-00000-of-00007.parquet"

RETRIEVAL_BENCHMARKS = ("arguana", "fiqa", "scidocs", "scifact", "nfcorpus")
# Free text lives under different column names across the repo's splits.
TEXT_COLUMNS = ("text", "sentence1", "sentence2", "premise", "hypothesis")


def normalize(text: str) -> str:
    """The repo's leakage key: case- and whitespace-insensitive exact text.

    Same form `_validate_classification_pair` uses, so "no overlap" means the same
    thing here as it does in the evaluation code.
    """
    return " ".join(str(text).strip().casefold().split())


def download_shard() -> Path:
    destination = CACHE_DIR / "ms_marco_v2.1_train-00000-of-00007.parquet"
    if destination.is_file():
        return destination
    url = f"https://huggingface.co/datasets/{MSMARCO_REPO}/resolve/{MSMARCO_SHA}/{MSMARCO_SHARD}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MSMARCO_REPO}@{MSMARCO_SHA[:8]} {MSMARCO_SHARD} (~240 MB)")
    partial = destination.with_suffix(".part")
    urllib.request.urlretrieve(url, partial)
    partial.rename(destination)
    print(f"  cached at {destination.relative_to(BASE_DIR)}")
    return destination


def forbidden_texts() -> dict[str, set[str]]:
    """Normalised text of everything a training row must not be equal to.

    The retrieval benchmarks are the reason this script exists, so they are checked
    document-by-document; the sentence-level test and validation splits are checked
    too, because a MS MARCO passage that happens to reproduce an evaluation sentence
    would leak just as much.
    """
    missing = [
        name
        for name in RETRIEVAL_BENCHMARKS
        if not (RETRIEVAL_DIR / name / "corpus.csv").is_file()
    ]
    if missing:
        raise SystemExit(
            f"Retrieval benchmarks not on disk ({', '.join(missing)}). "
            f"Run: python scripts/data/download_retrieval_benchmarks.py"
        )

    groups: dict[str, set[str]] = {}
    for name in RETRIEVAL_BENCHMARKS:
        corpus = pd.read_csv(RETRIEVAL_DIR / name / "corpus.csv", dtype=str).fillna("")
        queries = pd.read_csv(RETRIEVAL_DIR / name / "queries.csv", dtype=str).fillna("")
        texts = set(corpus["text"].map(normalize))
        # The document the evaluator actually embeds is title + text; a training row
        # equal to either half is already too close, so both forms are excluded.
        texts |= set((corpus["title"] + " " + corpus["text"]).map(normalize))
        texts |= set(queries["text"].map(normalize))
        texts.discard("")
        groups[name] = texts

    evaluation: set[str] = set()
    for directory in ("test_set", "val_set"):
        for path in sorted((BASE_DIR / "data" / directory).glob("*.csv")):
            frame = pd.read_csv(path)
            for column in TEXT_COLUMNS:
                if column in frame.columns:
                    evaluation |= set(frame[column].astype(str).map(normalize))
    evaluation.discard("")
    groups["sentence_eval_splits"] = evaluation

    for name, texts in groups.items():
        print(f"  {name:<22} {len(texts):>7} distinct normalised texts")
    return groups


def sample_base(path: Path, size: int, seed: int) -> pd.DataFrame:
    """Exactly `size` rows of the existing corpus, drawn once and pinned by seed.

    The file holds 102,873 rows; the paper's data-scaling ablation compares a 100k
    corpus against this 150k one, so the base has to be exactly 100k rather than
    "whatever the merge happened to produce". Original row order is kept, which
    only matters for reading the file -- training shuffles.
    """
    frame = pd.read_csv(path)
    if len(frame) < size:
        raise SystemExit(f"{path.name} has {len(frame)} rows, need {size}")
    chosen = np.sort(np.random.default_rng(seed).permutation(len(frame))[:size])
    sampled = frame.iloc[chosen][["text", "source"]].reset_index(drop=True)
    print(
        f"Base: {size} of {len(frame)} rows from {path.name} "
        f"({sampled.text.map(normalize).nunique()} distinct normalised texts)"
    )
    return sampled


def collect_msmarco(shard, wanted_queries, wanted_passages, seed, blocked, taken):
    """Draw queries and passages from disjoint halves of one MS MARCO shard.

    `blocked` maps a label to a set of normalised texts that disqualify a row;
    `taken` is the running set of what has already been accepted, so the new block
    is deduplicated against the base corpus and against itself in one pass.
    """
    reader = pq.ParquetFile(shard)
    total = reader.metadata.num_rows
    rng = np.random.default_rng(seed)
    order = rng.permutation(total)
    # One draw per row id, so which passage a row contributes does not depend on the
    # order rows are visited in.
    passage_choice = rng.integers(0, 2**31 - 1, size=total)

    half = total // 2
    query_rank = np.empty(total, dtype=np.int64)
    query_rank[order[:half]] = np.arange(half)
    passage_rank = np.empty(total, dtype=np.int64)
    passage_rank[order[half:]] = np.arange(total - half)
    is_query_pool = np.zeros(total, dtype=bool)
    is_query_pool[order[:half]] = True
    print(
        f"MS MARCO: {total} rows in the shard, split {half} query pool / "
        f"{total - half} passage pool (disjoint)"
    )

    # (rank within its pool, text) -- collected shard-order, sorted into pool order
    # afterwards so acceptance follows the permutation and not the file layout.
    candidates = {"query": [], "passage": []}
    offset = 0
    for group in range(reader.metadata.num_row_groups):
        table = reader.read_row_group(group, columns=["query", "passages"])
        queries = table.column("query").to_pylist()
        passages = table.column("passages").combine_chunks().field("passage_text")
        for local, row in enumerate(range(offset, offset + len(queries))):
            if is_query_pool[row]:
                candidates["query"].append((query_rank[row], queries[local]))
            else:
                options = passages[local].as_py()
                if not options:
                    continue
                pick = options[passage_choice[row] % len(options)]
                candidates["passage"].append((passage_rank[row], pick))
        offset += len(queries)

    blocked_counts = {kind: dict.fromkeys(blocked, 0) for kind in candidates}
    kept: dict[str, list[str]] = {}
    duplicates = {}
    for kind, wanted in (("query", wanted_queries), ("passage", wanted_passages)):
        accepted, seen_before = [], 0
        for _, text in sorted(candidates[kind]):
            text = str(text).strip()
            key = normalize(text)
            if len(key) < 3:
                continue
            if key in taken:
                seen_before += 1
                continue
            hit = next((name for name, texts in blocked.items() if key in texts), None)
            if hit is not None:
                blocked_counts[kind][hit] += 1
                continue
            taken.add(key)
            accepted.append(text)
            if len(accepted) == wanted:
                break
        if len(accepted) < wanted:
            # One shard splits into ~57.7k rows per pool, so a single-shard build
            # tops out near 215k rows in total.
            raise SystemExit(
                f"Only {len(accepted)} usable {kind} rows in the shard, wanted "
                f"{wanted}. Point MSMARCO_SHARD at a second shard of the train "
                f"split (there are 7) to go past this size."
            )
        kept[kind] = accepted
        duplicates[kind] = seen_before
        dropped = ", ".join(
            f"{name}={count}" for name, count in blocked_counts[kind].items() if count
        )
        print(
            f"  {kind:<8} kept {len(accepted)}  duplicates skipped {seen_before}"
            + (f"  benchmark overlap dropped [{dropped}]" if dropped else "")
        )
    return kept, blocked_counts, duplicates


def report_base_overlap(base: pd.DataFrame, blocked: dict[str, set[str]]) -> dict:
    """The 100k base is fixed, so overlap there is reported rather than removed."""
    keys = set(base["text"].map(normalize))
    counts = {name: len(keys & texts) for name, texts in blocked.items()}
    retrieval = {k: v for k, v in counts.items() if k in RETRIEVAL_BENCHMARKS}
    if any(retrieval.values()):
        print(f"  WARNING: base corpus overlaps the retrieval benchmarks: {retrieval}")
    else:
        print("  base corpus has no exact overlap with any retrieval benchmark")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=TRAIN_DIR / "train_100k.csv")
    parser.add_argument(
        "--total",
        type=int,
        default=150_000,
        help="corpus size; the MS MARCO block is the remainder over --base-size, "
        "split evenly between queries and passages",
    )
    parser.add_argument("--base-size", type=int, default=100_000)
    parser.add_argument(
        "--queries", type=int, help="override the query half of the MS MARCO block"
    )
    parser.add_argument(
        "--passages", type=int, help="override the passage half of the MS MARCO block"
    )
    parser.add_argument(
        "--out", type=Path, help="default: data/train_set/train_<total>k.csv"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --queries/--passages win where given, so an uneven split stays expressible;
    # otherwise the block is whatever --total leaves over the base, halved.
    block = args.total - args.base_size
    if block < 0:
        raise SystemExit(f"--total {args.total} is below --base-size {args.base_size}")
    if args.queries is None:
        args.queries = block // 2
    if args.passages is None:
        args.passages = block - block // 2
    total = args.base_size + args.queries + args.passages
    if args.out is None:
        if total % 1000:
            raise SystemExit(f"{total} rows has no <n>k name; pass --out explicitly")
        args.out = TRAIN_DIR / f"train_{total // 1000}k.csv"
    # --total 100000 names the base file the script is reading from, and writing a
    # 100k sample over a 102,361-row input would destroy the source of every rung.
    if args.out.resolve() == args.base.resolve():
        raise SystemExit(
            f"--out and --base are both {args.out}; writing there would overwrite "
            f"the corpus this build samples from. Pass an explicit --out."
        )

    print("Loading exclusion sets")
    blocked = forbidden_texts()

    base = sample_base(args.base, args.base_size, args.seed)
    base_overlap = report_base_overlap(base, blocked)

    taken = set(base["text"].map(normalize))
    if args.queries or args.passages:
        shard = download_shard()
        kept, blocked_counts, duplicates = collect_msmarco(
            shard, args.queries, args.passages, args.seed, blocked, taken
        )
    else:
        # The base-only rung of the ladder: the same 100k rows, no download.
        print("MS MARCO block is empty; writing the base sample alone")
        kept = {"query": [], "passage": []}
        blocked_counts = {kind: dict.fromkeys(blocked, 0) for kind in kept}
        duplicates = dict.fromkeys(kept, 0)

    merged = pd.concat(
        [
            base,
            pd.DataFrame({"text": kept["query"], "source": "MSMARCO_QUERY"}),
            pd.DataFrame({"text": kept["passage"], "source": "MSMARCO_PASSAGE"}),
        ],
        ignore_index=True,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=True)

    manifest = {
        "output": str(args.out.relative_to(BASE_DIR)),
        "rows": len(merged),
        "seed": args.seed,
        "base": {
            "file": str(args.base.relative_to(BASE_DIR)),
            "rows": int(args.base_size),
            "overlap_with_excluded_sets": base_overlap,
        },
        "msmarco": {
            "repo": MSMARCO_REPO,
            "revision": MSMARCO_SHA,
            "shard": MSMARCO_SHARD,
            "queries": args.queries,
            "passages": args.passages,
            "duplicates_skipped": duplicates,
            "overlap_dropped": blocked_counts,
            "labels_used": None,
        },
        "sources": {str(k): int(v) for k, v in merged.source.value_counts().items()},
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nWrote {args.out.relative_to(BASE_DIR)}")
    print(f"  rows                     : {len(merged)}")
    print(f"  distinct normalised texts: {merged.text.map(normalize).nunique()}")
    print(merged.source.value_counts().to_string())
    print(f"  manifest: {manifest_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    sys.exit(main())
