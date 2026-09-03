"""Prerequisites of the structural audit: a deduplicated corpus and a fixed probe set.

Two things every audit run depends on, and neither belongs to any single run:

* **Corpus deduplication** (protocol §0.2). ``train_100k`` contains sentences that
  also appear in the evaluation splits (most of SICK test, a fifth of STS-B), which
  inflates every arm equally but makes the absolute numbers unquotable.
  :func:`deduplicate_corpus` removes every corpus row whose normalised text is an
  evaluation sentence and writes a manifest saying what went.
* **The probe set** (protocol §0.1). A fixed, seeded list of sentences that every
  post-hoc quantity (ladder rungs, depth profiles, residual spectra, figures) is
  computed on: a corpus sample, every evaluation sentence, the retrieval queries
  and a fixed document sample per retrieval corpus. It is built once per
  (corpus, seed) and the same file is read by every arm, so arms are always
  compared on identical sentences.

Text normalisation is the one used by ``scripts/data/build_train_corpus.py`` (casefold
and whitespace collapse), so "appears in an evaluation split" means the same thing
here as it does for the corpus builder's exclusion set.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Column names that hold sentences in the evaluation CSVs. Classification tasks use
# ``text``; STS and pair tasks use ``sentence1``/``sentence2``.
EVAL_TEXT_COLUMNS = ("text", "sentence1", "sentence2")
# Columns a labelled subset can be coloured by (Figure 10). ``score`` is continuous
# and is kept for the STS ceiling.
EVAL_LABEL_COLUMNS = ("label_text", "label", "score")
RETRIEVAL_BENCHMARKS = ("arguana", "fiqa", "scidocs")


def normalize_text(text) -> str:
    """Casefold and collapse whitespace: the corpus builder's exact-match key."""
    return " ".join(str(text).strip().casefold().split())


def _eval_frames(project_dir: Path, splits: tuple[str, ...]) -> list[tuple[str, str, pd.DataFrame]]:
    frames = []
    for split in splits:
        split_dir = project_dir / "data" / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"missing evaluation split directory: {split_dir}")
        for path in sorted(split_dir.glob("*.csv")):
            frames.append((split, path.stem, pd.read_csv(path)))
    return frames


def evaluation_sentences(
    project_dir: str | os.PathLike[str],
    splits: tuple[str, ...] = ("val_set", "test_set"),
    include_retrieval_queries: bool = True,
) -> pd.DataFrame:
    """Every sentence of every evaluation file, one row per (task, column, row).

    Columns: ``text``, ``split``, ``task`` (file stem), ``column``, ``row`` (index in
    the file), ``label`` (the first available of :data:`EVAL_LABEL_COLUMNS`, as a
    string, or ``""``). A sentence that occurs in several files appears once per
    occurrence; callers that need unique texts deduplicate on ``text``.
    """
    project_dir = Path(project_dir)
    records = []
    for split, task, frame in _eval_frames(project_dir, splits):
        label_column = next((c for c in EVAL_LABEL_COLUMNS if c in frame.columns), None)
        labels = (
            frame[label_column].astype(str).tolist() if label_column else [""] * len(frame)
        )
        for column in EVAL_TEXT_COLUMNS:
            if column not in frame.columns:
                continue
            texts = frame[column].astype(str).tolist()
            records.extend(
                {
                    "text": text,
                    "split": split,
                    "task": task,
                    "column": column,
                    "row": row,
                    "label": label,
                }
                for row, (text, label) in enumerate(zip(texts, labels))
            )
    if include_retrieval_queries:
        retrieval_dir = project_dir / "data" / "test_set" / "retrieval"
        for name in RETRIEVAL_BENCHMARKS:
            path = retrieval_dir / name / "queries.csv"
            if not path.is_file():
                continue
            queries = pd.read_csv(path, dtype=str).fillna("")
            records.extend(
                {
                    "text": text,
                    "split": "test_set",
                    "task": f"{name}_queries",
                    "column": "text",
                    "row": row,
                    "label": "",
                }
                for row, text in enumerate(queries["text"].tolist())
            )
    return pd.DataFrame.from_records(
        records, columns=["text", "split", "task", "column", "row", "label"]
    )


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def deduplicate_corpus(
    corpus_path: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
    project_dir: str | os.PathLike[str],
    splits: tuple[str, ...] = ("val_set", "test_set"),
    manifest_path: str | os.PathLike[str] | None = None,
) -> dict:
    """Drop every corpus row whose normalised text is an evaluation sentence.

    The corpus keeps its columns (``text`` and ``source``) and row order; only the
    matching rows go. Duplicates *within* the corpus are left alone -- the protocol
    asks for evaluation leakage to be removed, nothing else, and changing the
    corpus composition beyond that would confound the comparison with older runs.

    Returns the manifest (also written next to the output, or to ``manifest_path``):
    the digests of both files, how many rows went, and a breakdown of the removed
    rows by corpus ``source`` and by the evaluation task that matched them.
    """
    corpus_path = Path(corpus_path)
    out_path = Path(out_path)
    project_dir = Path(project_dir)
    corpus = pd.read_csv(corpus_path)
    if "text" not in corpus.columns:
        raise ValueError(f"{corpus_path} has no 'text' column: {list(corpus.columns)}")
    # The corpus files carry a pandas index column; it is not data, so drop it.
    corpus = corpus.loc[:, [c for c in corpus.columns if not c.startswith("Unnamed")]]

    evaluation = evaluation_sentences(project_dir, splits=splits, include_retrieval_queries=True)
    evaluation = evaluation.assign(key=evaluation["text"].map(normalize_text))
    task_of_key: dict[str, str] = {}
    for key, task in zip(evaluation["key"], evaluation["task"]):
        task_of_key.setdefault(key, task)

    keys = corpus["text"].map(normalize_text)
    hit = keys.isin(task_of_key)
    removed = corpus[hit]
    kept = corpus[~hit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out_path, index=False)

    by_task = (
        keys[hit].map(task_of_key).value_counts().sort_index().to_dict() if hit.any() else {}
    )
    by_source = (
        removed["source"].value_counts().sort_index().to_dict()
        if "source" in removed.columns and hit.any()
        else {}
    )
    manifest = {
        "source_corpus": str(corpus_path),
        "source_digest": _file_digest(corpus_path),
        "output_corpus": str(out_path),
        "output_digest": _file_digest(out_path),
        "evaluation_splits": list(splits),
        "evaluation_sentences": int(evaluation["key"].nunique()),
        "rows_before": int(len(corpus)),
        "rows_after": int(len(kept)),
        "rows_removed": int(hit.sum()),
        "removed_by_source": {str(k): int(v) for k, v in by_source.items()},
        "removed_by_task": {str(k): int(v) for k, v in by_task.items()},
    }
    manifest_path = (
        Path(manifest_path) if manifest_path else out_path.with_suffix(".manifest.json")
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_probe_set(
    project_dir: str | os.PathLike[str],
    corpus_path: str | os.PathLike[str],
    n_corpus: int = 4096,
    n_docs_per_retrieval: int = 4096,
    eval_splits: tuple[str, ...] = ("test_set",),
    core_eval: int = 4096,
    core_retrieval: int = 1024,
    seed: int = 0,
) -> pd.DataFrame:
    """The fixed probe set of protocol §0.1: one row per unique sentence.

    Groups, in order:

    * ``corpus`` -- ``n_corpus`` sentences sampled (seeded) from the training corpus;
    * ``eval`` -- every sentence of every task in ``eval_splits`` (test by default:
      those are the sentences the reported numbers are computed on);
    * ``retrieval_query`` -- every query of the three retrieval benchmarks;
    * ``retrieval_doc`` -- a seeded sample of ``n_docs_per_retrieval`` documents per
      benchmark (the corpora are ~92k documents; a fixed sample is enough for the
      geometry and keeps the per-epoch dumps small).

    Sentences are unique by exact text: a sentence in two tasks keeps the ``task``
    and ``label`` of its first occurrence. The ``core`` flag marks the subset that
    is dumped at *every layer and every epoch* (all corpus rows plus seeded samples
    of ``core_eval`` evaluation sentences and ``core_retrieval`` retrieval rows); the
    final layer is dumped for every row. Row order is fixed, so ``probe_id`` is the
    row index and embeddings dumped by different arms line up by position.
    """
    project_dir = Path(project_dir)
    rng = np.random.default_rng(seed)

    corpus = pd.read_csv(corpus_path)
    corpus_texts = corpus["text"].astype(str).drop_duplicates()
    take = min(n_corpus, len(corpus_texts))
    picked = np.sort(rng.choice(len(corpus_texts), size=take, replace=False))
    rows = [
        {"text": text, "group": "corpus", "task": "corpus", "label": ""}
        for text in corpus_texts.iloc[picked].tolist()
    ]

    evaluation = evaluation_sentences(
        project_dir, splits=eval_splits, include_retrieval_queries=False
    )
    rows.extend(
        {"text": text, "group": "eval", "task": task, "label": label}
        for text, task, label in zip(evaluation["text"], evaluation["task"], evaluation["label"])
    )

    retrieval_dir = project_dir / "data" / "test_set" / "retrieval"
    for name in RETRIEVAL_BENCHMARKS:
        bench = retrieval_dir / name
        if not (bench / "queries.csv").is_file():
            continue
        queries = pd.read_csv(bench / "queries.csv", dtype=str).fillna("")
        rows.extend(
            {"text": text, "group": "retrieval_query", "task": f"{name}_queries", "label": ""}
            for text in queries["text"].tolist()
        )
        docs = pd.read_csv(bench / "corpus.csv", dtype=str).fillna("")
        documents = (docs["title"] + " " + docs["text"]).str.strip()
        take = min(n_docs_per_retrieval, len(documents))
        picked = np.sort(rng.choice(len(documents), size=take, replace=False))
        rows.extend(
            {"text": text, "group": "retrieval_doc", "task": f"{name}_corpus", "label": ""}
            for text in documents.iloc[picked].tolist()
        )

    probe = pd.DataFrame(rows, columns=["text", "group", "task", "label"])
    probe = probe[probe["text"].str.strip() != ""]
    probe = probe.drop_duplicates(subset="text", keep="first").reset_index(drop=True)

    core = np.zeros(len(probe), dtype=bool)
    core[(probe["group"] == "corpus").to_numpy()] = True
    for group, budget in (("eval", core_eval), ("retrieval_query", core_retrieval // 2),
                          ("retrieval_doc", core_retrieval - core_retrieval // 2)):
        candidates = np.flatnonzero((probe["group"] == group).to_numpy())
        if len(candidates) == 0 or budget <= 0:
            continue
        take = min(budget, len(candidates))
        core[rng.choice(candidates, size=take, replace=False)] = True
    probe.insert(0, "probe_id", np.arange(len(probe)))
    probe["core"] = core
    return probe


def probe_digest(probe: pd.DataFrame) -> str:
    """Fingerprint of the probe *texts in order*: what an embedding dump must match."""
    digest = hashlib.sha256()
    for text in probe["text"].astype(str):
        digest.update(text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def eval_pairs_in_probe(
    project_dir: str | os.PathLike[str],
    probe: pd.DataFrame,
    tasks: dict[str, str],
) -> dict[str, dict]:
    """Map sentence-pair tasks onto probe rows, for scoring embeddings post hoc.

    ``tasks`` maps a task name to its CSV path (relative to ``project_dir``). For
    each task the result holds ``left``/``right`` probe indices and ``target``
    (``score`` for STS, ``label`` for pair classification). Pairs whose sentences
    are not in the probe set are dropped and counted in ``missing``.
    """
    project_dir = Path(project_dir)
    index_of = {text: i for i, text in enumerate(probe["text"].astype(str))}
    result = {}
    for name, relative in tasks.items():
        frame = pd.read_csv(project_dir / relative)
        target_column = "score" if "score" in frame.columns else "label"
        left, right, target, missing = [], [], [], 0
        for s1, s2, y in zip(
            frame["sentence1"].astype(str), frame["sentence2"].astype(str), frame[target_column]
        ):
            if s1 in index_of and s2 in index_of:
                left.append(index_of[s1])
                right.append(index_of[s2])
                target.append(float(y))
            else:
                missing += 1
        result[name] = {
            "left": np.asarray(left, dtype=np.int64),
            "right": np.asarray(right, dtype=np.int64),
            "target": np.asarray(target, dtype=np.float64),
            "kind": "sts" if target_column == "score" else "pair",
            "missing": missing,
        }
    return result
