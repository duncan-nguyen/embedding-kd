import functools
import hashlib
import os
import warnings
import weakref
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Evaluation is forward-only, so the batch that fits is much larger than the
# training batch. Overridable because the probe train sets (up to 24k rows) are
# the dominant cost and their optimal batch depends on the GPU. It is now the
# reference size the token budget below is derived from rather than the batch
# every split is cut into.
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "256"))
# A batch also closes once its padded width would push it past this many tokens,
# so a split of short texts is not forwarded 256 sequences at a time with most of
# the GPU idle. The default is exactly the old worst case -- 256 sequences over
# the 512 positions a retrieval document may occupy -- so peak activation memory
# is unchanged while short batches grow to fill it. Set to 0 for the old
# fixed-size batching.
EVAL_TOKEN_BUDGET = int(os.environ.get("EVAL_TOKEN_BUDGET", str(EVAL_BATCH_SIZE * 512)))
# ...but not unboundedly: nfcorpus queries are ~10 tokens, and a batch of 13k of
# them would spend its time in per-sequence overhead (pooling, the device->host
# copy) rather than in the encoder. Derived, not a knob.
_MAX_BATCH_SEQUENCES = 8 * EVAL_BATCH_SIZE
# Padding a batch runs in a worker thread while the GPU consumes the previous
# one. Fast tokenizers release the GIL, so threads are enough here and avoid the
# per-DataLoader process fork that dominates on the small splits.
EVAL_PREFETCH = int(os.environ.get("EVAL_PREFETCH", "2"))
# The one tokenisation pass is chunked over threads for the same reason: the work
# is in Rust with the GIL released, so it scales even though every launcher sets
# TOKENIZERS_PARALLELISM=false (fiqa's 57.6k documents: 19.6s on one thread, 5.1s
# on four). Capped at 8 -- the curve is flat past that and eval shares the box.
_TOKENIZE_WORKERS = min(8, os.cpu_count() or 1)
_TOKENIZE_CHUNK = 512
# Cached ids are held as int32, so this caps the cache near 256 MB. The five
# retrieval corpora plus every probe split come to ~35M tokens, well inside it;
# the limit only exists so an unforeseen caller cannot grow it without bound.
_TOKEN_CACHE_LIMIT = 64_000_000


@functools.cache
def _read_eval_csv(file_path: str) -> pd.DataFrame:
    """Read an evaluation CSV once per process.

    Every split is re-read on every epoch's evaluation pass and the files never
    change during a run, so the parse is pure repeated work.
    """
    full_path = (
        BASE_DIR / file_path if not os.path.isabs(file_path) else Path(file_path)
    )
    return pd.read_csv(full_path)


def _column_as_list(frame: pd.DataFrame, column: str) -> list[str]:
    return frame[column].astype(str).tolist()


def _pooled_embedding(output):
    """Extract the sentence embedding from a student forward pass.

    Supports StellaModel (dict with 'pooled') and AutoModel (object or dict with
    last_hidden_state).
    """
    if isinstance(output, dict) and "pooled" in output:
        return output["pooled"]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0, :]
    return output["last_hidden_state"][:, 0, :]


class _Tokenized(NamedTuple):
    """One split's truncated ids, held flat rather than as a list of lists.

    `flat[offsets[i]:offsets[i + 1]]` is sequence `i`. A ragged buffer instead of
    57k little arrays keeps the per-object overhead off the cache and makes the
    per-batch pad a run of memcpys.
    """

    flat: np.ndarray
    offsets: np.ndarray
    lengths: np.ndarray


def _texts_digest(texts) -> str:
    """Content hash of a text list, for keying the tokenisation cache.

    Identity would be cheaper but does not hold: only `load_benchmark` hands back
    the same list object every epoch, while the probe readers rebuild theirs from
    a cached frame. Hashing fiqa's 44 MB of documents costs 59 ms against the
    19.6s the tokenisation it saves would take.
    """
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(len(texts)).encode())
    for text in texts:
        digest.update(text.encode("utf-8", "surrogatepass"))
        digest.update(b"\0")
    return digest.hexdigest()


# Keyed on the tokenizer object rather than on its name: two tokenizers can share
# a name_or_path, and keying on `id()` would let a collected tokenizer's address
# be recycled onto a live entry. A weak key also lets the ids die with it.
_TOKEN_CACHE: "weakref.WeakKeyDictionary[object, dict[tuple, _Tokenized]]" = (
    weakref.WeakKeyDictionary()
)
_TOKEN_CACHE_SIZE = 0


def _tokenize_all(tokenizer, texts, max_len) -> _Tokenized:
    chunks = [
        texts[start : start + _TOKENIZE_CHUNK]
        for start in range(0, len(texts), _TOKENIZE_CHUNK)
    ]

    def encode(chunk):
        return tokenizer(chunk, truncation=True, max_length=max_len)["input_ids"]

    with ThreadPoolExecutor(max_workers=_TOKENIZE_WORKERS) as pool:
        encoded = [ids for chunk in pool.map(encode, chunks) for ids in chunk]

    lengths = np.fromiter(
        (len(ids) for ids in encoded), dtype=np.int64, count=len(encoded)
    )
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    flat = np.empty(int(offsets[-1]), dtype=np.int32)
    for index, ids in enumerate(encoded):
        flat[offsets[index] : offsets[index + 1]] = ids
    return _Tokenized(flat=flat, offsets=offsets, lengths=lengths)


def _tokenized(tokenizer, texts, max_len) -> _Tokenized:
    """Tokenise `texts` once per process and reuse it for every later epoch.

    The evaluation splits never change during a run, so re-tokenising them on
    every pass is pure repeated work -- the same reason `_read_eval_csv` and
    `load_benchmark` cache their parses. It matters most for retrieval, whose
    ~101k documents are the bulk of a test pass.
    """
    global _TOKEN_CACHE_SIZE

    entries = _TOKEN_CACHE.setdefault(tokenizer, {})
    key = (max_len, _texts_digest(texts))
    cached = entries.get(key)
    if cached is not None:
        return cached

    tokenized = _tokenize_all(tokenizer, texts, max_len)
    if _TOKEN_CACHE_SIZE + tokenized.flat.size <= _TOKEN_CACHE_LIMIT:
        entries[key] = tokenized
        _TOKEN_CACHE_SIZE += tokenized.flat.size
    return tokenized


def _plan_batches(lengths: np.ndarray) -> list[np.ndarray]:
    """Group indices into batches of similar length under the token budget.

    Sorting by token length rather than by character count is what removes the
    padding: the two disagree badly on text whose tokens-per-character varies,
    and fiqa's financial prose is the worst of the five. Over the five retrieval
    corpora, character order at batch 256 forwards 31.76M padded tokens against
    20.15M real ones; token order under the budget forwards 20.78M in 162 batches
    rather than 397 -- 1.53x less work for the same result, padding being masked.
    """
    order = np.argsort(lengths, kind="stable")
    if not EVAL_TOKEN_BUDGET:
        return [
            order[start : start + EVAL_BATCH_SIZE]
            for start in range(0, order.size, EVAL_BATCH_SIZE)
        ]

    batches = []
    start = 0
    width = 0
    for position, index in enumerate(order):
        candidate = max(width, int(lengths[index]))
        count = position - start
        if count and (
            count >= _MAX_BATCH_SEQUENCES or (count + 1) * candidate > EVAL_TOKEN_BUDGET
        ):
            batches.append(order[start:position])
            start = position
            candidate = int(lengths[index])
        width = candidate
    if start < order.size:
        batches.append(order[start:])
    return batches


def _pad_batch(tokenized: _Tokenized, indices: np.ndarray, pad_id: int, left: bool):
    """Materialise one batch as (input_ids, attention_mask), padded to its widest.

    This is what `padding=True` did inside the tokenizer call; doing it here is
    both cheaper -- building fiqa's padded tensors cost 12.5 of the 32.1s that
    call took -- and the only way to batch by token length, which is not known
    until after tokenisation.
    """
    lengths = tokenized.lengths[indices]
    width = int(lengths.max())
    ids = np.full((indices.size, width), pad_id, dtype=np.int64)
    positions = np.arange(width)
    if left:
        mask = positions >= (width - lengths)[:, None]
        for row, index in enumerate(indices):
            ids[row, width - lengths[row] :] = tokenized.flat[
                tokenized.offsets[index] : tokenized.offsets[index + 1]
            ]
    else:
        mask = positions < lengths[:, None]
        for row, index in enumerate(indices):
            ids[row, : lengths[row]] = tokenized.flat[
                tokenized.offsets[index] : tokenized.offsets[index + 1]
            ]
    return torch.from_numpy(ids), torch.from_numpy(mask.astype(np.int64))


def _embed_texts(model, tokenizer, texts, max_len, desc=None):
    """Embed `texts` and return a float32 array in the original order.

    Batches are formed over length-sorted indices: with padding, the cost of a
    batch is `len(batch) * max_len_in_batch`, so grouping similar lengths removes
    padding the model would otherwise attend over. Padding is masked out, so the
    per-sequence result is unchanged. `_plan_batches` sorts on token length and
    caps the batch by token budget rather than by sequence count; `_tokenized`
    keeps the ids so later epochs skip the tokenizer entirely.
    """
    device = model.device
    total = len(texts)
    if total == 0:
        return np.empty((0, 0), dtype=np.float32)

    tokenized = _tokenized(tokenizer, texts, max_len)
    batches = _plan_batches(tokenized.lengths)

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        # Masked out either way; 0 is the one id every vocabulary has.
        pad_id = 0
    left = getattr(tokenizer, "padding_side", "right") == "left"

    embeddings = None
    autocast = torch.amp.autocast(
        "cuda", dtype=torch.float16, enabled=torch.cuda.is_available()
    )

    def prepare(indices):
        return _pad_batch(tokenized, indices, pad_id, left)

    with ThreadPoolExecutor(max_workers=1) as pool, autocast, torch.inference_mode():
        pending = deque()
        next_batch = 0
        while next_batch < len(batches) and len(pending) <= EVAL_PREFETCH:
            pending.append(
                (batches[next_batch], pool.submit(prepare, batches[next_batch]))
            )
            next_batch += 1

        for _ in tqdm(range(len(batches)), desc=desc, leave=False):
            indices, future = pending.popleft()
            if next_batch < len(batches):
                pending.append(
                    (batches[next_batch], pool.submit(prepare, batches[next_batch]))
                )
                next_batch += 1

            input_ids, attention_mask = future.result()
            output = model(
                input_ids=input_ids.to(device, non_blocking=True),
                attention_mask=attention_mask.to(device, non_blocking=True),
            )
            batch_embeddings = _pooled_embedding(output).float().cpu().numpy()
            if embeddings is None:
                embeddings = np.empty(
                    (total, batch_embeddings.shape[1]), dtype=np.float32
                )
            embeddings[indices] = batch_embeddings

    if embeddings is None:
        return np.empty((0, 0), dtype=np.float32)
    return embeddings


def _cosine_similarity(left, right):
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    return np.sum(left * right, axis=1)


# The three readers below just hold the columns of one split. Embedding goes
# through `_embed_texts` over plain lists of strings, so none of them is ever
# iterated by a DataLoader.
class STSDataset:
    """sentence1 / sentence2 / score of one STS split."""

    def __init__(self, file_path):
        frame = _read_eval_csv(file_path)
        self.sentence1 = _column_as_list(frame, "sentence1")
        self.sentence2 = _column_as_list(frame, "sentence2")
        self.labels = frame["score"].to_numpy(dtype=np.float32)


def eval_sts(model, tokenizer, path):
    dataset = STSDataset(path)
    emb1 = _embed_texts(model, tokenizer, dataset.sentence1, 128, desc="sts s1")
    emb2 = _embed_texts(model, tokenizer, dataset.sentence2, 128, desc="sts s2")

    # scale [-1,1] -> [0,5]
    preds = (_cosine_similarity(emb1, emb2) + 1) * 2.5

    spearman_corr, _ = spearmanr(preds, dataset.labels)
    print(f"Spearman: {spearman_corr:.4f}")

    return spearman_corr


def eval_sts_task(model, path_list, tokenizer):
    model.eval()
    print(" eval_sts_task")
    results = {}
    for path in path_list:
        print(path)
        results[path] = eval_sts(model, tokenizer, path)
    model.train()
    return results


def eval_cls(model, tokenizer, path):
    dataset = ClasssifyDataset(path)
    features = _embed_texts(model, tokenizer, dataset.texts, 512, desc=Path(path).stem)
    return features, dataset.labels


class ClasssifyDataset:
    """text / label of one classification split."""

    def __init__(self, file_path):
        frame = _read_eval_csv(file_path)
        self.texts = _column_as_list(frame, "text")
        self.labels = frame["label"].to_numpy(dtype=np.int64)


def _normalized_text_keys(dataset):
    return {" ".join(str(text).strip().casefold().split()) for text in dataset["text"]}


@functools.cache
def _validate_classification_pair(train_path, eval_path):
    """Check train/eval leakage once per (train, eval) pair per process.

    The files are fixed for the duration of a run, so re-normalising ~55k texts
    on every epoch's evaluation only repeats a verdict already reached.
    """
    eval_file = BASE_DIR / eval_path
    train_frame = _read_eval_csv(train_path)
    eval_frame = _read_eval_csv(eval_path)
    overlap = _normalized_text_keys(train_frame) & _normalized_text_keys(eval_frame)

    if "val_set" in Path(eval_path).parts:
        if overlap:
            raise ValueError(
                f"Classification train-validation leakage for {eval_file.stem}: "
                f"{len(overlap)} normalized texts overlap"
            )
        return

    dataset_name = eval_file.stem.removesuffix("_test")
    validation_path = f"data/val_set/{dataset_name}_validation.csv"
    validation_overlap = set()
    if (BASE_DIR / validation_path).is_file():
        validation_frame = _read_eval_csv(validation_path)
        validation_overlap = _normalized_text_keys(
            validation_frame
        ) & _normalized_text_keys(eval_frame)

    if overlap or validation_overlap:
        warnings.warn(
            f"Published test split {eval_file.stem} has normalized-text "
            f"overlap: train={len(overlap)}, validation={len(validation_overlap)}.",
            RuntimeWarning,
            stacklevel=2,
        )


def eval_classification_task(model, path_list, tokenizer):
    model.eval()
    print(" eval classifier")

    results = {}
    for train_path, dev_path in path_list:
        print(dev_path)
        _validate_classification_pair(train_path, dev_path)

        X_train, y_train = eval_cls(model, tokenizer, train_path)
        X_test, y_test = eval_cls(model, tokenizer, dev_path)

        clf = LogisticRegression(
            random_state=42,
            max_iter=200,
            verbose=0,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        scores = {}
        accuracy = accuracy_score(y_test, y_pred)
        scores["accuracy"] = accuracy
        f1 = f1_score(y_test, y_pred, average="macro")
        scores["f1"] = f1
        print(scores)
        results[dev_path] = scores

    model.train()
    return results


class PairDataset:
    """sentence1 / sentence2 / label of one pair-classification split."""

    def __init__(self, file_path):
        frame = _read_eval_csv(file_path)
        self.sentence1 = _column_as_list(frame, "sentence1")
        self.sentence2 = _column_as_list(frame, "sentence2")
        self.labels = frame["label"].to_numpy(dtype=np.float32)


def eval_pair(model, tokenizer, path, threshold=None):
    dataset = PairDataset(path)
    emb1 = _embed_texts(model, tokenizer, dataset.sentence1, 128, desc="pair s1")
    emb2 = _embed_texts(model, tokenizer, dataset.sentence2, 128, desc="pair s2")

    preds = (_cosine_similarity(emb1, emb2) + 1) / 2

    metric = get_metric_pair_classification(preds, dataset.labels, threshold=threshold)
    print(metric)

    return metric


def get_metric_pair_classification(scores, labels, threshold=None):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if threshold is None:
        # Accuracy over a fixed threshold grid, evaluated as one vectorised sweep
        # instead of 200 separate sklearn calls.
        candidates = np.linspace(0, 1, 200)
        decisions = scores[None, :] >= candidates[:, None]
        accuracies = (decisions == labels[None, :].astype(bool)).mean(axis=1)
        # np.argmax keeps the first maximum, matching the strict `>` update the
        # sequential loop used, so the selected threshold is unchanged.
        best_index = int(np.argmax(accuracies))
        best_acc, best_thr = (
            float(accuracies[best_index]),
            float(candidates[best_index]),
        )
    else:
        best_thr = float(threshold)
        best_acc = accuracy_score(labels, (scores >= best_thr).astype(int))
    preds = (scores >= best_thr).astype(int)
    return {
        "best_threshold": best_thr,
        "accuracy": best_acc,
        "f1": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro"),
        "average_precision": average_precision_score(labels, scores),
    }


def eval_pair_task(model, path_list, tokenizer, thresholds=None):
    model.eval()
    print(" eval_pair_task")
    results = {}
    selected_thresholds = {}
    for index, path in enumerate(path_list):
        print(path)
        threshold = None if thresholds is None else thresholds[index]
        metric = eval_pair(model, tokenizer, path, threshold=threshold)
        results[path] = metric
        selected_thresholds[index] = metric["best_threshold"]
    model.train()
    return results, selected_thresholds


# Evaluation datasets grouped by physical split.
eval_cls_tasks = [
    (
        "data/train_set/banking77_train.csv",
        "data/val_set/banking77_validation.csv",
    ),
    (
        "data/train_set/emotion_train.csv",
        "data/val_set/emotion_validation.csv",
    ),
    (
        "data/train_set/tweet_train.csv",
        "data/val_set/tweet_validation.csv",
    ),
]

eval_sts_tasks = [
    "data/val_set/sick_validation.csv",
    "data/val_set/sts12_validation.csv",
    "data/val_set/stsb_validation.csv",
]

eval_pair_tasks = [
    "data/val_set/mrpc_validation.csv",
    "data/val_set/scitail_validation.csv",
    "data/val_set/wic_validation.csv",
]

test_cls_tasks = [
    (
        "data/train_set/banking77_train.csv",
        "data/test_set/banking77_test.csv",
    ),
    (
        "data/train_set/emotion_train.csv",
        "data/test_set/emotion_test.csv",
    ),
    (
        "data/train_set/tweet_train.csv",
        "data/test_set/tweet_test.csv",
    ),
]

test_sts_tasks = [
    "data/test_set/sick_test.csv",
    "data/test_set/sts12_test.csv",
    "data/test_set/stsb_test.csv",
]

test_pair_tasks = [
    "data/test_set/mrpc_test.csv",
    "data/test_set/scitail_test.csv",
    "data/test_set/wic_test.csv",
]
