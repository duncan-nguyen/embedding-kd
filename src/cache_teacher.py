import hashlib
import os
import re
from collections import deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm
from transformers import AutoModel

from .pooling import pool_sentence_embedding

# How many batches ahead the tokenizer thread runs. Fast tokenizers release the
# GIL, so a thread is enough to keep the next batch ready while the GPU works on
# the current one, and it avoids the process fork a second DataLoader would cost.
CACHE_PREFETCH = 2

# Metadata fields that identify *which* teacher signal a cache file holds. A cache
# is reused only when every one of them matches the run asking for it: two
# teachers of the same width (Qwen3-Embedding-0.6B and BGE-M3 are both 1024-d)
# would otherwise be indistinguishable by shape alone, and a corpus rebuilt to the
# same path with the same row count would be indistinguishable by shape *or* name.
CACHE_IDENTITY_KEYS = (
    "teacher_model_name",
    "pooling_method",
    "normalize",
    "max_length",
    "train_data_digest",
)


def corpus_digest(path: str | os.PathLike[str]) -> str:
    """Content fingerprint of a training corpus, as a short hex digest.

    The cache is keyed on this rather than on the corpus *path*: `train_150k.csv`
    rebuilt from a different MS MARCO shard keeps its name and its row count, so
    nothing else would notice that the cached embeddings no longer describe it.
    Reading 30 MB to avoid re-running a 4B-parameter teacher is a good trade.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def cache_filename(
    *,
    teacher_model_name: str,
    pooling_method: str,
    train_data_path: str | os.PathLike[str],
    max_length: int,
    normalize: bool,
    train_data_digest: str | None = None,
) -> str:
    """A filename that encodes everything a cache's reusability depends on.

    With this name, one shared directory can hold every cache a project builds and
    a run either finds exactly its own or misses -- it can never load someone
    else's and be turned away by :func:`validate_cached_embeddings`, which is what
    a hand-written shared filename invites. The readable prefix is for humans
    listing the directory; the digest carries the rest, corpus contents included.
    """
    if train_data_digest is None:
        train_data_digest = corpus_digest(train_data_path)
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(teacher_model_name)).strip("-").lower()
    identity = "|".join(
        [
            str(teacher_model_name),
            str(pooling_method),
            str(int(max_length)),
            str(bool(normalize)),
            train_data_digest,
        ]
    )
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    corpus = Path(train_data_path).stem
    return f"{slug}__{corpus}__{pooling_method}__{key}.pt"


def _length_sorted_batches(texts: Sequence[str], batch_size: int) -> list[list[int]]:
    """Corpus row indices grouped into batches of similar length.

    With ``padding=True`` a batch costs ``len(batch) * max_len_in_batch``, so a
    batch drawn in corpus order pays for its single longest sentence on every row
    it holds. This corpus is heavily skewed -- median ~17 tokens against a p99 of
    ~144 -- and grouping by length removes 2.1x (batch 32) to 2.5x (batch 128) of
    the padded tokens the teacher would otherwise attend over.

    Character length stands in for token length: it needs no tokenizer pass of its
    own and it is the same proxy the evaluation encoder already sorts on. Batching
    is a grouping decision only -- padding is masked out, so each row's embedding
    is what it would have been in any other batch -- and the returned indices carry
    the original positions so the cache can be written back in corpus order.
    """
    order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    return [
        order[start : start + batch_size] for start in range(0, len(order), batch_size)
    ]


def cache_teacher_embeddings(
    model_teacher: AutoModel,
    texts: Sequence[str],
    tokenizer,
    device: torch.device,
    *,
    max_length: int,
    batch_size: int,
    pooling_method: str = "last_token",
    cache_path: str | None = None,
    dtype: torch.dtype = torch.float32,
    use_amp: bool = True,
    normalize: bool = False,
    metadata: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Run the teacher once over ``texts`` and cache the pooled embeddings.

    Row ``i`` of the result is the teacher's embedding of ``texts[i]``, whatever
    order the batches were formed in.

    The teacher tokenizer is applied here rather than in a shared collate: the
    training collate encodes both sides of a pair for both models, and this pass
    reads exactly one of those four encodings, so going through it would tokenize
    four times (twice over the *same* string, since a one-column corpus is read as
    a degenerate pair) and copy twice as many tensors to the GPU as it uses.

    ``metadata`` (teacher name, corpus, ...) is stored next to the tensor so a
    later run can tell whether the file is the cache it expects; see
    :func:`validate_cached_embeddings`.
    """
    if cache_path and os.path.exists(cache_path):
        embeddings, _ = load_cached_embeddings(cache_path)
        return embeddings

    print("Pre-computing teacher embeddings...")
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad_(False)

    texts = list(texts)
    batches = _length_sorted_batches(texts, batch_size)

    def tokenize(indices: list[int]):
        return tokenizer(
            [texts[index] for index in indices],
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

    # Allocated on the first batch, but deliberately *outside* inference mode: a
    # tensor created inside it is an inference tensor, and the targets go on to be
    # read by a loss that saves them for backward, which inference tensors refuse.
    # Writing inference tensors into a normal buffer is fine, only the buffer's own
    # origin matters.
    teacher_cls_all: torch.Tensor | None = None
    amp = autocast("cuda", enabled=use_amp and torch.cuda.is_available())

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending: deque = deque()
        next_batch = 0
        while next_batch < len(batches) and len(pending) <= CACHE_PREFETCH:
            pending.append(
                (batches[next_batch], pool.submit(tokenize, batches[next_batch]))
            )
            next_batch += 1

        for _ in tqdm(range(len(batches)), desc="Caching teacher embeddings"):
            indices, future = pending.popleft()
            if next_batch < len(batches):
                pending.append(
                    (batches[next_batch], pool.submit(tokenize, batches[next_batch]))
                )
                next_batch += 1

            encoded = future.result()

            with torch.inference_mode(), amp:
                input_ids = encoded["input_ids"].to(device, non_blocking=True)
                attention_mask = encoded["attention_mask"].to(device, non_blocking=True)
                out = model_teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                pooled = pool_sentence_embedding(
                    out.last_hidden_state, attention_mask, pooling_method
                )
                if normalize:
                    pooled = F.normalize(pooled, p=2, dim=-1)
                pooled = pooled.to(dtype).cpu()

            if teacher_cls_all is None:
                teacher_cls_all = torch.empty(
                    (len(texts), pooled.shape[-1]), dtype=pooled.dtype
                )
            # Undo the length sort: the cache is indexed by corpus row.
            teacher_cls_all[indices] = pooled

    if teacher_cls_all is None:
        raise ValueError("cache_teacher_embeddings was given no texts to encode")

    if cache_path:
        save_cached_embeddings(
            cache_path,
            teacher_cls_all,
            {
                "pooling_method": pooling_method,
                "normalize": bool(normalize),
                **(metadata or {}),
            },
        )
        print(f"Saved cached teacher embeddings to: {cache_path}")

    print(f"Done caching teacher embeddings: {teacher_cls_all.shape}")
    return teacher_cls_all


def save_cached_embeddings(
    cache_path: str, embeddings: torch.Tensor, metadata: dict[str, Any]
) -> None:
    """Write ``{"embeddings", "metadata"}``; the shape is recorded in the metadata too.

    Written to a temporary file next to the destination and moved into place, so
    the cache never exists half-written. That matters as soon as more than one
    training job shares a ``--cache_dir``: with a plain write, a job that starts
    while another is still saving loads a truncated file and dies on an unpickling
    error minutes into the sweep. Two jobs that both build the cache still both
    encode the corpus -- wasted teacher passes, not a corrupt file -- which is why
    the runner warms the cache once before it fans jobs out.
    """
    directory = os.path.dirname(cache_path)
    os.makedirs(directory if directory else ".", exist_ok=True)
    payload = {
        "embeddings": embeddings,
        "metadata": {
            **metadata,
            "rows": int(embeddings.shape[0]),
            "dim": int(embeddings.shape[-1]),
        },
    }
    temporary = f"{cache_path}.tmp.{os.getpid()}"
    try:
        torch.save(payload, temporary)
        os.replace(temporary, cache_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def load_cached_embeddings(cache_path: str) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return ``(embeddings, metadata)``.

    A bare tensor (the pre-metadata format) loads with empty metadata, so an old
    cache still works; it just cannot be checked against the run beyond its shape.
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    print(f"Loading cached embeddings from: {cache_path}")
    payload = torch.load(cache_path, map_location="cpu")
    if torch.is_tensor(payload):
        embeddings, metadata = payload, {}
    elif isinstance(payload, dict) and torch.is_tensor(payload.get("embeddings")):
        embeddings = payload["embeddings"]
        metadata = dict(payload.get("metadata") or {})
    else:
        raise ValueError(
            f"{cache_path} is not a teacher embedding cache: expected a tensor or a "
            f"dict with an 'embeddings' tensor, got {type(payload).__name__}"
        )
    print(f"Loaded embeddings: {tuple(embeddings.shape)}")
    return embeddings, metadata


def validate_cached_embeddings(
    embeddings: torch.Tensor,
    metadata: dict[str, Any],
    cache_path: str,
    *,
    teacher_model_name: str,
    pooling_method: str,
    normalize: bool,
    teacher_dim: int | None = None,
    rows: int | None = None,
    max_length: int | None = None,
    train_data_digest: str | None = None,
) -> None:
    """Refuse a cache that was built for a different teacher, pooling or corpus.

    Every identity field the cache carries has to equal the run's setting, and the
    tensor's shape has to match the teacher width and corpus length. A legacy cache
    without metadata is checked by shape only and says so, since a same-width
    teacher swap is exactly what shape cannot see.
    """
    expected = {
        "teacher_model_name": teacher_model_name,
        "pooling_method": pooling_method,
        "normalize": bool(normalize),
        "max_length": None if max_length is None else int(max_length),
        "train_data_digest": train_data_digest,
    }
    problems = []
    for key in CACHE_IDENTITY_KEYS:
        actual = metadata.get(key)
        # Either side may be absent: an older cache carries fewer fields, and a
        # caller may not know one. Only two *known and different* values are a clash.
        if actual is not None and expected[key] is not None and actual != expected[key]:
            problems.append(
                f"{key}: cache has {actual!r}, this run uses {expected[key]!r}"
            )
    if teacher_dim is not None and int(embeddings.shape[-1]) != int(teacher_dim):
        problems.append(
            f"embedding dim: cache has {int(embeddings.shape[-1])}, "
            f"{teacher_model_name} produces {int(teacher_dim)}"
        )
    if rows is not None and int(embeddings.shape[0]) != int(rows):
        problems.append(
            f"rows: cache has {int(embeddings.shape[0])}, training data has {int(rows)}"
        )
    if problems:
        details = "\n".join(f"  - {problem}" for problem in problems)
        raise ValueError(
            f"Cached teacher embeddings at {cache_path} do not match this run:\n"
            f"{details}\n"
            "Use a separate --cache_path per teacher / pooling / corpus, or delete "
            "the file to rebuild it."
        )
    if not metadata:
        print(
            f"[WARN] {cache_path} carries no metadata (built before caches were "
            "tagged); it matches this run by shape only, so make sure it really "
            f"holds {teacher_model_name} / {pooling_method} embeddings."
        )
