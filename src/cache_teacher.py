import os
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

from .pooling import pool_sentence_embedding

# Metadata fields that identify *which* teacher signal a cache file holds. A cache
# is reused only when every one of them matches the run asking for it: two
# teachers of the same width (Qwen3-Embedding-0.6B and BGE-M3 are both 1024-d)
# would otherwise be indistinguishable by shape alone.
CACHE_IDENTITY_KEYS = ("teacher_model_name", "pooling_method", "normalize")


def cache_teacher_embeddings(
    model_teacher: AutoModel,
    dataloader: DataLoader,
    device: torch.device,
    pooling_method: str = "last_token",
    cache_path: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    use_amp: bool = True,
    normalize: bool = False,
    metadata: Optional[dict[str, Any]] = None,
) -> torch.Tensor:
    """Run the teacher once over ``dataloader`` and cache the pooled embeddings.

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

    data_cls = []
    pbar = tqdm(dataloader, desc="Caching teacher CLS embeddings")

    with torch.inference_mode():
        for batch in pbar:
            batch_t = {}
            for k, v in batch.items():
                if not torch.is_tensor(v):
                    continue
                if k.endswith("_tea"):
                    batch_t[k] = v.to(device, non_blocking=True)

            with autocast(
                "cuda",
                enabled=use_amp and torch.cuda.is_available(),
            ):
                t_out1 = model_teacher(
                    input_ids=batch_t["input_ids1_tea"],
                    attention_mask=batch_t["attention_mask1_tea"],
                    return_dict=True
                )
                T_last1 = t_out1.last_hidden_state  # [B, L, d_t]
                T_cls1 = pool_sentence_embedding(
                    T_last1, batch_t["attention_mask1_tea"], pooling_method
                )

                if normalize:
                    T_cls1 = F.normalize(T_cls1, p=2, dim=-1)
                T_cls1 = T_cls1.to(dtype)
            data_cls.append(T_cls1.cpu())
    teacher_cls_all = torch.cat(data_cls, dim=0)
    if cache_path:
        save_cached_embeddings(
            cache_path,
            teacher_cls_all,
            {"pooling_method": pooling_method, "normalize": bool(normalize), **(metadata or {})},
        )
        print(f"Saved cached teacher embeddings to: {cache_path}")

    print(f"Done caching teacher embeddings: {teacher_cls_all.shape}")
    return teacher_cls_all


def save_cached_embeddings(
    cache_path: str, embeddings: torch.Tensor, metadata: dict[str, Any]
) -> None:
    """Write ``{"embeddings", "metadata"}``; the shape is recorded in the metadata too."""
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
    torch.save(payload, cache_path)


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
    teacher_dim: Optional[int] = None,
    rows: Optional[int] = None,
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
    }
    problems = []
    for key in CACHE_IDENTITY_KEYS:
        actual = metadata.get(key)
        if actual is not None and actual != expected[key]:
            problems.append(f"{key}: cache has {actual!r}, this run uses {expected[key]!r}")
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


def clear_cache_and_free_memory():
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    print("Done clearing GPU cache and freeing memory")
