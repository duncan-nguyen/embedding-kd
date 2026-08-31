"""Dataset and collate for the methods that train against a frozen teacher cache.

The teacher never runs during training here: row ``i`` of the cache is the
teacher's embedding of row ``i`` of the corpus, so the dataset only has to carry
the text and hand the matching cached vector along with it.
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset


class TextPairWithTeacher(Dataset):
    """Corpus rows paired with their cached teacher embeddings.

    ``teacher_topo`` is the same cache in the teacher's *own* dimension, carried
    only when the H0 term is switched on; every other run leaves it ``None`` and
    the batch looks exactly as it did before.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        task: str,
        teacher_cls: torch.Tensor,
        teacher_topo: torch.Tensor | None = None,
    ):
        self.task = task
        self.teacher_cls = teacher_cls    # [N, d_t]
        self.teacher_topo = teacher_topo  # [N, d_T] unprojected, or None

        if task == "single_cls":
            columns = (df["text"].astype(str), df["label"].astype(int))
        elif task == "pair_cls":
            columns = (df["premise"].astype(str), df["hypothesis"].astype(str))
        else:
            columns = (df["sentence1"].astype(str), df["sentence2"].astype(str))
        self.samples = list(zip(*columns))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        if self.teacher_topo is None:
            return item, self.teacher_cls[idx]
        return item, self.teacher_cls[idx], self.teacher_topo[idx]


class DualTokenizerCollateWithTeacher:
    """Tokenize for the student only; the teacher side arrives pre-computed."""

    def __init__(self, tok_student, task: str, max_len: int):
        self.ts = tok_student
        self.task = task
        self.max_len = max_len

    def _encode(self, texts, side: int, out: dict) -> None:
        encoding = self.ts(
            list(texts),
            max_length=self.max_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        # single_cls has one text per row and drops the index from the key.
        suffix = f"{side}_stu" if side else "_stu"
        out[f"input_ids{suffix}"] = encoding["input_ids"]
        out[f"attention_mask{suffix}"] = encoding["attention_mask"]
        out[f"special_tokens_mask{suffix}"] = encoding["special_tokens_mask"]
        if "token_type_ids" in encoding:
            out[f"token_type_ids{suffix}"] = encoding["token_type_ids"]

    def __call__(self, batch):
        columns = list(zip(*batch))
        samples = columns[0]
        out = {"teacher_cls": torch.stack(columns[1], dim=0)}  # [B, d_t]
        # Present only when the dataset also carries the unprojected teacher cache
        # (the H0 term); every other method sees the batch exactly as before.
        if len(columns) > 2:
            out["teacher_topo"] = torch.stack(columns[2], dim=0)

        if self.task == "single_cls":
            texts, labels = zip(*samples)
            self._encode(texts, 0, out)
            out["labels"] = torch.tensor(labels, dtype=torch.long)
            return out

        first, second = zip(*samples)
        self._encode(first, 1, out)
        self._encode(second, 2, out)
        return out
