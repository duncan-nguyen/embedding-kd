"""Dataset and collate for the methods that train against a frozen teacher cache.

The teacher never runs during training here: row ``i`` of the cache is the
teacher's embedding of row ``i`` of the corpus, so the dataset only has to carry
the text and hand the matching cached vector along with it.
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.criterions.h0_topological_loss import h0_death_times


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
    """Tokenize for the student only; the teacher side arrives pre-computed.

    Three switches say what the batch is actually going to be read for, because
    everything this builds is paid for on every step of every epoch:

    ``need_second_text``
        The second sentence of the pair. GeoODE's default contrastive view is two
        dropout passes over the *first* sentence, so the second one is tokenized,
        padded, stacked and copied to the GPU without ever being read -- and for a
        one-column corpus read as a degenerate pair it is a copy of the first.
    ``need_special_tokens_mask``
        Only the token-level methods (cdm, dskd) align token strings; the
        cached-teacher methods never look at it.
    ``topo_metric``
        When set, the teacher's H0 diagram is built here instead of in the training
        step. It is a constant -- a frozen cache under no_grad -- and the batch's
        rows are known at collate time, so a DataLoader worker can build it in
        parallel with the previous step rather than the GPU building it on the
        critical path. It also replaces a ``[B, d_T]`` host-to-device copy per step
        with a ``[B - 1]`` one.
    """

    def __init__(
        self,
        tok_student,
        task: str,
        max_len: int,
        *,
        need_second_text: bool = True,
        need_special_tokens_mask: bool = False,
        topo_metric: str | None = None,
    ):
        self.ts = tok_student
        self.task = task
        self.max_len = max_len
        self.need_second_text = bool(need_second_text)
        self.need_special_tokens_mask = bool(need_special_tokens_mask)
        self.topo_metric = topo_metric

    def _encode(self, texts, side: int, out: dict) -> None:
        encoding = self.ts(
            list(texts),
            max_length=self.max_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=self.need_special_tokens_mask,
        )
        # single_cls has one text per row and drops the index from the key.
        suffix = f"{side}_stu" if side else "_stu"
        out[f"input_ids{suffix}"] = encoding["input_ids"]
        out[f"attention_mask{suffix}"] = encoding["attention_mask"]
        if self.need_special_tokens_mask:
            out[f"special_tokens_mask{suffix}"] = encoding["special_tokens_mask"]
        if "token_type_ids" in encoding:
            out[f"token_type_ids{suffix}"] = encoding["token_type_ids"]

    def _teacher_topo(self, topo: torch.Tensor, out: dict) -> None:
        """Either the raw teacher cache, or the diagram that is all anyone reads."""
        if self.topo_metric is None:
            out["teacher_topo"] = topo
            return
        with torch.no_grad():
            out["teacher_deaths"] = h0_death_times(
                topo.float(), metric=self.topo_metric, sort=True
            )

    def __call__(self, batch):
        columns = list(zip(*batch))
        samples = columns[0]
        out = {"teacher_cls": torch.stack(columns[1], dim=0)}  # [B, d_t]
        # Present only when the dataset also carries the unprojected teacher cache
        # (the H0 term); every other method sees the batch exactly as before.
        if len(columns) > 2:
            topo = torch.stack(columns[2], dim=0)
            # A single-row tail batch has no MST, so there is no diagram to build.
            if self.topo_metric is not None and topo.shape[0] < 2:
                out["teacher_topo"] = topo
            else:
                self._teacher_topo(topo, out)

        if self.task == "single_cls":
            texts, labels = zip(*samples)
            self._encode(texts, 0, out)
            out["labels"] = torch.tensor(labels, dtype=torch.long)
            return out

        first, second = zip(*samples)
        self._encode(first, 1, out)
        if self.need_second_text:
            self._encode(second, 2, out)
        return out
