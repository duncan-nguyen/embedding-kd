"""Raw-text dataset and collate for the methods that run the teacher online."""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.criterions.contextual_dynamic_mapping import (
    build_special_token_mapper,
    row_alignment,
)
from src.criterions.emo_embedding_distillation import (
    align_batch,
    alignment_index_tensor,
)

# Methods whose KD term first has to align two tokenizations of the same sentence.
# The alignment is a function of the token ids alone, so it is built here rather
# than in the training step: a DataLoader worker derives it while the GPU is still
# on the previous batch, and the step is handed index tensors.
ALIGNMENT_METHODS = frozenset({"cdm", "emo"})


class TextPairRaw(Dataset):
    """Canonical raw-text dataset used by online teacher/student methods.

    Every row is ``(text1, text2 | None, label | None)``; which columns supply
    them is what the task type decides.
    """

    def __init__(self, df: pd.DataFrame, task: str):
        self.task = task
        if task == "single_cls":
            if "text" not in df.columns or "label" not in df.columns:
                raise ValueError("single_cls data requires 'text' and 'label' columns")
            self.samples = [
                (text, None, int(label))
                for text, label in zip(df["text"].astype(str), df["label"].astype(int))
            ]
        elif task == "pair_cls":
            if "premise" not in df.columns or "hypothesis" not in df.columns:
                raise ValueError(
                    "pair_cls data requires 'premise' and 'hypothesis' columns"
                )
            labels = (
                df["label"].astype(int).tolist()
                if "label" in df.columns
                else [None] * len(df)
            )
            self.samples = [
                (premise, hypothesis, label)
                for premise, hypothesis, label in zip(
                    df["premise"].astype(str),
                    df["hypothesis"].astype(str),
                    labels,
                )
            ]
        elif task == "pair_reg":
            if "sentence1" not in df.columns or "sentence2" not in df.columns:
                raise ValueError(
                    "pair_reg data requires 'sentence1' and 'sentence2' columns"
                )
            score_column = "score" if "score" in df.columns else "label"
            if score_column not in df.columns:
                raise ValueError("pair_reg data requires a 'score' or 'label' column")
            self.samples = [
                (sentence1, sentence2, float(score))
                for sentence1, sentence2, score in zip(
                    df["sentence1"].astype(str),
                    df["sentence2"].astype(str),
                    df[score_column].astype(float),
                )
            ]
        else:
            raise ValueError(f"Unsupported task type: {task}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[str, str | None, float | None]:
        return self.samples[idx]


class DualTokenizerCollate:
    """Tokenizes each row for the student and, when there is one, the teacher.

    ``tok_teacher=None`` is the teacher-free case (the SimCSE-only control): the
    batch then carries the ``*_stu`` tensors alone, so no teacher tokenizer has to
    be downloaded to run a method that never encodes anything with it.
    """

    def __init__(
        self,
        tok_student,
        tok_teacher,
        task: str,
        max_len: int,
        *,
        alignment: str | None = None,
        teacher_token: str = "\u0120",
        student_token: str = "##",
    ):
        self.ts = tok_student
        self.tt = tok_teacher
        self.task = task
        self.max_len = max_len
        if alignment is not None and alignment not in ALIGNMENT_METHODS:
            raise ValueError(
                f"Unsupported alignment={alignment!r}; expected one of "
                f"{sorted(ALIGNMENT_METHODS)} or None"
            )
        if alignment is not None and tok_teacher is None:
            raise ValueError(
                f"alignment={alignment!r} needs a teacher tokenizer to align against"
            )
        self.alignment = alignment
        # The same two config fields, read differently by the two methods, as the
        # config documents: CDM takes them as the sub-word markers to strip before
        # comparing token strings, EMO as the special tokens to refuse to align.
        self.teacher_token = teacher_token
        self.student_token = student_token
        self._spec_mapper = (
            build_special_token_mapper(tok_student, tok_teacher)
            if alignment == "cdm"
            else None
        )

    def _encode(self, tokenizer, texts: list[str]):
        return tokenizer(
            texts,
            max_length=self.max_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )

    @staticmethod
    def _add_encoding(out, encoding, side: int, model_suffix: str):
        key_suffix = f"{side}_{model_suffix}"
        out[f"input_ids{key_suffix}"] = encoding["input_ids"]
        out[f"attention_mask{key_suffix}"] = encoding["attention_mask"]
        out[f"special_tokens_mask{key_suffix}"] = encoding["special_tokens_mask"]
        if "token_type_ids" in encoding:
            out[f"token_type_ids{key_suffix}"] = encoding["token_type_ids"]

    def _cdm_alignment(self, student, teacher) -> torch.Tensor:
        """``[P, 3]`` of ``(row, teacher position, student position)`` for CDM.

        The positions are absolute in the padded batch, so the training step reads
        them straight off ``last_hidden_state``. Special and padding tokens are
        dropped before the DTW runs, exactly as the step used to drop them.
        """
        keep_stu = (
            student["attention_mask"].bool() & ~student["special_tokens_mask"].bool()
        )
        keep_tea = (
            teacher["attention_mask"].bool() & ~teacher["special_tokens_mask"].bool()
        )
        ids_stu, ids_tea = student["input_ids"], teacher["input_ids"]

        pairs = []
        for row in range(ids_stu.size(0)):
            positions_stu = keep_stu[row].nonzero().flatten().tolist()
            positions_tea = keep_tea[row].nonzero().flatten().tolist()
            if not positions_stu or not positions_tea:
                continue
            tokens_stu = self.ts.convert_ids_to_tokens(
                ids_stu[row, positions_stu].tolist()
            )
            tokens_tea = self.tt.convert_ids_to_tokens(
                ids_tea[row, positions_tea].tolist()
            )
            pairs.extend(
                (row, positions_tea[i], positions_stu[j])
                for i, j in row_alignment(
                    tokens_tea,
                    tokens_stu,
                    self.teacher_token,
                    self.student_token,
                    self._spec_mapper,
                )
            )
        if not pairs:
            return torch.zeros((0, 3), dtype=torch.long)
        return torch.tensor(pairs, dtype=torch.long)

    def _emo_alignment(self, student, teacher) -> torch.Tensor:
        """``[P, 3]`` of ``(row, teacher index, student index)`` for EMO.

        Indices into each row's *valid prefix*, which is what both EMO terms slice
        their sequences down to.
        """
        return alignment_index_tensor(
            align_batch(
                teacher["input_ids"],
                student["input_ids"],
                teacher["attention_mask"].sum(dim=1).tolist(),
                student["attention_mask"].sum(dim=1).tolist(),
                self.tt,
                self.ts,
                teacher_special=self.teacher_token,
                student_special=self.student_token,
            )
        )

    def _add_alignment(self, out: dict, student, teacher, side: int) -> None:
        if self.alignment == "cdm":
            # Only the first sentence of a row carries a CDM token term.
            if side == 1:
                out["cdm_align1"] = self._cdm_alignment(student, teacher)
        elif self.alignment == "emo":
            out[f"emo_align{side}"] = self._emo_alignment(student, teacher)

    def __call__(self, batch: list[tuple[str, str | None, float | None]]):
        text1s = [sample[0] for sample in batch]
        text2s = [sample[1] for sample in batch]
        labels = [sample[2] for sample in batch]

        student1 = self._encode(self.ts, text1s)
        out = {}
        self._add_encoding(out, student1, 1, "stu")
        if self.tt is not None:
            teacher1 = self._encode(self.tt, text1s)
            self._add_encoding(out, teacher1, 1, "tea")
            self._add_alignment(out, student1, teacher1, 1)

        has_second_text = all(text is not None for text in text2s)
        if any(text is not None for text in text2s) and not has_second_text:
            raise ValueError("A batch cannot mix single-text and pair-text samples")
        if has_second_text:
            pair_texts = [str(text) for text in text2s]
            student2 = self._encode(self.ts, pair_texts)
            self._add_encoding(out, student2, 2, "stu")
            if self.tt is not None:
                teacher2 = self._encode(self.tt, pair_texts)
                self._add_encoding(out, teacher2, 2, "tea")
                self._add_alignment(out, student2, teacher2, 2)

        has_labels = all(label is not None for label in labels)
        if any(label is not None for label in labels) and not has_labels:
            raise ValueError("A batch cannot mix labeled and unlabeled samples")
        if has_labels:
            dtype = torch.float32 if self.task == "pair_reg" else torch.long
            out["labels"] = torch.tensor(labels, dtype=dtype)
        return out
