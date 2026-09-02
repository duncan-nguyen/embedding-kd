"""The structural audit, read *during* training instead of after it.

:mod:`src.structural_audit` answers the questions the paper is about -- does the
student's geometry follow the teacher's, and at which rung does it stop following
-- but it answers them post hoc, from per-epoch weights re-encoded in a notebook.
That gives five points per run on curves whose shape is the finding, and it puts a
dump-and-re-encode cycle between a run and knowing what it did.

This module runs the same measurements on a fixed probe batch inside the training
loop. Three properties make that safe to switch on inside a seeded ablation:

* the probe encodes under ``torch.no_grad()`` with the student in ``eval()``, so no
  dropout mask is drawn and the RNG stream the next training step reads is
  untouched -- the trajectory of a seeded run is bit-identical with the probe on;
* the probe sentences are rows of the training corpus, so their teacher embeddings
  are already in the cache and the teacher -- which the cached methods free from
  the GPU before the first step -- never has to be run again;
* every rung above the first is invariant to an orthogonal rotation of either
  side and to the two spaces' widths, so the student is compared against the
  teacher *in the teacher's own* ``d_T``, not against the projected target. That
  is the comparison the audit is about: the target is one interface's opinion of
  the teacher, and measuring against it would grade the run on its own map.

Rung 1 is the exception and is reported against the target, because the target is
the only thing a coordinate-level cosine can be taken against at all.

:class:`WeightDriftTracker` answers the neighbouring question -- *which depths*
moved -- from the weights rather than from the embeddings, at the same cadence.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import torch

from src import structural_audit as audit

__all__ = ["TrainingProbe", "WeightDriftTracker"]

# `encoder.layer.11.attention...` (BERT), `layers.5.mlp...` (LLaMA-style), and the
# embedding table, which is depth 0 of neither but moves on its own account.
_LAYER_PATTERN = re.compile(r"(?:^|\.)layers?\.(\d+)\.")


class TrainingProbe:
    """A fixed batch of corpus sentences, re-measured on a stride.

    Args:
        texts: the probe sentences. A seeded sample of training-corpus rows, so
            their teacher embeddings are already cached.
        teacher: those rows of the teacher cache, ``[N, d_T]``, *unprojected*.
        targets: the same rows after ``P_T``, ``[N, d_S]`` -- what the run is
            actually trained against, and the only side rung 1 is defined against.
            ``None`` for a method that trains against the teacher's own space with
            no map in between, which simply has no rung 1 to report.
        tokenizer: the student tokenizer.
        pool: the distiller's own pooling, so the probe measures the vector the
            benchmarks score rather than a second opinion about it.
        knn_k: neighbourhood size of rung 3.
        seed: fixes the subsample of the TwoNN and anisotropy estimators, so their
            noise does not move between calls.

    The teacher side is fixed, so everything about it -- its neighbour table, its
    barcode -- is computed once here and reused by every call to :meth:`measure`.
    """

    def __init__(
        self,
        texts: Sequence[str],
        teacher: torch.Tensor,
        targets: torch.Tensor | None,
        tokenizer,
        pool: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        max_length: int = 256,
        batch_size: int = 128,
        knn_k: int = 10,
        seed: int = 0,
        anisotropy_pairs: int = 20_000,
    ):
        if len(texts) != teacher.shape[0]:
            raise ValueError(
                f"probe has {len(texts)} sentences but {teacher.shape[0]} teacher "
                "embeddings"
            )
        if targets is not None and len(texts) != targets.shape[0]:
            raise ValueError(
                f"probe has {len(texts)} sentences but {targets.shape[0]} targets"
            )
        if len(texts) <= knn_k:
            raise ValueError(
                f"a probe of {len(texts)} sentences cannot support k={knn_k} "
                "neighbours; raise probe_size or lower probe_knn_k"
            )
        self.texts = list(texts)
        self.teacher = teacher.detach().float().cpu()
        self.targets = None if targets is None else targets.detach().float().cpu()
        self.tokenizer = tokenizer
        self.pool = pool
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.knn_k = int(knn_k)
        self.seed = int(seed)
        self.anisotropy_pairs = int(anisotropy_pairs)
        # Tokenised once: the probe is the same sentences every time it runs.
        self._encoded = [
            tokenizer(
                self.texts[start : start + self.batch_size],
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            for start in range(0, len(self.texts), self.batch_size)
        ]
        self._teacher_neighbours = audit.knn_indices(self.teacher, self.knn_k)
        self._teacher_barcode = audit.mst_edge_weights(self.teacher)

    # ---------------------------------------------------------------- reference

    def reference(self) -> dict[str, float]:
        """What the frozen interface itself scores, before the student is involved.

        The ceiling of the run: the targets are the teacher seen through ``P_T``, so
        these numbers are the most any student trained against them could reach at
        each rung. A student number is only interesting read against its own row of
        this.
        """
        reference = {
            "probe_size": float(len(self.texts)),
            "probe_knn_k": float(self.knn_k),
            "teacher_effective_rank": audit.effective_rank(self.teacher),
            "teacher_anisotropy": audit.anisotropy(
                self.teacher, n_pairs=self.anisotropy_pairs, seed=self.seed
            ),
        }
        if self.targets is None:
            return reference
        reference.update(
            {
                "target_gram_rmse_teacher": audit.gram_rmse(self.targets, self.teacher),
                "target_gram_corr_teacher": audit.gram_correlation(
                    self.targets, self.teacher
                ),
                "target_cka_teacher": audit.linear_cka(self.targets, self.teacher),
                "target_knn_overlap_teacher": audit.knn_overlap(
                    self.targets,
                    self.teacher,
                    self.knn_k,
                    neighbours_b=self._teacher_neighbours,
                ),
                "target_h0_w1_teacher": audit.h0_barcode_distance(
                    self.targets, self.teacher
                ),
                "target_effective_rank": audit.effective_rank(self.targets),
            }
        )
        return reference

    # ------------------------------------------------------------------ measure

    @torch.no_grad()
    def encode(self, model, device) -> torch.Tensor:
        """The student's embedding of the probe, in ``eval()`` and without grad.

        ``eval()`` is what keeps a seeded run reproducible with the probe switched
        on: dropout draws nothing, so the generator the next training step reads is
        exactly where it would have been.
        """
        was_training = model.training
        model.eval()
        chunks = []
        for encoded in self._encoded:
            encoded = {key: value.to(device) for key, value in encoded.items()}
            out = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                return_dict=True,
            )
            pooled = self.pool(out.last_hidden_state, encoded["attention_mask"])
            chunks.append(pooled.float().cpu())
        model.train(was_training)
        return torch.cat(chunks, dim=0)

    def measure(self, model, device) -> dict[str, float]:
        """One full pass up the ladder, on the current weights.

        Rung 1 against the target (the only side a coordinate cosine exists
        against), rungs 2-4 and the spectrum against the teacher's own space.
        """
        student = self.encode(model, device)
        neighbours = audit.knn_indices(student, self.knn_k)
        measured = {
            # Rung 2: second-order structure, against the teacher in its own width.
            "probe_gram_rmse_teacher": audit.gram_rmse(student, self.teacher),
            "probe_gram_corr_teacher": audit.gram_correlation(student, self.teacher),
            "probe_cka_teacher": audit.linear_cka(student, self.teacher),
            # Rung 3: neighbourhoods.
            "probe_knn_overlap_teacher": audit.knn_overlap(
                student,
                self.teacher,
                self.knn_k,
                neighbours_a=neighbours,
                neighbours_b=self._teacher_neighbours,
            ),
            "probe_mutual_knn_teacher": audit.mutual_knn_jaccard(
                student,
                self.teacher,
                self.knn_k,
                neighbours_a=neighbours,
                neighbours_b=self._teacher_neighbours,
            ),
            # Rung 4: connectivity.
            "probe_h0_w1_teacher": float(
                audit.wasserstein_distance(
                    audit.mst_edge_weights(student), self._teacher_barcode
                )
            ),
            # Spectrum: is the student's own cloud spending its dimensions?
            "probe_effective_rank": audit.effective_rank(student),
            "probe_twonn": audit.twonn_intrinsic_dimension(student, seed=self.seed),
            "probe_anisotropy": audit.anisotropy(
                student, n_pairs=self.anisotropy_pairs, seed=self.seed
            ),
        }
        if self.targets is not None:
            cosine = audit.cosine_to_target(student, self.targets)
            measured.update(
                {
                    # Rung 1: coordinates. The only rung where the gauge shows, and
                    # the only one that needs the target rather than the teacher.
                    "probe_cos_target": float(cosine.mean()),
                    "probe_cos_target_std": float(cosine.std()),
                    # Rung 2 again against the target, where a Procrustes distance is
                    # defined because the two live in the same d_S.
                    "probe_gram_rmse_target": audit.gram_rmse(student, self.targets),
                    "probe_procrustes_target": audit.procrustes_distance(
                        student, self.targets
                    ),
                }
            )
        return measured


class WeightDriftTracker:
    """Per-depth relative weight movement, ``||W_l - W_l^0||_F / ||W_l^0||_F``.

    The depth question read from the weights rather than from the embeddings: with
    the endpoint the only supervised state, how far down the stack does the
    supervision actually reach? A layer whose drift stays at the noise floor while
    the layers above it move is a layer the objective never asked anything of, and
    that is a statement about the objective, not about the run.

    The reference copy is kept on the host in half precision. Relative drift is
    ``1e-2``-scale after a few hundred steps, so fp16's three decimal digits are
    two more than the reading needs, and the copy costs half of what the model does
    in memory the GPU is not being asked for.
    """

    def __init__(self, model: torch.nn.Module, dtype: torch.dtype = torch.float16):
        self.dtype = dtype
        self._reference: dict[str, list[torch.Tensor]] = {}
        self._norms: dict[str, float] = {}
        for name, parameter in model.named_parameters():
            group = self.group_of(name)
            self._reference.setdefault(group, []).append(
                parameter.detach().to("cpu", dtype).clone()
            )
        for group, tensors in self._reference.items():
            self._norms[group] = float(
                torch.linalg.vector_norm(
                    torch.stack([t.float().norm() for t in tensors])
                )
            )

    @property
    def groups(self) -> list[str]:
        """The depth groups the model's parameters fell into, in order."""
        return sorted(self._reference)

    @staticmethod
    def group_of(name: str) -> str:
        """Which depth a parameter belongs to: ``layer_<i>``, ``embeddings`` or ``other``."""
        match = _LAYER_PATTERN.search(name)
        if match:
            return f"layer_{int(match.group(1)):02d}"
        if "embed" in name:
            return "embeddings"
        return "other"

    @torch.no_grad()
    def measure(self, model: torch.nn.Module) -> dict[str, float]:
        """Relative Frobenius drift per depth, plus the whole model's."""
        current: dict[str, list[torch.Tensor]] = {}
        for name, parameter in model.named_parameters():
            current.setdefault(self.group_of(name), []).append(
                parameter.detach().to("cpu", self.dtype)
            )
        squared_total = 0.0
        squared_reference = 0.0
        out: dict[str, float] = {}
        for group, tensors in sorted(current.items()):
            reference = self._reference[group]
            if len(tensors) != len(reference):
                continue
            difference = float(
                torch.linalg.vector_norm(
                    torch.stack(
                        [(a.float() - b.float()).norm() for a, b in zip(tensors, reference)]
                    )
                )
            )
            base = self._norms[group]
            out[f"drift_{group}"] = difference / base if base else float("nan")
            squared_total += difference**2
            squared_reference += base**2
        out["drift_model"] = (
            (squared_total**0.5) / (squared_reference**0.5) if squared_reference else float("nan")
        )
        return out
