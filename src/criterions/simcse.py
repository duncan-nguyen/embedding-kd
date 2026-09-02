"""SimCSE-only: the student's contrastive objective with no teacher term.

Unsupervised SimCSE (Gao, Yao & Chen, EMNLP 2021) encodes the same sentence
twice under independent dropout masks and pulls the two views together against
the other sentences of the batch:

    L = -log exp(cos(h_i, h_i^+) / tau) / sum_j exp(cos(h_i, h_j^+) / tau)

Nothing here reads a teacher. It is the control row of the comparison: the
student, the corpus, the schedule and the pooling are the ones every distilled
row uses, and the only thing removed is the teacher signal, so whatever a
distillation method reports above this line is what its teacher term bought.

The same InfoNCE term also appears *inside* the other objectives as their task
loss, which is why the control is this loss rather than a masked-LM or
next-sentence objective: it isolates the teacher, not the training signal.

Gao et al. take the contrastive loss over an MLP projection of the pooled vector
rather than over the vector itself, and throw that MLP away at test time (the
``--mlp_only_train`` recipe of their release). ``mlp_head=True`` reproduces it:
the head is a parameter of this criterion, so it is trained by the same
optimizer and saved under ``criterion_state_dict``, never inside the encoder
weights the benchmarks load.
"""

from __future__ import annotations

import torch
from torch import nn

from src.loss import info_nce
from src.metrics import scalar_metrics


class SimCSEOnly(nn.Module):
    """In-batch contrastive loss between two views of the same sentences.

    Args:
        temperature: tau of the InfoNCE objective. Unsupervised SimCSE uses 0.05.
        hidden_size: width of the pooled student vector. Required by, and only
            read for, the projection head.
        mlp_head: train the contrastive term through Gao et al.'s
            ``Linear(d, d) + Tanh`` projection of the pooled vector.

    The two views are produced by the caller (a second forward pass under
    dropout, or the paired sentence of the row). Without the head the module
    holds no parameters at all; with it, the head is the only parameter, and it
    is discarded at inference, so the deployed model is the plain student
    encoder either way.
    """

    def __init__(
        self,
        temperature: float = 0.05,
        hidden_size: int | None = None,
        mlp_head: bool = False,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = float(temperature)

        self.mlp = None
        if mlp_head:
            if hidden_size is None or hidden_size <= 0:
                raise ValueError(
                    "mlp_head=True needs a positive hidden_size, got "
                    f"{hidden_size!r}"
                )
            # Gao et al.'s projection: square, biased, tanh. Square because the
            # head has to be droppable -- the encoder's own output width is what
            # the benchmarks read.
            self.mlp = nn.Sequential(
                nn.Linear(int(hidden_size), int(hidden_size)), nn.Tanh()
            )

    def forward(
        self, view1: torch.Tensor, view2: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Args:
        view1: pooled embeddings of the anchors, [B, d_S].
        view2: pooled embeddings of their positives, [B, d_S].

        The pooled vectors are what the caller passes and what inference uses;
        the head, when there is one, is applied here so that it sits between the
        encoder and the loss and nowhere else.
        """
        if view1.shape != view2.shape:
            raise ValueError(
                f"The two views must have the same shape, got "
                f"{tuple(view1.shape)} and {tuple(view2.shape)}"
            )

        if self.mlp is not None:
            view1 = self.mlp(view1)
            view2 = self.mlp(view2)

        loss, logits = info_nce(view1, view2, temperature=self.temperature)

        with torch.no_grad():
            # logits are cosines divided by tau; scale back so the two diagnostics
            # read on the [-1, 1] scale the benchmarks score on. With a head they
            # read in the head's space, which is the space the loss acts in, not
            # the one inference deploys.
            cosine = logits.float() * self.temperature
            positive = cosine.diagonal()
            off_diagonal = ~torch.eye(
                cosine.shape[0], dtype=torch.bool, device=cosine.device
            )
            targets = torch.arange(cosine.shape[0], device=cosine.device)
            metrics = scalar_metrics(
                loss_total=loss,
                # How often the positive view is the batch's nearest neighbour.
                inbatch_accuracy=(cosine.argmax(dim=-1) == targets).float().mean(),
                pos_cos=positive.mean(),
                # Watches for representation collapse, which this objective is the
                # one most exposed to: a negative cosine drifting towards pos_cos
                # means the encoder stopped separating sentences at all.
                neg_cos=cosine[off_diagonal].mean(),
            )

        return loss, metrics
