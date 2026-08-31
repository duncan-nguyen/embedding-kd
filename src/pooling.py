import torch
from torch import Tensor

# Pooling conventions of the teacher families this repo distils from. Decoder-style
# embedding models (Qwen3-Embedding) read the last token; encoder-style ones
# (BGE-M3, E5, MiniLM) read the CLS position or the masked mean.
POOLING_METHODS = ("last_token", "mean", "cls")


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def mean_pooling(last_hidden_state: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)  # sum over the length axis, [B, d]
    counts = mask.sum(dim=1).clamp(min=1e-9)        # [B, 1]
    return summed / counts


def pool_sentence_embedding(
    last_hidden_state: Tensor, attention_mask: Tensor, method: str
) -> Tensor:
    """Sentence vector of a teacher forward pass under one of ``POOLING_METHODS``.

    One dispatch shared by the cached-teacher methods (applied once at cache time)
    and the online ones (applied every step), so ``--teacher_pooling`` means the
    same thing for every method.
    """
    if method == "last_token":
        return last_token_pool(last_hidden_state, attention_mask)
    if method == "mean":
        return mean_pooling(last_hidden_state, attention_mask)
    if method == "cls":
        return last_hidden_state[:, 0, :]
    raise ValueError(
        f"Unknown pooling method: {method!r}; expected one of {POOLING_METHODS}"
    )
