"""EMO: attention-CKA plus optimal-transport distillation across two tokenizers.

The teacher and the student tokenize differently, so both terms first align the
two token sequences by minimum edit distance (:func:`align_tokens`). The
attention term then matches the maps of the aligned tokens through CKA, and the
OT term transports the student states onto the projected teacher states, with
the marginals set by the teacher's own attention mass.

Three things about *how* that is computed, none of which changes what it is:

``align_tokens`` runs once per row and side
    Both terms want the same alignment of the same two token sequences, and used
    to derive it separately -- twice per row, per side, per epoch. It is a
    function of the token ids alone, so :func:`align_batch` builds it once, and
    the collate can build it in a DataLoader worker before the step needs it.

the transport is solved for the whole batch at once
    :func:`sinkhorn_batched` replaces the per-sample loop. The rows of a batch
    never interact, so this is the same transport; what it drops is the device
    synchronisation the per-sample convergence test paid on every iteration.

only the teacher attention maps that are read get moved
    :func:`teacher_attention_layers` says which three of the teacher's 28 (or 36)
    maps the loss touches. On a two-GPU run the rest was a multi-gigabyte
    cross-device copy per step that nothing read.
"""

import math
from collections.abc import Sequence

import editdistance
import torch
from torch import nn

from src.metrics import scalar_metrics


class CKALoss(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, SH, TH):
        dT = TH.size(-1)
        dS = SH.size(-1)
        SH = SH.view(-1, dS).to(SH.device, torch.float64)
        TH = TH.view(-1, dT).to(SH.device, torch.float64)

        slen = SH.size(0)
        SH = SH - SH.mean(0, keepdim=True)
        TH = TH - TH.mean(0, keepdim=True)

        num = torch.norm(SH.t().matmul(TH), "fro")
        den1 = torch.norm(SH.t().matmul(SH), "fro") + self.eps
        den2 = torch.norm(TH.t().matmul(TH), "fro") + self.eps

        return 1 - num / torch.sqrt(den1 * den2)


def compute_token_importance(attention_weights, tokens):
    """Normalised attention mass each token receives, averaged over heads."""
    # 3D means [heads, seq, seq]; 2D is an already-averaged attention matrix.
    if len(attention_weights.shape) == 3:
        # Average attention across heads: [seq_len, seq_len]
        avg_attention = attention_weights.mean(dim=0)
    else:
        # Already a 2D attention matrix
        avg_attention = attention_weights

    # Ensure dimensions match
    seq_len = min(avg_attention.shape[0], len(tokens))

    # Truncate attention matrix if needed
    avg_attention = avg_attention[:seq_len, :seq_len]

    # Sum attention that each token receives: [seq_len]
    token_importance = avg_attention.sum(dim=0)

    token_importance = token_importance.clamp_min(0)
    total = token_importance.sum()
    if total <= 1e-12:
        norm_importance = torch.full_like(token_importance, 1.0 / max(seq_len, 1))
    else:
        norm_importance = token_importance / total

    return norm_importance


def project_importance(teacher_importance, student_tokens, mapping):
    device = teacher_importance.device
    if len(student_tokens) == 0:
        return torch.empty(0, device=device)
    min_importance = (
        teacher_importance.min()
        if teacher_importance.numel() > 0
        else torch.tensor(1.0, device=device)
    )
    student_importance = torch.full(
        (len(student_tokens),),
        float(min_importance.item()),
        device=device,
        dtype=teacher_importance.dtype,
    )
    for teacher_idx, student_idx in mapping.items():
        if teacher_idx < teacher_importance.numel() and student_idx < len(
            student_tokens
        ):
            student_importance[student_idx] = teacher_importance[teacher_idx]
    total = student_importance.sum()
    if total <= 1e-12:
        return torch.full_like(student_importance, 1.0 / len(student_tokens))
    return student_importance / total


def sinkhorn(
    cost_matrix: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float = 0.1,
    max_iter: int = 100,
    stop_thr: float = 1e-9,
    eps: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sinkhorn Optimal Transport for 2D cost matrix (single batch item).
    cost_matrix: [m, n]
    a: [m, 1] or [m] - source marginal
    b: [n, 1] or [n] - target marginal
    """
    m, n = cost_matrix.shape
    device = cost_matrix.device
    dtype = cost_matrix.dtype

    if m == 0 or n == 0:
        return (
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.zeros((m, n), device=device, dtype=dtype),
        )

    a = a.to(device=device, dtype=dtype)
    b = b.to(device=device, dtype=dtype)

    # Ensure correct shape
    if a.dim() == 1:
        a = a.view(-1, 1)
    if b.dim() == 1:
        b = b.view(-1, 1)

    # Fallback to uniform if dimensions don't match
    if a.shape[0] != m:
        a = torch.ones(m, 1, device=device, dtype=dtype) / m
    if b.shape[0] != n:
        b = torch.ones(n, 1, device=device, dtype=dtype) / n

    # Normalize marginals
    if torch.sum(a) < eps or torch.sum(b) < eps:
        a = torch.ones(m, 1, device=device, dtype=dtype) / m
        b = torch.ones(n, 1, device=device, dtype=dtype) / n
    else:
        a = a / torch.sum(a)
        b = b / torch.sum(b)

    # Sinkhorn iterations
    K = torch.exp(-cost_matrix / alpha)
    u = torch.ones(m, 1, device=device, dtype=dtype)
    v = torch.ones(n, 1, device=device, dtype=dtype)

    for _ in range(max_iter):
        u_prev = u.clone()
        KTu = torch.matmul(K.t(), u)  # [n, 1]
        v = b / (KTu + eps)
        Kv = torch.matmul(K, v)  # [m, 1]
        u = a / (Kv + eps)

        # Check convergence
        err = torch.norm(u - u_prev, p=float("inf"))
        if err < stop_thr:
            break

    # Compute transport matrix
    P = torch.diag(u.squeeze()) @ K @ torch.diag(v.squeeze())  # [m, n]

    # Compute OT loss
    ot_loss = torch.sum(P * cost_matrix)

    return ot_loss, P


def pairwise_attention_distance(x, y, eps=1e-8):
    d = x.shape[1]
    sim_mt = torch.mm(x, y.transpose(0, 1)) / math.sqrt(d)
    attention_weights = torch.softmax(sim_mt, dim=1)
    dist_mt = 1.0 - attention_weights
    return dist_mt


def batched_cka(
    student: torch.Tensor,
    teacher: torch.Tensor,
    rows_valid: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """:class:`CKALoss` for a batch of ``[B, N, d]`` pairs, padded rows masked out.

    The per-row version is a chain of about ten tiny kernels over operands a few
    tokens wide, and the attention term ran one such chain per row per matched
    layer -- 512 of them per training step at batch 128. None of that is
    arithmetic the GPU notices; it is launch latency. Here each of the three Gram
    matrices is one batched matmul.

    Padded *rows* are zeroed after a masked centring, so they contribute nothing to
    any of the three norms; padded *feature columns* are all-zero columns of the
    Gram matrices, which the Frobenius norm does not see either. The result is
    therefore what running each row on its own valid slice would give.
    """
    student = student.to(torch.float64)
    teacher = teacher.to(device=student.device, dtype=torch.float64)
    mask = rows_valid.unsqueeze(-1).to(student.dtype)
    counts = rows_valid.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)

    student = (student - (student * mask).sum(dim=1, keepdim=True) / counts) * mask
    teacher = (teacher - (teacher * mask).sum(dim=1, keepdim=True) / counts) * mask

    numerator = torch.linalg.matrix_norm(student.transpose(1, 2) @ teacher)
    student_gram = torch.linalg.matrix_norm(student.transpose(1, 2) @ student) + eps
    teacher_gram = torch.linalg.matrix_norm(teacher.transpose(1, 2) @ teacher) + eps
    return 1 - numerator / torch.sqrt(student_gram * teacher_gram)


def teacher_attention_layers(
    teacher_layer_num: int, student_layer_num: int, k: int
) -> list[int]:
    """Indices of the teacher attention maps the EMO loss actually reads.

    The attention term matches the last ``k`` of the teacher layers the uniform
    block mapping assigns to the student's layers, and the token importances are
    read off the teacher's final layer. Three maps out of 28 for Qwen3-0.6B,
    three out of 36 for Qwen3-4B; the rest is computed by the teacher either way
    but never has to leave the device it was computed on.
    """
    layers_per_block = teacher_layer_num // student_layer_num
    mapped = [
        index * layers_per_block + layers_per_block - 1
        for index in range(student_layer_num)
    ]
    needed = {index % teacher_layer_num for index in mapped[-k:]}
    needed.add(teacher_layer_num - 1)
    return sorted(needed)


def _as_layer_map(
    attentions, teacher_layer_num: int | None = None
) -> tuple[dict[int, torch.Tensor], int]:
    """Accept either the full attention tuple or a sparse ``{index: map}`` dict."""
    if isinstance(attentions, dict):
        if teacher_layer_num is None:
            raise ValueError(
                "teacher_layer_num is required when the teacher attentions are "
                "given as a sparse {layer index: map} dict"
            )
        return attentions, int(teacher_layer_num)
    layers = list(attentions)
    return dict(enumerate(layers)), len(layers)


def valid_prefix_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """``[B, L]`` bool over the first ``mask.sum()`` positions of each row.

    The per-sample code sliced ``ids[:L]`` with ``L = mask.sum()``. That is the
    attention mask itself under right padding and a prefix of it otherwise, so
    building it this way keeps the batched terms reading exactly what the loop
    read.
    """
    lengths = attention_mask.sum(dim=1, keepdim=True)
    positions = torch.arange(
        attention_mask.size(1), device=attention_mask.device
    ).unsqueeze(0)
    return positions < lengths


def batched_token_importance(
    attention_last: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """:func:`compute_token_importance` for a whole ``[B, H, L, L]`` batch.

    Padded rows and columns are zeroed, so a padded position neither receives
    mass nor contributes any, and a row whose mass vanishes falls back to the
    uniform distribution over its valid tokens.
    """
    average = attention_last.mean(dim=1)  # [B, L, L]
    keep = valid.unsqueeze(1) & valid.unsqueeze(2)
    importance = (average * keep).sum(dim=1).clamp_min(0) * valid
    total = importance.sum(dim=1, keepdim=True)
    counts = valid.sum(dim=1, keepdim=True).clamp_min(1)
    uniform = valid.to(importance.dtype) / counts
    return torch.where(total > 1e-12, importance / total.clamp_min(1e-12), uniform)


def batched_project_importance(
    teacher_importance: torch.Tensor,
    teacher_valid: torch.Tensor,
    student_valid: torch.Tensor,
    alignment: torch.Tensor,
) -> torch.Tensor:
    """:func:`project_importance` for the whole batch, as one scatter.

    ``alignment`` is ``[P, 3]`` of ``(row, teacher index, student index)``.
    Student positions no teacher token was aligned to keep the teacher's smallest
    mass, which is what the per-sample version filled them with.
    """
    floor = teacher_importance.masked_fill(~teacher_valid, float("inf")).amin(dim=1)
    floor = torch.where(torch.isfinite(floor), floor, torch.ones_like(floor))
    student_importance = floor.unsqueeze(1).expand_as(student_valid).clone()
    student_importance = student_importance * student_valid
    if alignment.numel():
        rows = alignment[:, 0]
        student_importance[rows, alignment[:, 2]] = teacher_importance[
            rows, alignment[:, 1]
        ]
    total = student_importance.sum(dim=1, keepdim=True)
    counts = student_valid.sum(dim=1, keepdim=True).clamp_min(1)
    uniform = student_valid.to(student_importance.dtype) / counts
    return torch.where(
        total > 1e-12, student_importance / total.clamp_min(1e-12), uniform
    )


def _normalised_marginal(
    mass: torch.Tensor, valid: torch.Tensor, eps: float
) -> torch.Tensor:
    """A ``[B, L]`` marginal summing to one over the valid positions of each row."""
    mass = mass.masked_fill(~valid, 0.0)
    total = mass.sum(dim=1, keepdim=True)
    counts = valid.sum(dim=1, keepdim=True).clamp_min(1)
    uniform = valid.to(mass.dtype) / counts
    return torch.where(total < eps, uniform, mass / total.clamp_min(eps))


def sinkhorn_batched(
    cost_matrix: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    valid_a: torch.Tensor,
    valid_b: torch.Tensor,
    alpha: float = 0.1,
    max_iter: int = 100,
    stop_thr: float = 1e-9,
    eps: float = 1e-9,
    check_every: int = 10,
) -> torch.Tensor:
    """Entropic OT for a batch of ``[B, m, n]`` cost matrices, as one solve.

    The rows of a batch never interact, so this is :func:`sinkhorn` run once per
    row -- but it is the only version that can be afforded on the critical path.
    The per-sample one tests ``if err < stop_thr`` on a device scalar every
    iteration, and that test is a device synchronisation: a batch of 128
    sentences, two sides of a pair, 100 iterations is 25,600 stalls per training
    step. Here the same test is one reduction over the batch, taken every
    ``check_every`` iterations, so a step pays ten. A row that has converged
    stops moving, so running it alongside a row that has not costs it nothing.

    Padded positions carry zero mass under a zero kernel: they can neither send
    nor receive. Returns the transport cost of each row.
    """
    batch, m, n = cost_matrix.shape
    dtype, device = cost_matrix.dtype, cost_matrix.device
    if m == 0 or n == 0:
        return torch.zeros(batch, device=device, dtype=dtype)

    keep = valid_a.unsqueeze(2) & valid_b.unsqueeze(1)
    cost_matrix = cost_matrix * keep

    a = _normalised_marginal(a, valid_a, eps).unsqueeze(2)
    b = _normalised_marginal(b, valid_b, eps).unsqueeze(2)

    kernel = torch.exp(-cost_matrix / alpha) * keep
    u = valid_a.to(dtype).unsqueeze(2)
    v = torch.ones(batch, n, 1, device=device, dtype=dtype)

    for iteration in range(max_iter):
        previous = u
        v = b / (torch.bmm(kernel.transpose(1, 2), u) + eps)
        u = a / (torch.bmm(kernel, v) + eps)
        if (iteration + 1) % check_every == 0 and (
            u - previous
        ).abs().amax() < stop_thr:
            break

    plan = u * kernel * v.transpose(1, 2)
    return (plan * cost_matrix).sum(dim=(1, 2))


def batched_attention_distance(
    student: torch.Tensor, teacher: torch.Tensor, valid_teacher: torch.Tensor
) -> torch.Tensor:
    """:func:`pairwise_attention_distance` over a padded batch.

    The softmax runs over the teacher's valid tokens only, which the per-sample
    version got for free by never materialising the padding.
    """
    scores = torch.bmm(student, teacher.transpose(1, 2)) / math.sqrt(student.size(-1))
    scores = scores.masked_fill(~valid_teacher.unsqueeze(1), float("-inf"))
    return torch.nan_to_num(1.0 - torch.softmax(scores, dim=-1), nan=0.0)


# Normalised edit distances between token strings. The MinED alignment's inner
# loop asks for one per cell, and a corpus reuses its vocabulary on every row, so
# the table turns an ``editdistance.eval`` call per cell into a dict lookup.
# Bounded, and the entries are short strings.
_SUBSTITUTION_CACHE: dict[tuple[str, str], float] = {}
_SUBSTITUTION_CACHE_MAX = 1 << 20

# The three moves of the alignment DP, in the order ``min`` used to break ties on.
_MATCH, _DELETE, _INSERT = 0, 1, 2


def _substitution_cost(teacher_token: str, student_token: str) -> float:
    key = (teacher_token, student_token)
    cost = _SUBSTITUTION_CACHE.get(key)
    if cost is None:
        denominator = max(len(teacher_token), len(student_token), 1)
        cost = editdistance.eval(teacher_token, student_token) / denominator
        if len(_SUBSTITUTION_CACHE) < _SUBSTITUTION_CACHE_MAX:
            _SUBSTITUTION_CACHE[key] = cost
    return cost


def _normalize_alignment_token(token: str) -> str:
    return token.replace("##", "").lstrip("Ġ▁").lower()


def align_tokens(
    teacher_tokens,
    student_tokens,
    teacher_special="<s>",
    student_special="[CLS]",
):
    """Order-preserving one-to-one MinED alignment over token strings.

    The returned mapping is index based, so repeated token strings remain
    distinct and cannot overwrite each other.
    """

    if not teacher_tokens or not student_tokens:
        return {}
    teacher_normalized = [_normalize_alignment_token(token) for token in teacher_tokens]
    student_normalized = [_normalize_alignment_token(token) for token in student_tokens]
    teacher_count = len(teacher_tokens)
    student_count = len(student_tokens)
    dp = [[0.0] * (student_count + 1) for _ in range(teacher_count + 1)]
    backtrace = [[None] * (student_count + 1) for _ in range(teacher_count + 1)]
    for teacher_idx in range(1, teacher_count + 1):
        dp[teacher_idx][0] = float(teacher_idx)
        backtrace[teacher_idx][0] = _DELETE
    for student_idx in range(1, student_count + 1):
        dp[0][student_idx] = float(student_idx)
        backtrace[0][student_idx] = _INSERT

    for teacher_idx in range(1, teacher_count + 1):
        teacher_token = teacher_normalized[teacher_idx - 1]
        previous, current = dp[teacher_idx - 1], dp[teacher_idx]
        moves = backtrace[teacher_idx]
        for student_idx in range(1, student_count + 1):
            # Strict ``<``, so a tie keeps the first of (match, delete, insert) --
            # which is what ``min(..., key=...)`` over that tuple returned.
            best = previous[student_idx - 1] + _substitution_cost(
                teacher_token, student_normalized[student_idx - 1]
            )
            move = _MATCH
            deletion = previous[student_idx] + 1.0
            if deletion < best:
                best, move = deletion, _DELETE
            insertion = current[student_idx - 1] + 1.0
            if insertion < best:
                best, move = insertion, _INSERT
            current[student_idx], moves[student_idx] = best, move

    mapping = {}
    teacher_idx = teacher_count
    student_idx = student_count
    while teacher_idx > 0 or student_idx > 0:
        operation = backtrace[teacher_idx][student_idx]
        if operation == _MATCH:
            source_idx = teacher_idx - 1
            target_idx = student_idx - 1
            if (
                teacher_tokens[source_idx] != teacher_special
                and student_tokens[target_idx] != student_special
            ):
                mapping[source_idx] = target_idx
            teacher_idx -= 1
            student_idx -= 1
        elif operation == _DELETE:
            teacher_idx -= 1
        else:
            student_idx -= 1
    return dict(sorted(mapping.items()))


def align_batch(
    input_ids_tea: torch.Tensor,
    input_ids_stu: torch.Tensor,
    lengths_tea: Sequence[int],
    lengths_stu: Sequence[int],
    tok_teacher,
    tok_student,
    teacher_special: str = "<s>",
    student_special: str = "[CLS]",
) -> list[list[tuple[int, int]]]:
    """:func:`align_tokens` for every row of a batch, once.

    Returns, per row, the ``(teacher index, student index)`` pairs in ascending
    teacher order -- the order ``align_tokens``' sorted mapping iterates in, which
    is what the attention term's stable ranking sort relies on.

    Both EMO terms read this and nothing about it depends on the weights, so the
    collate calls it in a DataLoader worker and the training step is handed the
    answer. ``input_ids_*`` are read on the host, so pass CPU tensors.
    """
    alignments = []
    for row in range(input_ids_stu.size(0)):
        length_t = int(lengths_tea[row])
        length_s = int(lengths_stu[row])
        if length_t == 0 or length_s == 0:
            alignments.append([])
            continue
        teacher_tokens = tok_teacher.convert_ids_to_tokens(
            input_ids_tea[row, :length_t].tolist()
        )
        student_tokens = tok_student.convert_ids_to_tokens(
            input_ids_stu[row, :length_s].tolist()
        )
        mapping = align_tokens(
            teacher_tokens,
            student_tokens,
            teacher_special=teacher_special,
            student_special=student_special,
        )
        alignments.append(list(mapping.items()))
    return alignments


def alignment_index_tensor(
    alignments: Sequence[Sequence[tuple[int, int]]],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """``[P, 3]`` of ``(row, teacher index, student index)`` over a whole batch."""
    rows = [
        (row, teacher_index, student_index)
        for row, pairs in enumerate(alignments)
        for teacher_index, student_index in pairs
    ]
    if not rows:
        return torch.zeros((0, 3), dtype=torch.long, device=device)
    return torch.tensor(rows, dtype=torch.long, device=device)


def group_alignment_by_row(
    alignment: torch.Tensor, batch_size: int
) -> list[list[tuple[int, int]]]:
    """The inverse of :func:`alignment_index_tensor`, on the host."""
    grouped: list[list[tuple[int, int]]] = [[] for _ in range(batch_size)]
    for row, teacher_index, student_index in alignment.tolist():
        grouped[row].append((teacher_index, student_index))
    return grouped


def compute_att_loss_2(
    teacher_atts,  # {layer index: [B, H, L_t, L_t]} on the student device
    student_atts,  # list of [B, H, L_s, L_s]
    input_ids_tea,  # [B, L_t]
    input_ids_stu,  # [B, L_s]
    attention_mask_tea,  # [B, L_t]
    attention_mask_stu,  # [B, L_s]
    tok_teacher,
    tok_student,
    k,  # how many of the last layers to match (usually 1)
    device,
    teacher_special="<s>",
    student_special="[CLS]",
    *,
    teacher_layer_num=None,
    alignments=None,  # per row, (teacher index, student index) pairs
    teacher_importance=None,  # [B, L_t] on the host, from batched_token_importance
    lengths_tea=None,  # host lengths, so the loop reads no device scalar
    lengths_stu=None,
):

    batch_size = input_ids_stu.size(0)
    layer_map, teacher_layer_num = _as_layer_map(teacher_atts, teacher_layer_num)
    student_layer_num = len(student_atts)
    layers_per_block = teacher_layer_num // student_layer_num
    mapped = [
        index * layers_per_block + layers_per_block - 1
        for index in range(student_layer_num)
    ]

    teacher_last_k_layers = [
        layer_map[index % teacher_layer_num] for index in mapped[-k:]
    ]
    student_last_k_layers = student_atts[-k:]
    last_teacher_att = layer_map[teacher_layer_num - 1]

    if lengths_tea is None:
        lengths_tea = attention_mask_tea.sum(dim=1).tolist()
    if lengths_stu is None:
        lengths_stu = attention_mask_stu.sum(dim=1).tolist()
    if teacher_importance is None:
        teacher_importance = batched_token_importance(
            last_teacher_att, valid_prefix_mask(attention_mask_tea)
        ).tolist()
    if alignments is None:
        alignments = align_batch(
            input_ids_tea.cpu(),
            input_ids_stu.cpu(),
            lengths_tea,
            lengths_stu,
            tok_teacher,
            tok_student,
            teacher_special,
            student_special,
        )

    # Which tokens of each row the term reads, and in what order: the aligned
    # pairs ranked by the teacher's attention mass, top third. Host-side, over
    # about twenty pairs a row, and it decides indices rather than values.
    teacher_index, student_index, rows_valid, contributes = _ranked_alignment_index(
        alignments, teacher_importance, lengths_tea, lengths_stu, device
    )
    if not bool(contributes.any()):
        # A zero that still carries a graph, so the step is differentiable even in
        # the degenerate case where no row aligned anything.
        return student_atts[-1][0, 0, 0, 0] * 0.0

    valid_teacher = valid_prefix_mask(attention_mask_tea).to(device)
    valid_student = valid_prefix_mask(attention_mask_stu).to(device)
    weights = contributes.to(torch.float64)

    def aligned_attention(layer, index, valid):
        """``[B, N, L]``: the attention of the N ranked tokens, head-averaged.

        The rows are picked before the heads are averaged, which is both the order
        the per-sample loop used and the cheap one: N is a third of the aligned
        tokens, some twenty times fewer rows than the full map has.
        """
        rows, heads, length = layer.size(0), layer.size(1), layer.size(-1)
        picked = layer.gather(
            2, index[:, None, :, None].expand(rows, heads, index.size(1), length)
        )  # [B, H, N, L]
        rows = picked.mean(dim=1)
        # Very negative entries are masked-out positions, not attention; padded
        # columns are not attention either.
        rows = torch.where(rows <= -1e2, torch.zeros_like(rows), rows)
        return rows * valid.unsqueeze(1)

    att_loss_total = None
    for teacher_att_layer, student_att_layer in zip(
        teacher_last_k_layers, student_last_k_layers
    ):
        per_row = batched_cka(
            aligned_attention(student_att_layer, student_index, valid_student),
            aligned_attention(teacher_att_layer, teacher_index, valid_teacher),
            rows_valid,
        )
        layer_total = (per_row * weights).sum()
        att_loss_total = (
            layer_total if att_loss_total is None else att_loss_total + layer_total
        )

    att_loss_terms = float(len(teacher_last_k_layers)) * float(weights.sum())
    if att_loss_total is None or att_loss_terms == 0:
        return student_atts[-1][0, 0, 0, 0] * 0.0
    # float64, as CKALoss has always returned: the per-sample loop promoted its
    # float32 accumulator on the first addition, and the backward ran in float64
    # from there. Keeping that keeps the term bit-for-bit what it was.
    return att_loss_total / att_loss_terms


def _ranked_alignment_index(
    alignments, teacher_importance, lengths_tea, lengths_stu, device
):
    """The aligned tokens each row contributes, as padded index tensors.

    Per row: the aligned pairs ranked by the teacher's attention mass, keeping the
    top ``n_map // 3`` (at least one) -- the selection the per-sample loop made,
    with the sort still stable so ties keep ascending teacher order.

    Returns ``(teacher index, student index, row mask, contributing rows)``; the
    first three are ``[B, N_max]`` and the last is ``[B]``.
    """
    ranked_rows = []
    for row, pairs in enumerate(alignments):
        if not pairs or int(lengths_tea[row]) == 0 or int(lengths_stu[row]) == 0:
            ranked_rows.append([])
            continue
        importance = teacher_importance[row]
        ranked_rows.append(
            sorted(pairs, key=lambda pair: importance[pair[0]], reverse=True)[
                : max(1, len(pairs) // 3)
            ]
        )

    batch_size = len(ranked_rows)
    width = max((len(pairs) for pairs in ranked_rows), default=0) or 1
    teacher_index = torch.zeros(batch_size, width, dtype=torch.long)
    student_index = torch.zeros(batch_size, width, dtype=torch.long)
    rows_valid = torch.zeros(batch_size, width, dtype=torch.bool)
    for row, pairs in enumerate(ranked_rows):
        for slot, (teacher, student) in enumerate(pairs):
            teacher_index[row, slot] = teacher
            student_index[row, slot] = student
            rows_valid[row, slot] = True

    return (
        teacher_index.to(device),
        student_index.to(device),
        rows_valid.to(device),
        rows_valid.any(dim=1),
    )


def compute_ot_loss(
    teacher_last,  # [B, L_t, d_t], on the student device
    student_last,  # [B, L_s, d_s]
    teacher_att_last,  # [B, H, L_t, L_t], the last layer's attention
    attention_mask_teacher,  # [B, L_t]
    attention_mask_student,  # [B, L_s]
    input_ids_tea,  # [B, L_t]
    input_ids_stu,  # [B, L_s]
    tok_teacher,
    tok_student,
    projector,  # proj_t2s: Linear(d_t -> d_s)
    alpha: float = 0.1,
    max_iter: int = 100,
    teacher_special="<s>",
    student_special="[CLS]",
    *,
    alignment=None,  # [P, 3] of (row, teacher index, student index)
    teacher_importance=None,  # [B, L_t] device tensor
    valid_teacher=None,
    valid_student=None,
    lengths_tea=None,
    lengths_stu=None,
):
    """The OT term, solved for the whole batch in one Sinkhorn.

    Every quantity here is the per-sample one padded up to the batch: the same
    marginals, the same cost, the same transport. What is gone is the Python loop
    over the batch and the device read it took per sample per iteration.
    """
    device = teacher_last.device
    if valid_teacher is None:
        valid_teacher = valid_prefix_mask(attention_mask_teacher.to(device))
    if valid_student is None:
        valid_student = valid_prefix_mask(attention_mask_student.to(device))
    if teacher_importance is None:
        teacher_importance = batched_token_importance(teacher_att_last, valid_teacher)
    if alignment is None:
        if lengths_tea is None:
            lengths_tea = valid_teacher.sum(dim=1).tolist()
        if lengths_stu is None:
            lengths_stu = valid_student.sum(dim=1).tolist()
        alignment = alignment_index_tensor(
            align_batch(
                input_ids_tea.cpu(),
                input_ids_stu.cpu(),
                lengths_tea,
                lengths_stu,
                tok_teacher,
                tok_student,
                teacher_special,
                student_special,
            )
        )
    alignment = alignment.to(device)

    teacher_importance = teacher_importance.to(device=device, dtype=torch.float32)
    student_importance = batched_project_importance(
        teacher_importance, valid_teacher, valid_student, alignment
    )

    student_seq = student_last.to(torch.float32)
    projected_teacher_seq = projector(teacher_last).to(torch.float32)
    cost_matrix = batched_attention_distance(
        student_seq, projected_teacher_seq, valid_teacher
    )

    per_row = sinkhorn_batched(
        cost_matrix,
        student_importance,
        teacher_importance,
        valid_student,
        valid_teacher,
        alpha=alpha,
        max_iter=max_iter,
    )
    return per_row.sum() / student_last.size(0)


class EMODistillation(nn.Module):
    def __init__(
        self,
        d_teacher: int,
        d_student: int,
        k_layers: int = 2,
        alpha_ot: float = 0.1,
        max_iter: int = 100,
        teacher_special: str = "<s>",
        student_special: str = "[CLS]",
    ):
        super().__init__()

        self.k_layers = k_layers
        self.alpha_ot = alpha_ot
        self.max_iter = max_iter
        self.teacher_special = teacher_special
        self.student_special = student_special

        self.proj_t2s = nn.Linear(d_teacher, d_student, bias=False)

    def compute_emo_loss(
        self,
        teacher_outputs,
        student_outputs,
        input_ids_tea: torch.Tensor,
        input_ids_stu: torch.Tensor,
        attention_mask_tea: torch.Tensor,
        attention_mask_stu: torch.Tensor,
        tok_teacher,
        tok_student,
        att_loss_weight: float = 1.0,
        ot_loss_weight: float = 1.0,
        *,
        teacher_layer_num: int | None = None,
        alignment: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Both EMO terms over one side of a batch.

        The alignment, the row lengths and the teacher's token importances are
        wanted by both terms and are derived here once. ``alignment`` is the
        collate's ``[P, 3]`` index tensor when the run precomputes it; passing
        ``None`` falls back to deriving it here, which is what a test or a
        one-off call wants.
        """
        device = student_outputs.last_hidden_state.device
        layer_map, teacher_layer_num = _as_layer_map(
            teacher_outputs.attentions, teacher_layer_num
        )
        last_teacher_att = layer_map[teacher_layer_num - 1]

        valid_teacher = valid_prefix_mask(attention_mask_tea)
        valid_student = valid_prefix_mask(attention_mask_stu)
        batch_size = attention_mask_tea.size(0)
        # One device read for both sets of row lengths, and one for the
        # importances the attention term's ranking sorts on. The per-sample code
        # read a scalar back for each of those, per row, per term.
        lengths = torch.cat(
            [valid_teacher.sum(dim=1), valid_student.sum(dim=1)]
        ).tolist()
        lengths_tea, lengths_stu = lengths[:batch_size], lengths[batch_size:]

        teacher_importance = batched_token_importance(last_teacher_att, valid_teacher)

        if alignment is None:
            alignments = align_batch(
                input_ids_tea.cpu(),
                input_ids_stu.cpu(),
                lengths_tea,
                lengths_stu,
                tok_teacher,
                tok_student,
                teacher_special=self.teacher_special,
                student_special=self.student_special,
            )
            alignment = alignment_index_tensor(alignments)
        else:
            alignments = group_alignment_by_row(alignment, batch_size)

        att_loss = compute_att_loss_2(
            teacher_atts=layer_map,
            student_atts=list(student_outputs.attentions),
            input_ids_tea=input_ids_tea,
            input_ids_stu=input_ids_stu,
            attention_mask_tea=attention_mask_tea,
            attention_mask_stu=attention_mask_stu,
            tok_teacher=tok_teacher,
            tok_student=tok_student,
            k=self.k_layers,
            device=device,
            teacher_special=self.teacher_special,
            student_special=self.student_special,
            teacher_layer_num=teacher_layer_num,
            alignments=alignments,
            teacher_importance=teacher_importance.tolist(),
            lengths_tea=lengths_tea,
            lengths_stu=lengths_stu,
        )

        ot_loss = compute_ot_loss(
            teacher_last=teacher_outputs.last_hidden_state,
            student_last=student_outputs.last_hidden_state,
            teacher_att_last=last_teacher_att,
            attention_mask_teacher=attention_mask_tea,
            attention_mask_student=attention_mask_stu,
            input_ids_tea=input_ids_tea,
            input_ids_stu=input_ids_stu,
            tok_teacher=tok_teacher,
            tok_student=tok_student,
            projector=self.proj_t2s,
            alpha=self.alpha_ot,
            max_iter=self.max_iter,
            teacher_special=self.teacher_special,
            student_special=self.student_special,
            alignment=alignment,
            teacher_importance=teacher_importance,
            valid_teacher=valid_teacher,
            valid_student=valid_student,
            lengths_tea=lengths_tea,
            lengths_stu=lengths_stu,
        )

        total_loss = att_loss_weight * att_loss + ot_loss_weight * ot_loss
        loss_dict = scalar_metrics(
            att_loss=att_loss, ot_loss=ot_loss, total_kd=total_loss
        )

        return total_loss, loss_dict
