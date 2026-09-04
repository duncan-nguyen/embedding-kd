"""CDM: token-level distillation across two tokenizers, aligned by DTW.

The student and the teacher segment the same sentence differently, so the token
states cannot be compared position by position. DTW over the token *strings*
(edit distance as the local cost) gives a monotone alignment path, and only its
strict one-to-one, name-matching steps are kept -- everything ambiguous is
dropped rather than averaged, so the KD term never compares two tokens that are
not the same piece of text.

Nothing in that alignment depends on the weights: it is a function of the two
token *id* sequences alone, which a corpus row fixes for the whole run. So the
path is built by :func:`row_alignment` in the collate -- in a DataLoader worker,
in parallel with the previous step -- and the training step is handed the pairs
of positions as one index tensor. What is left on the critical path is a single
gather per side and a single MSE, instead of a Python loop over the batch that
re-derived the same alignment every epoch and read every sample back to the host
to do it.
"""

from collections.abc import Sequence

import editdistance
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# Edit distances between token strings, keyed by the pair of *normalised* strings.
# The DTW cost of a cell depends on nothing else, and a corpus reuses its vocabulary
# on every row, so the table is worth keeping: it turns the inner loop from an
# ``editdistance.eval`` call per cell into a dict lookup per cell. Bounded so a long
# run cannot grow it without limit; the entries are short strings.
_COST_CACHE: dict[tuple[str, str], float] = {}
_COST_CACHE_MAX = 1 << 20


def cost_fn(
    a: str,
    b: str,
    blending_model_special_token: str = "G",
    base_model_special_token: str = "##",
    specTok_mapper: dict | None = None,
) -> float:

    if specTok_mapper is None:
        specTok_mapper = {}

    if a in specTok_mapper and b in specTok_mapper.values():
        return 0.0
    if b in specTok_mapper and a in specTok_mapper.values():
        return 0.0

    aa = a.replace(blending_model_special_token, "").replace(" ", "")
    bb = b.replace(base_model_special_token, "").replace(" ", "")

    key = (aa, bb)
    dist = _COST_CACHE.get(key)
    if dist is None:
        dist = float(editdistance.eval(aa, bb))
        if len(_COST_CACHE) < _COST_CACHE_MAX:
            _COST_CACHE[key] = dist
    return dist


def _dtw_path(series_1, series_2, norm_func, series1_factor=None, series2_factor=None):
    """The DTW alignment path and its accumulated cost, on Python lists.

    Lists rather than a NumPy array because every access here is a scalar one:
    the recurrence reads three cells and writes one, and a NumPy scalar read
    boxes a value on the way out. The array only exists so ``debug_align`` can
    print a corner of it, so :func:`dtw` converts at the end and the alignment
    itself never pays for it.
    """
    rows, columns = len(series_1), len(series_2)
    infinity = float("inf")
    # One border of infinities, so the first real cell has no cheaper way in.
    matrix = [[infinity] * (columns + 1) for _ in range(rows + 1)]
    matrix[0][0] = 0.0

    scaled = series1_factor is not None and series2_factor is not None
    for i in range(rows):
        first, second = matrix[i], matrix[i + 1]
        value_1 = series_1[i]
        factor_1 = series1_factor[i] if scaled else 1.0
        for j in range(columns):
            cost = norm_func(value_1, series_2[j])
            if scaled:
                cost *= factor_1 * series2_factor[j]
            best = first[j]
            best = min(best, first[j + 1])
            best = min(best, second[j])
            second[j + 1] = cost + best

    matrix = [row[1:] for row in matrix[1:]]
    i, j = rows - 1, columns - 1
    matches = []
    while i > 0 or j > 0:
        matches.append((i, j))
        option_diag = matrix[i - 1][j - 1] if i > 0 and j > 0 else infinity
        option_up = matrix[i - 1][j] if i > 0 else infinity
        option_left = matrix[i][j - 1] if j > 0 else infinity
        # ``argmin`` returned the *first* smallest, so diagonal beats up beats left.
        if option_diag <= option_up and option_diag <= option_left:
            i -= 1
            j -= 1
        elif option_up <= option_left:
            i -= 1
        else:
            j -= 1

    matches.append((0, 0))
    matches.reverse()
    return matches, matrix


def dtw(
    series_1: list[str],
    series_2: list[str],
    series1_factor: list | None = None,
    series2_factor: list | None = None,
    norm_func=None,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Dynamic time warping over two token-string sequences.

    Returns the backtraced alignment path and the accumulated cost matrix (the
    latter only so ``debug_align`` can print a corner of it).
    """

    if norm_func is None:
        norm_func = cost_fn

    matches, matrix = _dtw_path(
        series_1, series_2, norm_func, series1_factor, series2_factor
    )
    return matches, np.array(matrix, dtype=float).reshape(len(series_1), len(series_2))


def _normalize_token(t: str, marker: str | None = None) -> str:
    markers = []
    if marker:
        markers.append(marker)
    markers += ["▁", "Ġ", "##"]

    for m in markers:
        t = t.replace(m, "")
    return t.lower()


def strict_one_to_one_pairs(
    path: Sequence[tuple[int, int]],
    base_tokens: list[str],
    blend_tokens: list[str],
    base_marker: str,
    blend_marker: str,
    specTok_mapper: dict | None = None,
    *,
    max_print: int = 0,
) -> tuple[list[tuple[int, int]], list, list]:
    """The steps of a DTW path that are kept: strictly one-to-one and same-named.

    Index-only, so it can run in a DataLoader worker with no tensors in sight; the
    tensor-gathering wrapper below is what the debug path still calls. The two
    reject lists are the debug material and stay empty unless ``max_print`` asks
    for them.
    """
    if specTok_mapper is None:
        specTok_mapper = {}

    base_counts, blend_counts = {}, {}
    for i, j in path:
        base_counts[i] = base_counts.get(i, 0) + 1
        blend_counts[j] = blend_counts.get(j, 0) + 1

    base_norm = [_normalize_token(t, base_marker) for t in base_tokens]
    blend_norm = [_normalize_token(t, blend_marker) for t in blend_tokens]

    specTok_mapper_rev = (
        {v: k for k, v in specTok_mapper.items()} if specTok_mapper else {}
    )

    def _is_special_pair_ok(b_tok: str, s_tok: str) -> bool:
        if b_tok in specTok_mapper and specTok_mapper[b_tok] == s_tok:
            return True
        if s_tok in specTok_mapper_rev and specTok_mapper_rev[s_tok] == b_tok:
            return True
        return False

    kept_pairs, name_mismatch, multi_align = [], [], []
    for i, j in path:
        if base_counts.get(i, 0) != 1 or blend_counts.get(j, 0) != 1:
            if len(multi_align) < max_print:
                multi_align.append(
                    (
                        i,
                        j,
                        base_tokens[i],
                        blend_tokens[j],
                        base_counts.get(i, 0),
                        blend_counts.get(j, 0),
                    )
                )
            continue

        bi_raw, sj_raw = base_tokens[i], blend_tokens[j]
        if _is_special_pair_ok(bi_raw, sj_raw) or (base_norm[i] == blend_norm[j]):
            kept_pairs.append((i, j))
        else:
            if len(name_mismatch) < max_print:
                name_mismatch.append(
                    (i, j, bi_raw, sj_raw, base_norm[i], blend_norm[j])
                )

    return kept_pairs, multi_align, name_mismatch


def row_alignment(
    teacher_tokens: list[str],
    student_tokens: list[str],
    teacher_marker: str,
    student_marker: str,
    specTok_mapper: dict | None = None,
) -> list[tuple[int, int]]:
    """``(teacher index, student index)`` pairs the CDM term compares, for one row.

    The whole of CDM's alignment: DTW over the token strings, then the strict
    one-to-one filter. It reads token strings and nothing else, so a row's answer
    is the same on every epoch and can be built anywhere -- which is why the
    collate builds it, in a worker, instead of the training step.

    The marker arguments keep the call the training step used to make: the teacher
    sequence is the DTW's first series and is normalised with ``student_marker``,
    the student sequence with ``teacher_marker``. ``_normalize_token`` strips the
    three common sub-word markers regardless, so the pairing only matters for a
    tokenizer whose marker is none of them.
    """
    if not teacher_tokens or not student_tokens:
        return []
    path, _ = _dtw_path(
        teacher_tokens,
        student_tokens,
        lambda a, b: cost_fn(a, b, teacher_marker, student_marker, specTok_mapper),
    )
    kept, _, _ = strict_one_to_one_pairs(
        path,
        base_tokens=teacher_tokens,
        blend_tokens=student_tokens,
        base_marker=student_marker,
        blend_marker=teacher_marker,
        specTok_mapper=specTok_mapper,
    )
    return kept


def build_special_token_mapper(tok_student, tok_teacher) -> dict[str, str]:
    """Student special token -> the teacher special token it stands for.

    Lifted off :class:`ContextualDynamicMapping` so the collate can build the same
    table without holding a criterion.
    """
    mapper: dict[str, str] = {}
    pairs = (
        (tok_student.cls_token, tok_teacher.bos_token),
        (tok_student.sep_token, tok_teacher.eos_token),
        (tok_student.pad_token, tok_teacher.pad_token),
        (tok_student.unk_token, tok_teacher.unk_token),
        (tok_student.mask_token, tok_teacher.mask_token),
    )
    for student_token, teacher_token in pairs:
        if student_token and teacher_token:
            mapper[student_token] = teacher_token
    return mapper


def align_strict_one_to_one(
    base_vals: torch.Tensor,
    blend_vals: torch.Tensor,
    path: Sequence[tuple[int, int]],
    base_tokens: list[str],
    blend_tokens: list[str],
    base_marker: str,
    blend_marker: str,
    specTok_mapper: dict | None = None,
    *,
    debug: bool = False,
    max_print: int = 20,
    dtw_matrix: np.ndarray | None = None,
    dtw_crop: int = 12,
) -> tuple[torch.Tensor, torch.Tensor]:
    kept_pairs, multi_align, name_mismatch = strict_one_to_one_pairs(
        path,
        base_tokens=base_tokens,
        blend_tokens=blend_tokens,
        base_marker=base_marker,
        blend_marker=blend_marker,
        specTok_mapper=specTok_mapper,
        max_print=max_print if debug else 0,
    )
    if debug:
        base_norm = [_normalize_token(t, base_marker) for t in base_tokens]
        blend_norm = [_normalize_token(t, blend_marker) for t in blend_tokens]

    if len(kept_pairs) == 0:
        A_base = base_vals.new_empty((0, base_vals.size(-1)))
        A_blend = blend_vals.new_empty((0, blend_vals.size(-1)))
    else:
        A_base = torch.stack([base_vals[i] for (i, j) in kept_pairs], dim=0)
        A_blend = torch.stack([blend_vals[j] for (i, j) in kept_pairs], dim=0)

    if debug:
        print("\n================= [ALIGN DEBUG] =================")
        print(
            f"L_base={base_vals.size(0)}, L_blend={blend_vals.size(0)}, |path|={len(path)}"
        )
        print(f"Final kept (strict name match + special map): {len(kept_pairs)}")

        if multi_align:
            print(f"\n[Examples dropped for multi-align] (show up to {max_print})")
            for i, j, braw, sraw, bc, sc in multi_align[:max_print]:
                print(
                    f"  (i={i}, j={j}) teacher='{braw}' student='{sraw}'  counts=({bc},{sc})"
                )

        if name_mismatch:
            print(
                f"\n[Examples dropped for name mismatch after normalize] (show up to {max_print})"
            )
            for i, j, braw, sraw, bn, sn in name_mismatch[:max_print]:
                print(
                    f"  (i={i}, j={j}) teacher='{braw}'→'{bn}'  vs  student='{sraw}'→'{sn}'"
                )

        if len(kept_pairs) > 0:
            print(f"\n[First kept pairs] (up to {max_print}):")
            for i, j in kept_pairs[:max_print]:
                print(
                    f"  (i={i}, j={j})  '{base_tokens[i]}' ↔ '{blend_tokens[j]}'  "
                    f"norm='{base_norm[i]}' ↔ '{blend_norm[j]}'"
                )

        print(
            f"\nAligned 1–1 shapes: A_t={tuple(A_base.shape)}, A_s={tuple(A_blend.shape)}"
        )

        if dtw_matrix is not None:
            H, W = dtw_matrix.shape
            h, w = min(dtw_crop, H), min(dtw_crop, W)
            print(f"\n[DTW matrix] shape={dtw_matrix.shape}  (show {h}x{w} top-left)")
            print(np.array2string(dtw_matrix[:h, :w], precision=2, suppress_small=True))
        print("=================================================\n")

    return A_base, A_blend


class ContextualDynamicMapping:
    def __init__(
        self,
        tok_student,
        tok_teacher,
        blending_model_special_token: str = "G",
        base_model_special_token: str = "##",
        w_task: float = 0.5,
        alpha_dtw: float = 0.5,
        debug_align: bool = False,
    ):
        self.tok_student = tok_student
        self.tok_teacher = tok_teacher
        self.blending_model_special_token = blending_model_special_token
        self.base_model_special_token = base_model_special_token
        self.w_task = w_task
        self.alpha_dtw = alpha_dtw
        self.debug_align = debug_align

        self.specTok_mapper = build_special_token_mapper(tok_student, tok_teacher)

    def aligned_token_loss(
        self,
        S_last: torch.Tensor,
        T_last: torch.Tensor,
        alignment: torch.Tensor,
        proj_s2t: nn.Module,
    ) -> torch.Tensor:
        """The CDM token term over pairs the collate already found.

        ``alignment`` is ``[P, 3]`` of ``(row, teacher position, student position)``
        over the whole batch, so both sides are one gather and the term is one MSE
        -- which is what the per-sample loop summed by hand. ``F.mse_loss`` reduces
        by the mean over elements, i.e. the loop's ``kd_sum / denom``.
        """
        base_dtype = S_last.dtype
        if alignment.numel() == 0:
            return torch.zeros((), device=S_last.device, dtype=base_dtype)
        rows = alignment[:, 0]
        A_t = T_last[rows, alignment[:, 1]]
        A_s = S_last[rows, alignment[:, 2]]
        S_proj_tok = F.normalize(proj_s2t(A_s).to(base_dtype), p=2, dim=-1)
        A_t = F.normalize(A_t.to(base_dtype), p=2, dim=-1)
        return F.mse_loss(S_proj_tok, A_t)

    def compute_cdm_loss(
        self,
        S_last: torch.Tensor,
        T_last: torch.Tensor,
        batch_input_ids_stu: torch.Tensor,
        batch_input_ids_tea: torch.Tensor,
        keep_mask_stu: torch.Tensor,
        keep_mask_tea: torch.Tensor,
        proj_s2t: nn.Module,
        device_s: torch.device,
        epoch: int = 0,
        step: int = 0,
    ) -> torch.Tensor:
        kd_sum, denom = 0.0, 0
        base_dtype = S_last.dtype
        Bsz = S_last.size(0)

        for i in range(Bsz):
            # Get tokens
            stu_tok_full = self.tok_student.convert_ids_to_tokens(
                batch_input_ids_stu[i].cpu().tolist(), skip_special_tokens=False
            )
            tea_tok_full = self.tok_teacher.convert_ids_to_tokens(
                batch_input_ids_tea[i].cpu().tolist(), skip_special_tokens=False
            )

            s_tok_i = [
                t
                for t, m in zip(stu_tok_full, keep_mask_stu[i].detach().cpu().tolist())
                if m
            ]
            t_tok_i = [
                t
                for t, m in zip(tea_tok_full, keep_mask_tea[i].detach().cpu().tolist())
                if m
            ]

            Si = S_last[i][keep_mask_stu[i]]  # [Ns_i, d_s]
            Ti = T_last[i][keep_mask_tea[i]]  # [Nt_i, d_t]

            if (
                Si.numel() > 0
                and Ti.numel() > 0
                and len(s_tok_i) > 0
                and len(t_tok_i) > 0
            ):
                matches, dtw_mat = dtw(
                    series_1=t_tok_i,
                    series_2=s_tok_i,
                    norm_func=lambda a, b: cost_fn(
                        a,
                        b,
                        self.blending_model_special_token,
                        self.base_model_special_token,
                        self.specTok_mapper,
                    ),
                )

                debug_here = (
                    self.debug_align and (epoch == 0) and (step < 1) and (i < 1)
                )

                A_t, A_s = align_strict_one_to_one(
                    base_vals=Ti,
                    blend_vals=Si,
                    base_tokens=t_tok_i,
                    blend_tokens=s_tok_i,
                    base_marker=self.base_model_special_token,
                    blend_marker=self.blending_model_special_token,
                    specTok_mapper=self.specTok_mapper,
                    path=matches,
                    debug=debug_here,
                    dtw_matrix=dtw_mat,
                    dtw_crop=12,
                )

                if A_t.size(0) > 0:
                    S_proj_tok = proj_s2t(A_s).to(base_dtype)
                    A_t = A_t.to(base_dtype)
                    S_proj_tok = F.normalize(S_proj_tok, p=2, dim=-1)
                    A_t = F.normalize(A_t, p=2, dim=-1)
                    kd_sum += F.mse_loss(S_proj_tok, A_t, reduction="sum")
                    denom += A_t.numel()
                    del S_proj_tok, A_t, A_s

        if denom == 0:
            return torch.tensor(0.0, device=device_s, dtype=base_dtype)
        else:
            loss = (kd_sum / denom).to(device=device_s, dtype=base_dtype)
            return loss
