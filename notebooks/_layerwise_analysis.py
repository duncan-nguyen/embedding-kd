"""Shared computations for the all-pairs layerwise-analysis notebook.

This module keeps the notebook readable while making the expensive geometry
calculations resumable through cached layer encodings and CSV matrices.
"""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


TEXT_COLUMNS = ("text", "sentence1", "sentence2", "premise", "hypothesis")


def _text_digest(texts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def _clean_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = " ".join(str(value).split())
    return text if text else None


def _texts_from_frame(frame: pd.DataFrame) -> list[str]:
    texts: list[str] = []
    for column in TEXT_COLUMNS:
        if column not in frame:
            continue
        for value in frame[column]:
            text = _clean_text(value)
            if text is not None:
                texts.append(text)
    return texts


def build_heldout_probe(
    project_dir: str | Path,
    train_data: str | Path,
    validation_files: Iterable[str | Path],
    *,
    size: int,
    seed: int = 0,
) -> tuple[list[str], pd.DataFrame]:
    """Sample validation sentences absent from the distillation corpus.

    The returned table records the source file of each sentence.  Duplicate
    strings are removed globally before sampling.
    """
    project_dir = Path(project_dir)
    train_frame = pd.read_csv(train_data)
    blocked = {text.casefold() for text in _texts_from_frame(train_frame)}

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for relative in validation_files:
        path = Path(relative)
        if not path.is_absolute():
            path = project_dir / path
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        for text in _texts_from_frame(frame):
            key = text.casefold()
            if key in blocked or key in seen:
                continue
            seen.add(key)
            rows.append({"text": text, "source": path.stem})

    if len(rows) < size:
        raise RuntimeError(
            f"Only {len(rows)} held-out validation texts remain after filtering; "
            f"cannot sample PROBE_SIZE={size}."
        )
    rng = np.random.default_rng(seed)
    # Keep the RNG order.  Sorting the sampled indices would cluster the probe by
    # source file, which would confound both the fixed H0 clouds and the disjoint
    # Procrustes fit/evaluation split.
    chosen = rng.choice(len(rows), size=size, replace=False)
    table = pd.DataFrame([rows[index] for index in chosen]).reset_index(drop=True)
    return table["text"].tolist(), table


def encode_layers_cached(
    *,
    model_name: str,
    checkpoint: str | Path | None,
    pooling: str,
    texts: list[str],
    cache_path: str | Path,
    device: str = "cuda",
    batch_size: int = 64,
    max_length: int = 256,
) -> torch.Tensor:
    """Load or create pooled hidden states with shape ``[L, N, D]``."""
    cache_path = Path(cache_path)
    probe_digest = _text_digest(texts)
    if cache_path.is_file():
        saved = torch.load(cache_path, map_location="cpu", weights_only=False)
        layers = saved["layers"] if isinstance(saved, dict) else saved
        if layers.shape[1] != len(texts):
            raise RuntimeError(
                f"Probe-size mismatch in {cache_path}: {layers.shape[1]} vs {len(texts)}"
            )
        saved_digest = saved.get("probe_sha256") if isinstance(saved, dict) else None
        if saved_digest != probe_digest:
            raise RuntimeError(
                f"Probe-content mismatch in {cache_path}; use a new RUN_NAME or "
                "remove this stale encoding cache."
            )
        return layers

    from transformers import AutoTokenizer
    from src import structural_audit as audit

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = audit.load_student(model_name, checkpoint, device=device)
    encoded = audit.encode_texts(
        model,
        tokenizer,
        texts,
        device=device,
        pooling=pooling,
        batch_size=batch_size,
        max_length=max_length,
        layers=True,
        progress=True,
    )
    layers = encoded["layers"]
    if layers is None:
        raise RuntimeError(f"{model_name} returned no hidden states")
    payload = {
        "layers": layers,
        "model_name": model_name,
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "pooling": pooling,
        "probe_size": len(texts),
        "probe_sha256": probe_digest,
    }
    torch.save(payload, cache_path)
    del model, tokenizer, encoded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return layers


def encode_pair_grid(
    *,
    pair_key: str,
    pair: dict,
    pair_run_root: str | Path,
    seeds: list[int],
    epochs: int,
    texts: list[str],
    encoding_root: str | Path,
    device: str,
    batch_size: int,
    max_length: int,
) -> dict[str, object]:
    """Encode teacher, student initialization, and every final checkpoint."""
    pair_run_root = Path(pair_run_root)
    encoding_root = Path(encoding_root) / pair_key
    paths: dict[str, object] = {}

    teacher_path = encoding_root / "teacher_layers.pt"
    encode_layers_cached(
        model_name=pair["teacher"], checkpoint=None,
        pooling=pair["teacher_pooling"], texts=texts,
        cache_path=teacher_path, device=device, batch_size=batch_size,
        max_length=max_length,
    )
    paths["teacher"] = teacher_path

    init_path = encoding_root / "student_init_layers.pt"
    encode_layers_cached(
        model_name=pair["student"], checkpoint=None,
        pooling=pair["student_pooling"], texts=texts,
        cache_path=init_path, device=device, batch_size=batch_size,
        max_length=max_length,
    )
    paths["init"] = init_path

    for arm in ("ours", "talas"):
        arm_paths: dict[int, Path] = {}
        for seed in seeds:
            checkpoint = pair_run_root / arm / f"seed_{seed}" / f"checkpoint_epoch_{epochs}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            path = encoding_root / f"{arm}_seed_{seed}_layers.pt"
            encode_layers_cached(
                model_name=pair["student"], checkpoint=checkpoint,
                pooling=pair["student_pooling"], texts=texts,
                cache_path=path, device=device, batch_size=batch_size,
                max_length=max_length,
            )
            arm_paths[seed] = path
        paths[arm] = arm_paths
    return paths


def _load_layers(path: str | Path) -> torch.Tensor:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    return saved["layers"] if isinstance(saved, dict) else saved


def _normalized_gram_vectors(layers: torch.Tensor, device: str) -> torch.Tensor:
    """Centered sample-space Gram matrices, flattened and Frobenius-normalized."""
    rows = []
    for layer in layers:
        matrix = layer.to(device=device, dtype=torch.float32)
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
        gram = matrix @ matrix.T
        rows.append((gram / gram.norm().clamp_min(1e-12)).reshape(-1))
    return torch.stack(rows)


def layerwise_linear_cka(
    teacher_layers: torch.Tensor,
    student_layers: torch.Tensor,
    *,
    device: str,
) -> np.ndarray:
    """Linear CKA for every teacher-layer/student-layer pair.

    The sample-space implementation is algebraically equal to feature-space
    linear CKA and is much faster when hidden widths exceed the probe size.
    """
    teacher_grams = _normalized_gram_vectors(teacher_layers, device)
    student_grams = _normalized_gram_vectors(student_layers, device)
    values = teacher_grams @ student_grams.T
    return values.clamp(0.0, 1.0).cpu().numpy()


def _cka_from_teacher_grams(
    teacher_grams: torch.Tensor,
    student_layers: torch.Tensor,
    *,
    device: str,
) -> np.ndarray:
    """Reuse teacher Gram matrices across checkpoints of the same pair."""
    student_grams = _normalized_gram_vectors(student_layers, device)
    values = teacher_grams @ student_grams.T
    return values.clamp(0.0, 1.0).cpu().numpy()


def _layerwise_h0_deaths(layers: torch.Tensor, cloud_size: int) -> torch.Tensor:
    from src.criterions.h0_topological_loss import h0_death_times

    return torch.stack([
        h0_death_times(
            layer.to(dtype=torch.float32), metric="chord", sort=True,
            chunk_size=cloud_size,
        ).cpu()
        for layer in layers
    ])


def layerwise_h0_residual(
    teacher_deaths: torch.Tensor,
    student_deaths: torch.Tensor,
) -> np.ndarray:
    if teacher_deaths.shape[1:] != student_deaths.shape[1:]:
        raise ValueError(
            f"H0 shape mismatch: {teacher_deaths.shape} vs {student_deaths.shape}"
        )
    residual = (
        teacher_deaths[:, None, ...] - student_deaths[None, :, ...]
    ).square().mean(dim=(-1, -2))
    return residual.numpy()


def _teacher_procrustes_views(
    teacher_layers: torch.Tensor,
    *,
    fit_rows: int,
    rank: int,
    student_width: int,
    device: str,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """PCA teacher coordinates for the fit and evaluation halves of the probe."""
    views = []
    for layer_index, layer in enumerate(teacher_layers):
        matrix = layer.to(device=device, dtype=torch.float32)
        fit, evaluate = matrix[:fit_rows], matrix[fit_rows:]
        centered = fit - fit.mean(dim=0, keepdim=True)
        q = min(rank, student_width, fit_rows - 1, centered.shape[1])
        if q < 1:
            raise ValueError(f"Invalid Procrustes rank {q}")
        # pca_lowrank is randomized, so isolate a deterministic seed per layer.
        parsed_device = torch.device(device)
        devices = []
        if parsed_device.type == "cuda":
            devices = [
                parsed_device.index
                if parsed_device.index is not None
                else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed + layer_index)
            _, _, basis = torch.pca_lowrank(centered, q=q, center=False, niter=4)
        # Match the paper recipe: center to estimate directions, but apply them to
        # the uncentered vectors before row normalization.
        views.append((
            F.normalize(fit @ basis, dim=-1),
            F.normalize(evaluate @ basis, dim=-1),
        ))
    return views


def cross_validated_procrustes_cosine(
    teacher_layers: torch.Tensor,
    student_layers: torch.Tensor,
    *,
    fit_rows: int,
    rank: int,
    device: str,
    seed: int,
) -> np.ndarray:
    """Held-out cosine after a semi-orthogonal PCA--Procrustes fit.

    Teacher layers are reduced to ``rank`` PCA coordinates.  A rectangular
    Procrustes map embeds those coordinates into the native student width and is
    fitted on the first probe split; cosine is measured on the disjoint second
    split.  Setting ``rank`` to the student width recovers the square gauge case.
    """
    student_width = int(student_layers.shape[-1])
    teacher_views = _teacher_procrustes_views(
        teacher_layers, fit_rows=fit_rows, rank=rank,
        student_width=student_width, device=device, seed=seed,
    )
    return _procrustes_from_teacher_views(
        teacher_views, student_layers, fit_rows=fit_rows, device=device,
    )


def _procrustes_from_teacher_views(
    teacher_views: list[tuple[torch.Tensor, torch.Tensor]],
    student_layers: torch.Tensor,
    *,
    fit_rows: int,
    device: str,
) -> np.ndarray:
    """Reuse teacher PCA views across checkpoints of the same pair."""
    student_views = []
    for layer in student_layers:
        matrix = layer.to(device=device, dtype=torch.float32)
        student_views.append((
            F.normalize(matrix[:fit_rows], dim=-1),
            F.normalize(matrix[fit_rows:], dim=-1),
        ))

    scores = torch.empty(
        len(teacher_views), len(student_views), dtype=torch.float32,
    )
    for teacher_index, (teacher_fit, teacher_eval) in enumerate(teacher_views):
        for student_index, (student_fit, student_eval) in enumerate(student_views):
            cross = teacher_fit.T @ student_fit
            left, _, right_t = torch.linalg.svd(cross, full_matrices=False)
            mapping = left @ right_t
            aligned = F.normalize(teacher_eval @ mapping, dim=-1)
            scores[teacher_index, student_index] = (
                aligned * student_eval
            ).sum(dim=-1).mean().cpu()
    return scores.numpy()


def _save_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        matrix,
        index=pd.Index(range(matrix.shape[0]), name="teacher_layer"),
        columns=[f"student_layer_{index}" for index in range(matrix.shape[1])],
    )
    frame.to_csv(path)


def _aggregate(per_seed: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stack = np.stack([per_seed[seed] for seed in sorted(per_seed)])
    mean = stack.mean(axis=0)
    std = stack.std(axis=0, ddof=1) if len(stack) > 1 else np.zeros_like(mean)
    return mean, std


def analyze_pair(
    *,
    pair_key: str,
    encoded_paths: dict[str, object],
    seeds: list[int],
    analysis_root: str | Path,
    h0_cloud_size: int,
    procrustes_fit_rows: int,
    procrustes_rank: int,
    device: str,
    seed: int = 0,
) -> dict[str, object]:
    """Compute and persist all matrices needed by the three figure options."""
    pair_root = Path(analysis_root) / pair_key
    matrix_root = pair_root / "matrices"
    matrix_root.mkdir(parents=True, exist_ok=True)

    teacher = _load_layers(encoded_paths["teacher"])
    initial = _load_layers(encoded_paths["init"])
    teacher_deaths = _layerwise_h0_deaths(teacher, h0_cloud_size)
    initial_deaths = _layerwise_h0_deaths(initial, h0_cloud_size)
    teacher_grams = _normalized_gram_vectors(teacher, device)
    initial_cka = _cka_from_teacher_grams(
        teacher_grams, initial, device=device,
    )
    initial_h0 = layerwise_h0_residual(teacher_deaths, initial_deaths)
    teacher_procrustes_views = _teacher_procrustes_views(
        teacher,
        fit_rows=procrustes_fit_rows,
        rank=procrustes_rank,
        student_width=int(initial.shape[-1]),
        device=device,
        seed=seed,
    )
    _save_matrix(matrix_root / "cka_init.csv", initial_cka)
    _save_matrix(matrix_root / "h0_residual_init.csv", initial_h0)

    per_arm: dict[str, dict[str, object]] = {}
    for arm in ("ours", "talas"):
        cka_by_seed: dict[int, np.ndarray] = {}
        h0_by_seed: dict[int, np.ndarray] = {}
        proc_by_seed: dict[int, np.ndarray] = {}
        for run_seed in seeds:
            student = _load_layers(encoded_paths[arm][run_seed])
            cka = _cka_from_teacher_grams(
                teacher_grams, student, device=device,
            )
            student_deaths = _layerwise_h0_deaths(student, h0_cloud_size)
            h0 = layerwise_h0_residual(teacher_deaths, student_deaths)
            proc = _procrustes_from_teacher_views(
                teacher_procrustes_views,
                student,
                fit_rows=procrustes_fit_rows,
                device=device,
            )
            cka_by_seed[run_seed] = cka
            h0_by_seed[run_seed] = h0
            proc_by_seed[run_seed] = proc
            _save_matrix(matrix_root / f"cka_{arm}_seed_{run_seed}.csv", cka)
            _save_matrix(matrix_root / f"h0_residual_{arm}_seed_{run_seed}.csv", h0)
            _save_matrix(matrix_root / f"procrustes_{arm}_seed_{run_seed}.csv", proc)
            del student, student_deaths
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cka_mean, cka_std = _aggregate(cka_by_seed)
        h0_mean, h0_std = _aggregate(h0_by_seed)
        proc_mean, proc_std = _aggregate(proc_by_seed)
        delta_cka = cka_mean - initial_cka
        h0_reduction = initial_h0 - h0_mean
        for name, matrix in (
            (f"cka_{arm}_mean", cka_mean),
            (f"cka_{arm}_std", cka_std),
            (f"delta_cka_{arm}_mean", delta_cka),
            (f"h0_residual_{arm}_mean", h0_mean),
            (f"h0_residual_{arm}_std", h0_std),
            (f"h0_reduction_{arm}_mean", h0_reduction),
            (f"procrustes_{arm}_mean", proc_mean),
            (f"procrustes_{arm}_std", proc_std),
        ):
            _save_matrix(matrix_root / f"{name}.csv", matrix)
        per_arm[arm] = {
            "cka_mean": cka_mean,
            "cka_std": cka_std,
            "delta_cka": delta_cka,
            "h0_mean": h0_mean,
            "h0_std": h0_std,
            "h0_reduction": h0_reduction,
            "procrustes_mean": proc_mean,
            "procrustes_std": proc_std,
        }

    metadata = {
        "pair": pair_key,
        "seeds": seeds,
        "teacher_layers": int(teacher.shape[0]),
        "student_layers": int(initial.shape[0]),
        "probe_size": int(teacher.shape[1]),
        "h0_cloud_size": h0_cloud_size,
        "procrustes_fit_rows": procrustes_fit_rows,
        "procrustes_eval_rows": int(teacher.shape[1] - procrustes_fit_rows),
        "procrustes_rank": min(
            procrustes_rank, int(initial.shape[-1]), procrustes_fit_rows - 1,
        ),
    }
    (pair_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )
    return {
        "init_cka": initial_cka,
        "init_h0": initial_h0,
        "ours": per_arm["ours"],
        "talas": per_arm["talas"],
        "metadata": metadata,
    }


def _paper_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.titlesize": 7.8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "axes.linewidth": 0.65,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _decorate_heatmap(
    axis,
    matrix: np.ndarray,
    *,
    show_y_ticks: bool,
    row_label: str | None,
) -> None:
    import matplotlib.patheffects as path_effects
    from matplotlib.patches import Rectangle

    teacher_layers, student_layers = matrix.shape
    axis.set_xticks(sorted(set([0, student_layers // 2, student_layers - 1])))
    axis.set_yticks(sorted(set([0, teacher_layers // 2, teacher_layers - 1])))
    axis.tick_params(axis="both", which="major", length=2.5, width=0.65, pad=1.5)
    axis.tick_params(axis="y", left=show_y_ticks, labelleft=show_y_ticks)
    if row_label is not None:
        axis.set_ylabel(row_label, fontsize=7.1, labelpad=6.0)

    # Inset the endpoint marker so the top and right edges are not clipped by the
    # axes boundary.  A black line with a white halo stays visible on every map;
    # a dashed line is illegible once a single cell is printed at ICLR width.
    marker = Rectangle(
        (student_layers - 1.42, teacher_layers - 1.42), 0.84, 0.84,
        fill=False, edgecolor="black", linewidth=0.8,
        linestyle="-", clip_on=True,
    )
    marker.set_path_effects([
        path_effects.Stroke(linewidth=2.0, foreground="white"),
        path_effects.Normal(),
    ])
    axis.add_patch(marker)


def _save_figure(figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"]
    # The ICLR style uses a 5.5-inch text block.  Every canvas below is already
    # exactly that width, so avoid bbox_inches="tight", which changes the final
    # physical size and silently shrinks type when LaTeX includes the figure.
    figure.savefig(paths[0], facecolor="white")
    figure.savefig(paths[1], facecolor="white")
    return paths


def _nondegenerate_limits(low: float, high: float) -> tuple[float, float]:
    """Keep color normalization valid for constant-valued smoke-test matrices."""
    if high > low:
        return low, high
    padding = max(abs(low) * 1e-6, 1e-8)
    return low - padding, high + padding


def render_all_options(
    results: dict[str, dict[str, object]],
    *,
    pair_order: list[str],
    pair_labels: dict[str, str],
    output_dir: str | Path,
    figure_note: str | None = None,
) -> list[Path]:
    """Render three all-pairs appendix candidates from real analysis matrices."""
    import matplotlib.pyplot as plt

    _paper_style()
    output_dir = Path(output_dir)
    written: list[Path] = []

    def add_optional_note(figure) -> None:
        if figure_note:
            figure.text(
                0.51, 0.50, figure_note, ha="center", va="center",
                rotation=24, fontsize=21, fontweight="bold",
                color="#444444", alpha=0.10, zorder=20,
            )

    # Option 1: rows are model pairs; columns are initialization, GATE-KD, TALAS.
    figure, axes = plt.subplots(len(pair_order), 3, figsize=(5.5, 5.25), squeeze=False)
    column_titles = ("Pretrained student", "GATE-KD", "TALAS")
    image = None
    for row, pair_key in enumerate(pair_order):
        result = results[pair_key]
        matrices = (
            result["init_cka"], result["ours"]["cka_mean"],
            result["talas"]["cka_mean"],
        )
        for column, matrix in enumerate(matrices):
            axis = axes[row, column]
            image = axis.imshow(
                matrix, origin="lower", aspect="auto", cmap="viridis",
                vmin=0.0, vmax=1.0, interpolation="nearest", rasterized=True,
            )
            if row == 0:
                axis.set_title(column_titles[column])
            _decorate_heatmap(
                axis, matrix, show_y_ticks=column == 0,
                row_label=pair_labels[pair_key] if column == 0 else None,
            )
    figure.subplots_adjust(left=0.20, right=0.86, bottom=0.08, top=0.94,
                           hspace=0.27, wspace=0.12)
    color_axis = figure.add_axes([0.89, 0.20, 0.016, 0.62])
    figure.colorbar(image, cax=color_axis, label=r"Linear CKA $\uparrow$")
    figure.supxlabel("Student layer", x=0.53, y=0.012, fontsize=7.5)
    figure.supylabel("Teacher layer", x=0.018, fontsize=7.5)
    add_optional_note(figure)
    written.extend(_save_figure(figure, output_dir, "option1_all_pairs_raw_cka"))
    plt.close(figure)

    cka_limit = max(1e-8, max(
        float(np.abs(results[key][arm]["delta_cka"]).max())
        for key in pair_order for arm in ("ours", "talas")
    ))
    h0_gain_limit = max(1e-8, max(
        float(np.abs(results[key][arm]["h0_reduction"]).max())
        for key in pair_order for arm in ("ours", "talas")
    ))
    # Option 2: four columns keep all three pairs readable on one appendix page.
    figure, axes = plt.subplots(len(pair_order), 4, figsize=(5.5, 5.05), squeeze=False)
    titles = (
        "GATE-KD\n" + r"$\Delta$CKA", "TALAS\n" + r"$\Delta$CKA",
        "GATE-KD\n" + r"$H_0$ reduction", "TALAS\n" + r"$H_0$ reduction",
    )
    cka_image = h0_image = None
    for row, pair_key in enumerate(pair_order):
        result = results[pair_key]
        matrices = (
            result["ours"]["delta_cka"], result["talas"]["delta_cka"],
            result["ours"]["h0_reduction"], result["talas"]["h0_reduction"],
        )
        for column, matrix in enumerate(matrices):
            limit = cka_limit if column < 2 else h0_gain_limit
            axis = axes[row, column]
            current = axis.imshow(
                matrix, origin="lower", aspect="auto", cmap="RdBu_r",
                vmin=-limit, vmax=limit, interpolation="nearest", rasterized=True,
            )
            if column < 2:
                cka_image = current
            else:
                h0_image = current
            if row == 0:
                axis.set_title(titles[column])
            _decorate_heatmap(
                axis, matrix, show_y_ticks=column == 0,
                row_label=pair_labels[pair_key] if column == 0 else None,
            )
    figure.subplots_adjust(left=0.20, right=0.84, bottom=0.08, top=0.91,
                           hspace=0.27, wspace=0.13)
    cka_axis = figure.add_axes([0.865, 0.58, 0.015, 0.25])
    h0_axis = figure.add_axes([0.865, 0.19, 0.015, 0.25])
    figure.colorbar(cka_image, cax=cka_axis, label=r"$\Delta$CKA $\uparrow$")
    figure.colorbar(
        h0_image, cax=h0_axis, label=r"$H_0$ residual reduction $\uparrow$",
    )
    figure.supxlabel("Student layer", x=0.51, y=0.012, fontsize=7.5)
    figure.supylabel("Teacher layer", x=0.018, fontsize=7.5)
    add_optional_note(figure)
    written.extend(_save_figure(figure, output_dir, "option2_all_pairs_delta_cka_h0"))
    plt.close(figure)

    proc_min = min(
        float(results[key][arm]["procrustes_mean"].min())
        for key in pair_order for arm in ("ours", "talas")
    )
    proc_max = max(
        float(results[key][arm]["procrustes_mean"].max())
        for key in pair_order for arm in ("ours", "talas")
    )
    h0_min = min(
        float(results[key][arm]["h0_mean"].min())
        for key in pair_order for arm in ("ours", "talas")
    )
    h0_max = max(
        float(results[key][arm]["h0_mean"].max())
        for key in pair_order for arm in ("ours", "talas")
    )
    proc_min, proc_max = _nondegenerate_limits(proc_min, proc_max)
    h0_min, h0_max = _nondegenerate_limits(h0_min, h0_max)
    figure, axes = plt.subplots(len(pair_order), 4, figsize=(5.5, 5.05), squeeze=False)
    titles = ("GATE-KD\nExtrinsic", "TALAS\nExtrinsic",
              "GATE-KD\nIntrinsic", "TALAS\nIntrinsic")
    proc_image = h0_image = None
    for row, pair_key in enumerate(pair_order):
        result = results[pair_key]
        matrices = (
            result["ours"]["procrustes_mean"],
            result["talas"]["procrustes_mean"],
            result["ours"]["h0_mean"],
            result["talas"]["h0_mean"],
        )
        for column, matrix in enumerate(matrices):
            axis = axes[row, column]
            if column < 2:
                current = axis.imshow(
                    matrix, origin="lower", aspect="auto", cmap="viridis",
                    vmin=proc_min, vmax=proc_max,
                    interpolation="nearest", rasterized=True,
                )
                proc_image = current
            else:
                current = axis.imshow(
                    matrix, origin="lower", aspect="auto", cmap="magma_r",
                    vmin=h0_min, vmax=h0_max,
                    interpolation="nearest", rasterized=True,
                )
                h0_image = current
            if row == 0:
                axis.set_title(titles[column])
            _decorate_heatmap(
                axis, matrix, show_y_ticks=column == 0,
                row_label=pair_labels[pair_key] if column == 0 else None,
            )
    figure.subplots_adjust(left=0.20, right=0.84, bottom=0.08, top=0.91,
                           hspace=0.27, wspace=0.13)
    proc_axis = figure.add_axes([0.865, 0.58, 0.015, 0.25])
    h0_axis = figure.add_axes([0.865, 0.19, 0.015, 0.25])
    figure.colorbar(
        proc_image, cax=proc_axis,
        label=r"PCA--Procrustes cosine $\uparrow$",
    )
    figure.colorbar(
        h0_image, cax=h0_axis,
        label=r"Native-space $H_0$ residual $\downarrow$",
    )
    figure.supxlabel("Student layer", x=0.51, y=0.012, fontsize=7.5)
    figure.supylabel("Teacher layer", x=0.018, fontsize=7.5)
    add_optional_note(figure)
    written.extend(_save_figure(figure, output_dir, "option3_all_pairs_procrustes_h0"))
    plt.close(figure)
    return written
