"""Shared computations for the all-pairs layerwise-analysis notebook.

This module keeps the notebook readable while making the layerwise CKA
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
    # Keep the RNG order. Sorting the sampled indices would cluster the probe by
    # source file and make the representation comparison source-order dependent.
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
        norm = gram.norm()
        if norm <= 1e-12:
            # Centered CKA is undefined for a constant representation. This is
            # common at pooled embedding-output layer 0 because every example
            # reads the same special token. Preserve the undefined value rather
            # than silently displaying it as zero.
            rows.append(torch.full_like(gram.reshape(-1), torch.nan))
        else:
            rows.append((gram / norm).reshape(-1))
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
    device: str,
) -> dict[str, object]:
    """Compute and persist raw and initialization-relative CKA matrices."""
    pair_root = Path(analysis_root) / pair_key
    matrix_root = pair_root / "matrices"
    matrix_root.mkdir(parents=True, exist_ok=True)

    teacher = _load_layers(encoded_paths["teacher"])
    initial = _load_layers(encoded_paths["init"])
    teacher_grams = _normalized_gram_vectors(teacher, device)
    initial_cka = _cka_from_teacher_grams(
        teacher_grams, initial, device=device,
    )
    _save_matrix(matrix_root / "cka_init.csv", initial_cka)

    per_arm: dict[str, dict[str, object]] = {}
    for arm in ("ours", "talas"):
        cka_by_seed: dict[int, np.ndarray] = {}
        for run_seed in seeds:
            student = _load_layers(encoded_paths[arm][run_seed])
            cka = _cka_from_teacher_grams(
                teacher_grams, student, device=device,
            )
            cka_by_seed[run_seed] = cka
            _save_matrix(matrix_root / f"cka_{arm}_seed_{run_seed}.csv", cka)
            del student
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cka_mean, cka_std = _aggregate(cka_by_seed)
        delta_cka = cka_mean - initial_cka
        for name, matrix in (
            (f"cka_{arm}_mean", cka_mean),
            (f"cka_{arm}_std", cka_std),
            (f"delta_cka_{arm}_mean", delta_cka),
        ):
            _save_matrix(matrix_root / f"{name}.csv", matrix)
        per_arm[arm] = {
            "cka_mean": cka_mean,
            "cka_std": cka_std,
            "delta_cka": delta_cka,
        }

    metadata = {
        "pair": pair_key,
        "seeds": seeds,
        "teacher_layers": int(teacher.shape[0]),
        "student_layers": int(initial.shape[0]),
        "probe_size": int(teacher.shape[1]),
        "displayed_layer_offset": 1,
    }
    (pair_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )
    return {
        "init_cka": initial_cka,
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
    layer_offset: int = 1,
) -> None:
    teacher_layers, student_layers = matrix.shape
    x_ticks = sorted(set([0, student_layers // 2, student_layers - 1]))
    y_ticks = sorted(set([0, teacher_layers // 2, teacher_layers - 1]))
    axis.set_xticks(x_ticks, [index + layer_offset for index in x_ticks])
    axis.set_yticks(y_ticks, [index + layer_offset for index in y_ticks])
    axis.tick_params(axis="both", which="major", length=2.5, width=0.65, pad=1.5)
    axis.tick_params(axis="y", left=show_y_ticks, labelleft=show_y_ticks)
    if row_label is not None:
        axis.set_ylabel(row_label, fontsize=7.1, labelpad=6.0)


def _save_figure(figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"]
    # The ICLR style uses a 5.5-inch text block.  Every canvas below is already
    # exactly that width, so avoid bbox_inches="tight", which changes the final
    # physical size and silently shrinks type when LaTeX includes the figure.
    figure.savefig(paths[0], facecolor="white")
    figure.savefig(paths[1], facecolor="white")
    return paths


def render_cka_figures(
    results: dict[str, dict[str, object]],
    *,
    pair_order: list[str],
    pair_labels: dict[str, str],
    output_dir: str | Path,
    figure_note: str | None = None,
) -> list[Path]:
    """Render the two CKA figures that directly test the layerwise claim."""
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

    def displayed(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or min(matrix.shape) < 2:
            raise ValueError(f"Expected a layer matrix including layer 0, got {matrix.shape}")
        # Pooled embedding-output layer 0 is constant for the special-token
        # pooling used here, so centered CKA is undefined. Only Transformer
        # layers, numbered from 1 in the model, belong in the visualization.
        return matrix[1:, 1:]

    # Raw similarity: initialization controls for inherited correspondence.
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
            matrix = displayed(matrix)
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
    written.extend(_save_figure(figure, output_dir, "layerwise_raw_cka_all_pairs"))
    plt.close(figure)

    cka_limit = max(1e-8, max(
        float(np.nanmax(np.abs(displayed(results[key][arm]["delta_cka"]))))
        for key in pair_order for arm in ("ours", "talas")
    ))
    # Initialization-relative similarity isolates changes induced by training.
    figure, axes = plt.subplots(len(pair_order), 2, figsize=(5.5, 5.25), squeeze=False)
    titles = ("GATE-KD", "TALAS")
    image = None
    for row, pair_key in enumerate(pair_order):
        result = results[pair_key]
        matrices = (
            result["ours"]["delta_cka"], result["talas"]["delta_cka"],
        )
        for column, matrix in enumerate(matrices):
            matrix = displayed(matrix)
            axis = axes[row, column]
            image = axis.imshow(
                matrix, origin="lower", aspect="auto", cmap="RdBu_r",
                vmin=-cka_limit, vmax=cka_limit,
                interpolation="nearest", rasterized=True,
            )
            if row == 0:
                axis.set_title(titles[column])
            _decorate_heatmap(
                axis, matrix, show_y_ticks=column == 0,
                row_label=pair_labels[pair_key] if column == 0 else None,
            )
    figure.subplots_adjust(left=0.20, right=0.86, bottom=0.08, top=0.94,
                           hspace=0.27, wspace=0.14)
    color_axis = figure.add_axes([0.89, 0.20, 0.016, 0.62])
    figure.colorbar(image, cax=color_axis, label=r"$\Delta$ linear CKA $\uparrow$")
    figure.supxlabel("Student layer", x=0.53, y=0.012, fontsize=7.5)
    figure.supylabel("Teacher layer", x=0.018, fontsize=7.5)
    add_optional_note(figure)
    written.extend(_save_figure(figure, output_dir, "layerwise_delta_cka_all_pairs"))
    plt.close(figure)
    return written
