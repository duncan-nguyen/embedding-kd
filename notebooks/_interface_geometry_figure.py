"""Publication renderer for notebook 01's teacher-interface schematic.

The paper figure is intentionally a schematic, not a low-dimensional claim about
one evaluation split. Numerical annotations still come from the real fit corpus.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch


CLASS_PALETTE = ("#2F6FA3", "#DD6B3B", "#2A9D78", "#D8A126")
INK = "#202327"
MUTED = "#737984"
GHOST = "#ADB3BC"
STUDENT = "#27344A"
REDUCE = "#16847E"
SELECTED = "#7048B5"

FIGURE_WIDTH_IN = 5.5
FIGURE_HEIGHT_IN = 1.82


def _rotation(angle: float) -> np.ndarray:
    return np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )


def _elliptical_noise(
    rng: np.random.Generator,
    count: int,
    scale: tuple[float, float],
    angle: float,
) -> np.ndarray:
    noise = rng.normal(size=(count, 2)) * np.asarray(scale)
    return noise @ _rotation(angle)


def _schematic_geometry(seed: int = 19) -> tuple[dict[str, object], np.ndarray, list[str]]:
    """Create a restrained, deterministic visual metaphor for the method."""

    rng = np.random.default_rng(seed)
    count = 52
    names = ["group 1", "group 2", "group 3", "group 4"]
    labels = np.repeat(names, count)

    reduced_centres = np.array(
        [
            [-1.30, 0.70],
            [1.22, 0.78],
            [-0.95, -0.88],
            [1.02, -0.72],
        ]
    )
    scales = ((0.25, 0.17), (0.27, 0.18), (0.24, 0.19), (0.28, 0.16))
    angles = (0.35, -0.45, -0.20, 0.42)
    reduced = np.vstack(
        [
            centre + _elliptical_noise(rng, count, scale, angle)
            for centre, scale, angle in zip(
                reduced_centres, scales, angles, strict=True
            )
        ]
    )

    teacher_centres = np.array(
        [
            [-1.35, 0.72, 0.42],
            [1.25, 0.88, 0.72],
            [-0.78, -0.92, -0.58],
            [1.05, -0.68, -0.28],
        ]
    )
    teacher_parts = []
    for index, centre in enumerate(teacher_centres):
        noise_2d = _elliptical_noise(
            rng, count, (0.27, 0.18), angles[index]
        )
        depth = rng.normal(0, 0.15, size=(count, 1))
        teacher_parts.append(centre + np.column_stack((noise_2d, depth)))
    teacher = np.vstack(teacher_parts)

    selected = reduced @ _rotation(0.58)
    student_parts = []
    for index in range(len(names)):
        section = slice(index * count, (index + 1) * count)
        offset = np.array((0.035, -0.025)) * (index - 1.5)
        student_parts.append(
            selected[section]
            + offset
            + rng.normal(0, (0.075, 0.065), size=(count, 2))
        )
    student = np.vstack(student_parts)
    ghosts = [reduced @ _rotation(angle) for angle in (-0.82, -0.30, 1.08)]

    return (
        {
            "T": teacher,
            "Y_orbit": reduced,
            "YR": selected,
            "Z": student,
            "ghosts": ghosts,
        },
        labels,
        names,
    )


def _writer(ax):
    return getattr(ax, "text2D", ax.text)


def _heading(ax, title: str, meta: str, accent: str) -> None:
    ax.set_title(
        title,
        loc="left",
        fontsize=8.0,
        fontweight="normal",
        color=INK,
        pad=4,
    )
    _writer(ax)(
        0.99,
        1.025,
        meta,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.7,
        fontweight="bold",
        color=accent,
    )


def _centroids(
    points: np.ndarray, labels: np.ndarray, label_order: Sequence[str]
) -> np.ndarray:
    return np.stack([points[labels == str(label)].mean(axis=0) for label in label_order])


def _square_window(ax, clouds: Sequence[np.ndarray], pad: float = 0.12) -> None:
    points = np.vstack(clouds)
    low, high = np.quantile(points, (0.005, 0.995), axis=0)
    centre = (low + high) / 2
    half = max(float((high - low).max()) * (1 + pad) / 2, 1e-6)
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def _flow_arrow(figure, left_ax, right_ax, color: str) -> None:
    left = left_ax.get_position()
    right = right_ax.get_position()
    y = left.y0 + 0.49 * left.height
    figure.add_artist(
        FancyArrowPatch(
            (left.x1 + 0.006, y),
            (right.x0 - 0.006, y),
            transform=figure.transFigure,
            arrowstyle="-|>",
            mutation_scale=6,
            linewidth=0.80,
            color=color,
            clip_on=False,
            zorder=30,
        )
    )


def _resolve_view(
    view: Mapping[str, object] | None,
    plot_labels: Sequence[str] | None,
    label_order: Sequence[str] | None,
    *,
    schematic: bool,
) -> tuple[Mapping[str, object], np.ndarray, Sequence[str]]:
    if schematic:
        return _schematic_geometry()
    if view is None or plot_labels is None or label_order is None:
        raise ValueError("Data mode requires view, plot_labels, and label_order.")
    if "Y_orbit" not in view:
        if "Y" not in view:
            raise KeyError("VIEW thiếu 'Y_orbit'. Hãy chạy lại cell 6 trước cell 7.")
        warnings.warn(
            "Đang dùng VIEW cũ qua key 'Y'. Hãy chạy lại cell 6 để dùng data view mới.",
            RuntimeWarning,
            stacklevel=3,
        )
    return view, np.asarray(plot_labels, dtype=str), label_order


def render_interface_geometry(
    *,
    view: Mapping[str, object] | None = None,
    plot_labels: Sequence[str] | None = None,
    label_order: Sequence[str] | None = None,
    teacher_dim: int,
    student_dim: int,
    explained_energy: float,
    output_dir: Path,
    stem: str,
    schematic: bool = True,
    show: bool = True,
):
    """Render the one-row Figure 1 and export paper-safe formats."""

    view, labels, label_order = _resolve_view(
        view, plot_labels, label_order, schematic=schematic
    )
    colour_of = {
        str(label): CLASS_PALETTE[index % len(CLASS_PALETTE)]
        for index, label in enumerate(label_order)
    }
    point_colours = np.asarray([colour_of[label] for label in labels])
    centroid_colours = [colour_of[str(label)] for label in label_order]

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Tinos", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.0,
            "figure.dpi": 170,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        figure = plt.figure(
            figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), facecolor="white"
        )
        grid = figure.add_gridspec(
            1,
            3,
            left=0.018,
            right=0.992,
            top=0.84,
            bottom=0.025,
            wspace=0.27,
        )
        ax_teacher = figure.add_subplot(grid[0, 0], projection="3d")
        ax_reduced = figure.add_subplot(grid[0, 1])
        ax_selected = figure.add_subplot(grid[0, 2])

        teacher = np.asarray(view["T"])
        reduced = np.asarray(
            view["Y_orbit"] if "Y_orbit" in view else view["Y"]
        )
        selected = np.asarray(view["YR"])
        student = np.asarray(view["Z"])
        ghosts = [np.asarray(cloud) for cloud in view["ghosts"]]

        ax_teacher.scatter(
            *teacher.T,
            s=4.6,
            c=point_colours,
            alpha=0.66,
            linewidths=0,
            depthshade=False,
            rasterized=True,
        )
        teacher_centres = _centroids(teacher, labels, label_order)
        ax_teacher.scatter(
            *teacher_centres.T,
            s=24,
            c=centroid_colours,
            edgecolors="white",
            linewidths=0.65,
            depthshade=False,
            zorder=8,
        )
        low, high = teacher.min(axis=0), teacher.max(axis=0)
        span = max(float((high - low).max()), 1e-6)
        base = low - 0.02 * span
        for axis_index in range(3):
            direction = np.zeros(3)
            direction[axis_index] = 0.18 * span
            ax_teacher.quiver(
                *base,
                *direction,
                color=MUTED,
                linewidth=0.55,
                arrow_length_ratio=0.15,
            )
        ax_teacher.set_proj_type("ortho")
        ax_teacher.view_init(elev=18, azim=-58)
        ax_teacher.set_box_aspect((1, 1, 0.86), zoom=1.27)
        ax_teacher.set_axis_off()
        _heading(ax_teacher, "Teacher space", f"{teacher_dim}-D", REDUCE)

        for ghost in ghosts:
            ax_reduced.scatter(
                *ghost.T,
                s=3.0,
                c=GHOST,
                alpha=0.095,
                linewidths=0,
                rasterized=True,
                zorder=1,
            )
        ax_reduced.scatter(
            *reduced.T,
            s=4.8,
            c=point_colours,
            alpha=0.74,
            linewidths=0,
            rasterized=True,
            zorder=4,
        )
        reduced_centres = _centroids(reduced, labels, label_order)
        ax_reduced.scatter(
            *reduced_centres.T,
            s=23,
            c=centroid_colours,
            edgecolors="white",
            linewidths=0.65,
            zorder=7,
        )
        _square_window(ax_reduced, [reduced, *ghosts])
        _heading(
            ax_reduced,
            "Reduced geometry",
            f"{student_dim}-D  ·  {explained_energy:.1%}",
            REDUCE,
        )

        thread_indices = np.arange(0, len(selected), max(1, len(selected) // 24))
        ax_selected.add_collection(
            LineCollection(
                np.stack(
                    (selected[thread_indices], student[thread_indices]), axis=1
                ),
                colors=GHOST,
                linewidths=0.38,
                alpha=0.38,
                zorder=1,
            )
        )
        ax_selected.scatter(
            *student.T,
            s=5.2,
            marker="x",
            c=STUDENT,
            alpha=0.30,
            linewidths=0.38,
            rasterized=True,
            zorder=2,
        )
        ax_selected.scatter(
            *selected.T,
            s=4.8,
            c=point_colours,
            alpha=0.76,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
        selected_centres = _centroids(selected, labels, label_order)
        student_centres = _centroids(student, labels, label_order)
        ax_selected.add_collection(
            LineCollection(
                np.stack((selected_centres, student_centres), axis=1),
                colors=GHOST,
                linewidths=0.60,
                alpha=0.70,
                zorder=5,
            )
        )
        ax_selected.scatter(
            *selected_centres.T,
            s=24,
            c=centroid_colours,
            edgecolors="white",
            linewidths=0.65,
            zorder=7,
        )
        ax_selected.scatter(
            *student_centres.T,
            s=19,
            marker="x",
            c=STUDENT,
            linewidths=0.78,
            zorder=8,
        )
        _square_window(ax_selected, [selected, student])
        _heading(
            ax_selected,
            "Student-selected target",
            f"{student_dim}-D",
            SELECTED,
        )

        _flow_arrow(figure, ax_teacher, ax_reduced, REDUCE)
        _flow_arrow(figure, ax_reduced, ax_selected, SELECTED)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "pdf": output_dir / f"{stem}.pdf",
            "svg": output_dir / f"{stem}.svg",
            "png": output_dir / f"{stem}.png",
        }
        metadata = {
            "Title": "Teacher interface: dimension reduction and coordinate selection",
            "Creator": "notebooks/01_interface_geometry_figure.ipynb",
            "Subject": "Schematic; numerical annotations are measured on the fit corpus",
        }
        figure.savefig(paths["pdf"], metadata=metadata)
        figure.savefig(paths["svg"])
        figure.savefig(paths["png"])
        if show:
            plt.show()
        return figure, paths
