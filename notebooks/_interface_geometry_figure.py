"""Publication renderer for notebook 01's interface-geometry figure.

The renderer deliberately receives already-computed arrays.  All scientific
choices (fit set, PCA, Procrustes, and viewing frames) stay in the notebook;
this module owns only layout, visual semantics, and paper-safe export.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Okabe--Ito with a darkened yellow.  Class identity is carried by colour,
# representation identity by marker shape, so the comparison still works in
# grayscale and for the common forms of colour-vision deficiency.
CLASS_PALETTE = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#7A6F59",
    "#6B7280",
)

INK = "#202124"
MUTED = "#626975"
HAIR = "#D9DCE1"
GRID = "#E7E9ED"
PANEL = "#FAFAF8"
GHOST = "#A8ADB7"
STUDENT = "#25324A"
SELECTED = "#6F4AA8"
REDUCE = "#137C78"

FIGURE_WIDTH_IN = 5.5
FIGURE_HEIGHT_IN = 2.05


def _writer(ax):
    """Return an axes-coordinate text method for 2-D and 3-D axes."""

    return getattr(ax, "text2D", ax.text)


def _panel_title(ax, title: str, kicker: str, accent: str) -> None:
    ax.set_title(
        title,
        loc="left",
        color=INK,
        fontsize=8.3,
        fontweight="semibold",
        pad=7,
    )
    _writer(ax)(
        0.025,
        0.965,
        kicker,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.4,
        fontweight="semibold",
        color=accent,
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.90,
        },
        zorder=20,
    )


def _panel_card(figure, ax) -> None:
    """Put a restrained card behind a panel without changing its data limits."""

    box = ax.get_position()
    card = FancyBboxPatch(
        (box.x0, box.y0),
        box.width,
        box.height,
        boxstyle="round,pad=0.004,rounding_size=0.008",
        transform=figure.transFigure,
        facecolor=PANEL,
        edgecolor=HAIR,
        linewidth=0.55,
        zorder=-10,
        clip_on=False,
    )
    figure.add_artist(card)
    ax.set_facecolor("none")


def _flat_panel(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axhline(0, color=GRID, linewidth=0.40, zorder=-1)
    ax.axvline(0, color=GRID, linewidth=0.40, zorder=-1)
    ax.set_box_aspect(1)


def _shared_square_window(
    axes, clouds: Sequence[np.ndarray], pad: float = 0.08
) -> float:
    """Apply one robust coordinate window within a single visual comparison."""

    points = np.vstack(clouds)
    low, high = np.quantile(points, (0.005, 0.995), axis=0)
    center = (low + high) / 2
    half = max(float((high - low).max()) * (1 + pad) / 2, 1e-6)
    for ax in axes:
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
    return half


def _flow_arrow(figure, left_ax, right_ax, color: str) -> None:
    """Connect the three conceptual stages in figure coordinates."""

    left = left_ax.get_position()
    right = right_ax.get_position()
    y = left.y0 + 0.49 * left.height
    x0, x1 = left.x1 + 0.004, right.x0 - 0.004
    if x1 <= x0:
        return
    figure.add_artist(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            transform=figure.transFigure,
            arrowstyle="-|>",
            mutation_scale=6,
            linewidth=0.75,
            color=color,
            clip_on=False,
            zorder=30,
        )
    )


def render_interface_geometry(
    *,
    view: Mapping[str, object],
    plot_labels: Sequence[str],
    label_order: Sequence[str],
    teacher_dim: int,
    student_dim: int,
    explained_energy: float,
    output_dir: Path,
    stem: str,
    show: bool = True,
):
    """Render Figure 1 at its final ICLR width and export PDF, SVG, and PNG."""

    colour_of = {
        label: CLASS_PALETTE[index % len(CLASS_PALETTE)]
        for index, label in enumerate(label_order)
    }
    point_colours = np.asarray([colour_of[str(label)] for label in plot_labels])

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.2,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 5.8,
            "figure.dpi": 170,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": True,
        }
    ):
        figure = plt.figure(
            figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), facecolor="white"
        )
        grid = figure.add_gridspec(
            1,
            3,
            left=0.035,
            right=0.985,
            top=0.87,
            bottom=0.055,
            wspace=0.22,
        )
        ax_a = figure.add_subplot(grid[0, 0], projection="3d")
        ax_b = figure.add_subplot(grid[0, 1])
        ax_c = figure.add_subplot(grid[0, 2])

        for ax in (ax_a, ax_b, ax_c):
            _panel_card(figure, ax)

        # (a) The 3-D view is only a view of T; no quantity is measured here.
        teacher = np.asarray(view["T"])
        ax_a.scatter(
            *teacher.T,
            s=4.4,
            c=point_colours,
            alpha=0.78,
            linewidths=0,
            depthshade=False,
            rasterized=True,
        )
        ax_a.set_proj_type("ortho")
        ax_a.view_init(elev=17, azim=-57)
        ax_a.set_box_aspect((1, 1, 0.86), zoom=1.18)
        ax_a.set_axis_off()

        low, high = teacher.min(axis=0), teacher.max(axis=0)
        span = max(float((high - low).max()), 1e-6)
        base = low - 0.025 * span
        length = 0.22 * span
        for axis_index in range(3):
            direction = np.zeros(3)
            direction[axis_index] = length
            ax_a.quiver(
                *base,
                *direction,
                color=MUTED,
                linewidth=0.58,
                arrow_length_ratio=0.14,
            )
        _panel_title(
            ax_a,
            "Teacher",
            f"{teacher_dim}-D",
            REDUCE,
        )

        # (b) A single coloured realization sits above monochrome orbit samples.
        ghosts = [np.asarray(cloud) for cloud in view["ghosts"]]
        legacy_view = "Y_orbit" not in view or "Y_before" not in view
        if legacy_view:
            if "Y" not in view:
                raise KeyError(
                    "VIEW thiếu 'Y_orbit'/'Y_before'. Hãy chạy lại cell 6 trước cell 7."
                )
            warnings.warn(
                "Đang dùng VIEW cũ qua key 'Y'. Cell 7 vẫn render được, nhưng hãy "
                "chạy lại cell 6 để dùng framing mới tối ưu cho paper.",
                RuntimeWarning,
                stacklevel=2,
            )
        reduced = np.asarray(view["Y_orbit"] if "Y_orbit" in view else view["Y"])
        for ghost in ghosts:
            ax_b.scatter(
                *ghost.T,
                s=2.3,
                c=GHOST,
                alpha=0.105,
                linewidths=0,
                rasterized=True,
                zorder=1,
            )
        ax_b.scatter(
            *reduced.T,
            s=4.4,
            c=point_colours,
            alpha=0.86,
            linewidths=0,
            rasterized=True,
            zorder=4,
        )
        _flat_panel(ax_b)
        _panel_title(
            ax_b,
            "Reduced teacher",
            f"{student_dim}-D  •  {explained_energy:.1%} retained",
            REDUCE,
        )
        start_b = reduced.mean(axis=0)
        end_b = ghosts[0].mean(axis=0)
        ax_b.add_patch(
            FancyArrowPatch(
                start_b,
                end_b,
                connectionstyle="arc3,rad=-0.28",
                arrowstyle="-|>",
                mutation_scale=6,
                linewidth=0.72,
                color=REDUCE,
                zorder=8,
            )
        )

        # (c) Marker shape separates student x from target dots even in grayscale.
        student = np.asarray(view["Z"])
        selected = np.asarray(view["YR"])
        before = np.asarray(view["Y_before"] if "Y_before" in view else view["Y"])
        n_links = min(18, len(student))
        link_indices = np.linspace(0, len(student) - 1, n_links, dtype=int)
        segments = np.stack((selected[link_indices], student[link_indices]), axis=1)
        ax_c.add_collection(
            LineCollection(
                segments,
                colors=STUDENT,
                linewidths=0.25,
                alpha=0.12,
                zorder=2,
                rasterized=True,
            )
        )
        ax_c.scatter(
            *before.T,
            s=2.2,
            c=GHOST,
            alpha=0.16,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        ax_c.scatter(
            *selected.T,
            s=4.5,
            c=point_colours,
            alpha=0.88,
            linewidths=0,
            rasterized=True,
            zorder=4,
        )
        ax_c.scatter(
            *student.T,
            s=6.0,
            marker="x",
            c=STUDENT,
            alpha=0.42,
            linewidths=0.42,
            rasterized=True,
            zorder=5,
        )
        _flat_panel(ax_c)
        _panel_title(
            ax_c,
            "Student-selected target",
            f"{student_dim}-D",
            SELECTED,
        )

        # Each visual question gets a readable crop. Every overlay inside that crop
        # still shares one projection and one numerical scale.
        _shared_square_window((ax_b,), [reduced, *ghosts])
        _shared_square_window((ax_c,), [before, student, selected])
        _flow_arrow(figure, ax_a, ax_b, REDUCE)
        _flow_arrow(figure, ax_b, ax_c, SELECTED)


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
        }
        figure.savefig(paths["pdf"], metadata=metadata)
        figure.savefig(paths["svg"])
        figure.savefig(paths["png"])
        if show:
            plt.show()
        return figure, paths
