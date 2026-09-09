"""Publication renderer for notebook 01's interface-geometry figure.

The renderer deliberately receives already-computed arrays.  All scientific
choices (fit set, PCA, Procrustes, and viewing frames) stay in the notebook;
this module owns only layout, visual semantics, and paper-safe export.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
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
FIGURE_HEIGHT_IN = 3.30


def _writer(ax):
    """Return an axes-coordinate text method for 2-D and 3-D axes."""

    return getattr(ax, "text2D", ax.text)


def _panel_title(ax, letter: str, title: str, kicker: str, accent: str) -> None:
    ax.set_title(
        f"({letter})  {title}",
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
    """Apply one honest coordinate window to every supplied 2-D panel."""

    points = np.vstack(clouds)
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    half = max(float((high - low).max()) * (1 + pad) / 2, 1e-6)
    for ax in axes:
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
    return half


def _flow_arrow(figure, left_ax, right_ax, label: str, color: str) -> None:
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
    figure.text(
        (x0 + x1) / 2,
        y + 0.018,
        label,
        ha="center",
        va="bottom",
        fontsize=5.2,
        fontweight="semibold",
        color=color,
    )


def _smooth_histogram(
    values: np.ndarray, bins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(values, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    # Small Gaussian convolution avoids depending on scipy while preserving a
    # faithful empirical density at the scale printed in the paper.
    kernel_x = np.linspace(-2.5, 2.5, 13)
    kernel = np.exp(-0.5 * kernel_x**2)
    kernel /= kernel.sum()
    smooth = np.convolve(counts, kernel, mode="same")
    return centers, smooth


def render_interface_geometry(
    *,
    view: Mapping[str, object],
    plot_labels: Sequence[str],
    label_order: Sequence[str],
    cosines: Mapping[str, np.ndarray],
    teacher_dim: int,
    student_dim: int,
    explained_energy: float,
    gauge_stats: Mapping[str, float],
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
            2,
            3,
            height_ratios=(2.15, 0.92),
            left=0.045,
            right=0.985,
            top=0.925,
            bottom=0.115,
            wspace=0.22,
            hspace=0.42,
        )
        ax_a = figure.add_subplot(grid[0, 0], projection="3d")
        ax_b = figure.add_subplot(grid[0, 1])
        ax_c = figure.add_subplot(grid[0, 2])
        ax_d = figure.add_subplot(grid[1, :])

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
        for axis_index, label in enumerate((r"$t_1$", r"$t_2$", r"$t_3$")):
            direction = np.zeros(3)
            direction[axis_index] = length
            ax_a.quiver(
                *base,
                *direction,
                color=MUTED,
                linewidth=0.58,
                arrow_length_ratio=0.14,
            )
            endpoint = base + 1.12 * direction
            ax_a.text(*endpoint, label, color=MUTED, fontsize=6.0)
        _panel_title(
            ax_a,
            "a",
            "High-dimensional teacher",
            f"RAW  •  $d_T={teacher_dim}$",
            REDUCE,
        )

        # (b) A single coloured realization sits above monochrome orbit samples.
        ghosts = [np.asarray(cloud) for cloud in view["ghosts"]]
        reduced = np.asarray(view["Y"])
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
            "b",
            "Coordinate orbit",
            f"REDUCE  •  $d_S={student_dim}$  •  {explained_energy:.1%} ENERGY",
            REDUCE,
        )
        ax_b.text(
            0.96,
            0.055,
            r"$YY^{\top}=(YQ)(YQ)^{\top}$",
            transform=ax_b.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.0,
            color=REDUCE,
            bbox={
                "boxstyle": "round,pad=0.24",
                "facecolor": "white",
                "edgecolor": HAIR,
                "linewidth": 0.45,
            },
            zorder=20,
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
        midpoint_b = (start_b + end_b) / 2
        ax_b.annotate(
            r"$Q\in O(d_S)$",
            xy=midpoint_b,
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=6.1,
            color=REDUCE,
            zorder=9,
        )

        # (c) Marker shape separates student x from target dots even in grayscale.
        student = np.asarray(view["Z"])
        selected = np.asarray(view["YR"])
        before = reduced
        n_links = min(36, len(student))
        link_indices = np.linspace(0, len(student) - 1, n_links, dtype=int)
        segments = np.stack((selected[link_indices], student[link_indices]), axis=1)
        ax_c.add_collection(
            LineCollection(
                segments,
                colors=STUDENT,
                linewidths=0.28,
                alpha=0.17,
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
            s=7.0,
            marker="x",
            c=STUDENT,
            alpha=0.50,
            linewidths=0.42,
            rasterized=True,
            zorder=5,
        )
        _flat_panel(ax_c)
        _panel_title(
            ax_c,
            "c",
            "Student-selected target",
            r"ALIGN  •  ONE CORPUS-LEVEL $R^{\star}$",
            SELECTED,
        )
        legend_handles = (
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=3.0,
                markerfacecolor=SELECTED,
                markeredgewidth=0,
                label=r"target $YR^{\star}$",
            ),
            Line2D(
                [],
                [],
                marker="x",
                linestyle="none",
                markersize=3.2,
                color=STUDENT,
                markeredgewidth=0.65,
                label=r"student $Z$",
            ),
        )
        ax_c.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.985, 0.865),
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.88,
            borderaxespad=0.30,
            handletextpad=0.20,
            labelspacing=0.18,
        )
        ax_c.text(
            0.04,
            0.055,
            r"$R^{\star}=\arg\min_{R\in O(d_S)}\|YR-Z\|_F^2$",
            transform=ax_c.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.0,
            color=SELECTED,
            bbox={
                "boxstyle": "round,pad=0.24",
                "facecolor": "white",
                "edgecolor": HAIR,
                "linewidth": 0.45,
            },
            zorder=20,
        )

        # The same view and the same numerical scale are mandatory for (b)/(c).
        _shared_square_window((ax_b, ax_c), [reduced, student, selected, *ghosts])
        _flow_arrow(figure, ax_a, ax_b, "PCA", REDUCE)
        _flow_arrow(figure, ax_b, ax_c, r"$R^{\star}$", SELECTED)

        # (d) Compact ridgelines: distributional evidence, not another projection.
        distribution_order = (
            "PCA only",
            "PCA + Haar $Q$",
            r"PCA + Procrustes $R^{\star}$",
        )
        distribution_labels = ("PCA only", r"Haar $Q$", r"Selected $R^{\star}$")
        distribution_colours = ("#B76A35", "#9EA4AE", SELECTED)
        stacked = np.concatenate(
            [np.asarray(cosines[name]) for name in distribution_order]
        )
        edge = min(
            1.0, 0.035 + max(abs(float(stacked.min())), abs(float(stacked.max())))
        )
        bins = np.linspace(-edge, edge, 120)
        baselines = np.asarray((2.0, 1.0, 0.0))
        means: list[float] = []

        ax_d.axvline(0, color=HAIR, linewidth=0.60, zorder=0)
        for baseline, name, colour in zip(
            baselines, distribution_order, distribution_colours, strict=True
        ):
            values = np.asarray(cosines[name])
            centers, density = _smooth_histogram(values, bins)
            height = 0.58 * density / max(float(density.max()), 1e-12)
            ax_d.fill_between(
                centers,
                baseline,
                baseline + height,
                color=colour,
                alpha=0.28,
                linewidth=0,
                zorder=2,
            )
            ax_d.plot(
                centers, baseline + height, color=colour, linewidth=0.78, zorder=3
            )
            ax_d.hlines(
                baseline, bins[0], bins[-1], color=GRID, linewidth=0.42, zorder=1
            )
            mean = float(values.mean())
            means.append(mean)
            ax_d.plot(
                (mean, mean),
                (baseline, baseline + 0.64),
                color=colour,
                linewidth=0.70,
                linestyle=(0, (1.4, 1.2)),
                zorder=4,
            )
            ax_d.scatter(
                mean,
                baseline + 0.64,
                s=8.5,
                color=colour,
                edgecolor="white",
                linewidth=0.35,
                zorder=5,
            )
            ax_d.annotate(
                f"{mean:+.3f}",
                xy=(mean, baseline + 0.64),
                xytext=(0, 2.5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=5.7,
                fontweight="semibold",
                color=colour,
            )

        ax_d.set_xlim(bins[0], bins[-1])
        ax_d.set_ylim(-0.10, 2.78)
        ax_d.set_yticks([])
        label_x = bins[0] + 0.012 * (bins[-1] - bins[0])
        for baseline, label, colour in zip(
            baselines, distribution_labels, distribution_colours, strict=True
        ):
            ax_d.text(
                label_x,
                baseline + 0.07,
                label,
                ha="left",
                va="bottom",
                fontsize=6.1,
                fontweight="semibold",
                color=colour,
                zorder=7,
            )
        tick_start = np.ceil(bins[0] / 0.25) * 0.25
        ax_d.set_xticks(np.arange(tick_start, bins[-1] + 1e-9, 0.25))
        ax_d.tick_params(axis="x", colors=MUTED, length=2.0, width=0.55, pad=1.5)
        for name, spine in ax_d.spines.items():
            spine.set_visible(name == "bottom")
            spine.set_color(HAIR)
            spine.set_linewidth(0.55)
        ax_d.set_xlabel(
            r"per-sentence cosine  $\cos(z_i,\tau_i)$", color=MUTED, labelpad=1.5
        )
        ax_d.set_title(
            "(d)  Endpoint compatibility",
            loc="left",
            fontsize=8.3,
            fontweight="semibold",
            color=INK,
            pad=3,
        )
        ax_d.text(
            0.995,
            1.06,
            f"original {student_dim}-D space  •  $n={int(gauge_stats['samples']):,}$  •  "
            f"cross-covariance PR {float(gauge_stats['participation_ratio']):.2f}",
            transform=ax_d.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.6,
            color=MUTED,
        )
        gain = means[2] - means[0]
        arrow_y = 2.63
        ax_d.annotate(
            "",
            xy=(means[2], arrow_y),
            xytext=(means[0], arrow_y),
            arrowprops={
                "arrowstyle": "-|>",
                "color": SELECTED,
                "linewidth": 0.75,
                "shrinkA": 0,
                "shrinkB": 0,
                "mutation_scale": 6,
            },
            zorder=8,
        )
        ax_d.text(
            (means[0] + means[2]) / 2,
            arrow_y + 0.06,
            f"{gain:+.3f} mean cosine",
            ha="center",
            va="bottom",
            fontsize=5.7,
            fontweight="semibold",
            color=SELECTED,
            zorder=9,
        )

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
