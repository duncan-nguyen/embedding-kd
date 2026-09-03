#!/usr/bin/env python3
"""Render paper-figure mockups with synthetic data for layout review.

Every output is visibly marked as mock data. The values in these figures must not
be used as experimental evidence or copied into the paper's result tables.
"""

from __future__ import annotations

import os
from pathlib import Path

MPL_CACHE = Path("/tmp/embedding-kd-matplotlib")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figure_mockups" / "final"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(17)

BLUE = "#2B6CB0"
ORANGE = "#DD6B20"
GREEN = "#2F855A"
PURPLE = "#6B46C1"
RED = "#C53030"
GRAY = "#6B7280"
LIGHT_GRAY = "#E5E7EB"
INK = "#1F2937"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    }
)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )


def clean_axis(ax: plt.Axes, grid: bool = True) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6, zorder=0)


def add_mock_stamp(fig: plt.Figure) -> None:
    fig.text(
        0.995,
        0.004,
        "MOCK DATA — LAYOUT REVIEW ONLY",
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=RED,
        fontweight="bold",
        alpha=0.9,
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    add_mock_stamp(fig)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 7.5,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.0,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = GRAY,
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.0,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


def figure_method_overview() -> None:
    fig, ax = plt.subplots(figsize=(5.5, 2.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    top_lane = FancyBboxPatch(
        (0.01, 0.52), 0.98, 0.43, boxstyle="round,pad=0.006,rounding_size=0.015",
        facecolor="#FFF9F5", edgecolor="#F2C6A5", linewidth=0.8,
    )
    bottom_lane = FancyBboxPatch(
        (0.01, 0.08), 0.98, 0.36, boxstyle="round,pad=0.006,rounding_size=0.015",
        facecolor="#FAF8FE", edgecolor="#CDBBEF", linewidth=0.8,
    )
    ax.add_patch(top_lane)
    ax.add_patch(bottom_lane)
    ax.text(0.025, 0.83, "EXTRINSIC\nINTERFACE", color=ORANGE, fontsize=6.5, fontweight="bold", va="top")
    ax.text(0.025, 0.35, "INTRINSIC\nSTRUCTURE", color=PURPLE, fontsize=6.5, fontweight="bold", va="top")

    draw_box(ax, (0.17, 0.72), 0.15, 0.13, "Teacher $Z_T$", "#E8F1FB", BLUE, 7.2)
    draw_box(ax, (0.38, 0.72), 0.17, 0.13, "PCA target\n$Z_TP$", "#E8F1FB", BLUE, 7.0)
    draw_box(ax, (0.62, 0.72), 0.17, 0.13, "Aligned target\n$Z_TPR^{(e)}$", "#FFF1E7", ORANGE, 6.8)
    draw_box(ax, (0.62, 0.55), 0.17, 0.11, "Student $Z_S^{(e)}$", "#ECF8F1", GREEN, 6.9)
    draw_box(ax, (0.86, 0.63), 0.105, 0.18, "$L_{end}$", "#FFF1E7", ORANGE, 8.0)
    arrow(ax, (0.32, 0.785), (0.38, 0.785), BLUE)
    arrow(ax, (0.55, 0.785), (0.62, 0.785), ORANGE)
    arrow(ax, (0.79, 0.785), (0.86, 0.73), ORANGE)
    arrow(ax, (0.79, 0.605), (0.86, 0.69), GREEN)
    arrow(ax, (0.705, 0.66), (0.705, 0.72), ORANGE)
    ax.text(0.35, 0.82, "fit once", color=BLUE, fontsize=6.0, ha="center")
    ax.text(0.60, 0.535, "fit $R^{(e)}$ each epoch on one frozen subset", color=ORANGE, fontsize=6.2, ha="center")

    draw_box(ax, (0.17, 0.27), 0.15, 0.105, "Teacher $Z_T$", "#E8F1FB", BLUE, 7.0)
    draw_box(ax, (0.17, 0.12), 0.15, 0.105, "Student $Z_S$", "#ECF8F1", GREEN, 7.0)
    draw_box(ax, (0.40, 0.27), 0.20, 0.105, r"$H_0$ deaths $\delta_T$", "#F2ECFC", PURPLE, 7.0)
    draw_box(ax, (0.40, 0.12), 0.20, 0.105, r"$H_0$ deaths $\delta_S$", "#F2ECFC", PURPLE, 7.0)
    draw_box(ax, (0.70, 0.175), 0.12, 0.14, "$L_{H_0}$", "#F2ECFC", PURPLE, 8.0)
    arrow(ax, (0.32, 0.322), (0.40, 0.322), PURPLE)
    arrow(ax, (0.32, 0.172), (0.40, 0.172), PURPLE)
    arrow(ax, (0.60, 0.322), (0.70, 0.265), PURPLE)
    arrow(ax, (0.60, 0.172), (0.70, 0.225), PURPLE)
    ax.text(0.905, 0.245, "Final objective\n" + r"$L_{end}+\lambda L_{H_0}$", ha="center", va="center", fontsize=6.3, color=INK)
    save_figure(fig, "figure_1_method_overview")


def figure_a1_same_geometry_interface() -> None:
    fig = plt.figure(figsize=(5.5, 4.05))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25], hspace=0.53, wspace=0.48)
    gram_gs = gs[0, 0:2].subgridspec(1, 2, wspace=0.08)
    ax0 = fig.add_subplot(gram_gs[0, 0])
    ax1 = fig.add_subplot(gram_gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, :])

    features = RNG.normal(size=(18, 8))
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    q, _ = np.linalg.qr(RNG.normal(size=(8, 8)))
    rotated = features @ q
    gram = features @ features.T
    gram_rotated = rotated @ rotated.T
    vmax = 1.0
    im = ax0.imshow(gram, cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax1.imshow(gram_rotated, cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
    for ax, title in ((ax0, "Original $G$"), (ax1, "Rotated $G_Q$")):
        ax.set_title(title)
        ax.set_xticks([0, 17], [1, 18])
        ax.set_yticks([0, 17], [1, 18])
    ax1.set_yticklabels([])
    ax0.set_ylabel("Example index")
    ax0.set_xlabel("Example")
    ax1.set_xlabel("Example")
    cbar = fig.colorbar(im, ax=[ax0, ax1], orientation="horizontal", fraction=0.055, pad=0.16, aspect=28)
    cbar.set_label("Cosine Gram", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)
    max_diff = np.max(np.abs(gram - gram_rotated))
    ax0.text(-0.14, 1.10, "(a)", transform=ax0.transAxes, fontsize=9, fontweight="bold", va="top")
    ax1.text(
        0.98, 0.03, f"CKA = 1.000\nmax |ΔG| = {max_diff:.1e}", transform=ax1.transAxes,
        ha="right", va="bottom", fontsize=6.0, color="white",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "black", "alpha": 0.48, "edgecolor": "none"},
    )

    rotation_scores = [
        RNG.normal(75.25, 0.48, 24),
        RNG.normal(76.05, 0.22, 24),
        RNG.normal(76.56, 0.12, 24),
    ]
    parts = ax2.violinplot(rotation_scores, positions=[1, 2, 3], showmeans=False, showmedians=True, widths=0.72)
    for body, color in zip(parts["bodies"], [GRAY, BLUE, ORANGE]):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.42)
    for key in ("cmins", "cmaxes", "cbars", "cmedians"):
        parts[key].set_color(INK)
        parts[key].set_linewidth(0.8)
    ax2.set_xticks([1, 2, 3], ["Haar\nrotation", "Fit\nonce", "Refit /\nepoch"])
    ax2.set_ylabel("Final average score")
    ax2.set_title("Same geometry, different KD")
    clean_axis(ax2)
    panel_label(ax2, "(b)")

    epochs = 5
    fixed_x = np.linspace(0, epochs, 151)
    fixed_loss = 0.78 * np.exp(-fixed_x / 2.45) + 0.17 + 0.010 * np.sin(4.2 * fixed_x)
    refit_x: list[float] = []
    refit_loss: list[float] = []
    starts = [0.95, 0.47, 0.27, 0.18, 0.13]
    ends = [0.61, 0.34, 0.22, 0.15, 0.11]
    post_refit = [0.45, 0.25, 0.17, 0.12]
    for epoch in range(epochs):
        local_x = np.linspace(epoch, epoch + 1, 28)
        local_t = local_x - epoch
        local_loss = starts[epoch] - (starts[epoch] - ends[epoch]) * (1 - np.exp(-3 * local_t)) / (1 - np.exp(-3))
        refit_x.extend(local_x.tolist())
        refit_loss.extend(local_loss.tolist())
        if epoch < epochs - 1:
            refit_x.append(float(epoch + 1))
            refit_loss.append(post_refit[epoch])
    ax3.plot(fixed_x, fixed_loss, color=GRAY, linestyle="--", label="Fit once")
    ax3.plot(refit_x, refit_loss, color=ORANGE, label="Refit every epoch")
    for epoch, after in enumerate(post_refit, start=1):
        ax3.axvline(epoch, color=LIGHT_GRAY, linewidth=0.8, zorder=0)
        ax3.annotate("", xy=(epoch, after), xytext=(epoch, ends[epoch - 1]),
                     arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": 1.0})
    ax3.text(1.04, 0.50, "exact refit", color=ORANGE, fontsize=6.3)
    ax3.set_xlim(0, epochs)
    ax3.set_ylim(0.07, 1.0)
    ax3.set_xticks(np.arange(epochs + 1))
    ax3.set_xlabel("Training epoch")
    ax3.set_ylabel("Endpoint loss on frozen probe ↓")
    ax3.set_title("Refitting changes the interface immediately—not the teacher geometry")
    clean_axis(ax3)
    ax3.legend(frameon=False, loc="upper right", ncol=2)
    panel_label(ax3, "(c)")

    fig.suptitle("A1. Same geometry, different interface", fontsize=10, fontweight="bold", y=1.01)
    save_figure(fig, "figure_A1_same_geometry_interface")


def figure_a2_teacher_redundancy() -> None:
    fig = plt.figure(figsize=(5.35, 3.75))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.15], hspace=0.55, wspace=0.38)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])

    rank = np.arange(1, 2561)
    spectrum = 0.018 * np.exp(-rank / 120) + 0.0013 * np.exp(-rank / 700) + 3e-6
    spectrum *= 1 + 0.025 * np.sin(rank / 19)
    cumulative = np.cumsum(spectrum) / np.sum(spectrum)
    ax0.plot(rank, cumulative, color=BLUE)
    ax0.axvline(384, color=ORANGE, linestyle="--", linewidth=1.1)
    retained_384 = cumulative[383]
    ax0.plot([384], [retained_384], "o", color=ORANGE, markersize=3.5)
    ax0.text(425, retained_384 - 0.08, f"$d_S=384$\n{retained_384:.0%} retained", color=ORANGE, fontsize=6.3)
    ax0.set_xlabel("Principal component")
    ax0.set_ylabel("Cumulative variance")
    ax0.set_ylim(0, 1.02)
    ax0.set_title("Most energy lies in a compact subspace")
    clean_axis(ax0)
    panel_label(ax0, "(a)")

    dims = np.array([64, 128, 256, 384, 768])
    pca_knn = np.array([0.43, 0.58, 0.73, 0.82, 0.91])
    rnd_knn = np.array([0.25, 0.37, 0.52, 0.62, 0.76])
    rnd_knn_std = np.array([0.035, 0.032, 0.026, 0.022, 0.016])
    ax1.plot(dims, pca_knn, marker="o", color=BLUE, label="PCA")
    ax1.plot(dims, rnd_knn, marker="s", color=GRAY, label="Random projection")
    ax1.fill_between(dims, rnd_knn - rnd_knn_std, rnd_knn + rnd_knn_std, color=GRAY, alpha=0.18, linewidth=0)
    ax1.axvline(384, color=ORANGE, linestyle="--", linewidth=1.0)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(dims, ["64", "128", "256", "384", "768"])
    ax1.set_ylim(0.15, 1.0)
    ax1.set_xlabel("Projected dimension")
    ax1.set_ylabel("kNN overlap@10")
    ax1.set_title("Local geometry survives compression")
    clean_axis(ax1)
    ax1.legend(frameon=False, loc="lower right")
    panel_label(ax1, "(b)")

    pca_score = np.array([74.2, 75.5, 76.3, 76.65, 76.82])
    rnd_score = np.array([71.6, 72.9, 74.0, 74.7, 75.6])
    rnd_score_std = np.array([0.38, 0.32, 0.24, 0.20, 0.14])
    ax2.plot(dims, pca_score, marker="o", color=BLUE, label="PCA")
    ax2.plot(dims, rnd_score, marker="s", color=GRAY, label="Random projection")
    ax2.fill_between(dims, rnd_score - rnd_score_std, rnd_score + rnd_score_std, color=GRAY, alpha=0.18, linewidth=0)
    ax2.axhline(76.9, color=INK, linestyle=":", linewidth=1.0, label="Full teacher")
    ax2.axvline(384, color=ORANGE, linestyle="--", linewidth=1.0)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(dims, ["64", "128", "256", "384", "768"])
    ax2.set_ylim(70.8, 77.2)
    ax2.set_xlabel("Projected dimension")
    ax2.set_ylabel("Teacher downstream score")
    ax2.set_title("PCA retains downstream utility better than an equally narrow random subspace")
    clean_axis(ax2)
    ax2.legend(frameon=False, loc="lower right", ncol=3)
    panel_label(ax2, "(c)")

    fig.suptitle("A2. How redundant is the teacher space?", fontsize=10, fontweight="bold", y=1.015)
    fig.text(0.02, 0.026, "PCA: full teacher cache. Random projection: mean ± SD over 10 draws.", ha="left", fontsize=6.5, color=GRAY)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.87, hspace=0.72, wspace=0.42)
    save_figure(fig, "figure_A2_teacher_redundancy")


def prim_edges(distance: np.ndarray) -> list[tuple[int, int]]:
    mst = minimum_spanning_tree(distance).toarray()
    rows, cols = np.nonzero(mst)
    return list(zip(rows.tolist(), cols.tolist()))


def mst_deaths(distance: np.ndarray) -> np.ndarray:
    mst = minimum_spanning_tree(distance).toarray()
    return np.sort(mst[mst > 0])


def plot_mst(ax: plt.Axes, xy: np.ndarray, distance: np.ndarray, title: str, color: str) -> None:
    for i, j in prim_edges(distance):
        ax.plot(
            [xy[i, 0], xy[j, 0]],
            [xy[i, 1], xy[j, 1]],
            color=color,
            linewidth=0.9,
            alpha=0.82,
            zorder=1,
        )
    ax.plot(xy[:, 0], xy[:, 1], "o", ms=2.6, color=INK, markeredgewidth=0, zorder=2)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")


def figure_a3_h0_connectivity() -> None:
    centers = np.array([[-1.25, 0.0], [0.15, 1.05], [1.2, -0.25]])
    xy = np.vstack([c + RNG.normal(0, 0.26, (12, 2)) for c in centers])
    teacher_dist = cdist(xy, xy)
    endpoint_xy = xy @ np.array([[1.14, 0.22], [0.0, 0.75]]) + RNG.normal(0, 0.15, xy.shape)
    topo_xy = xy @ np.array([[1.02, 0.05], [-0.03, 0.96]]) + RNG.normal(0, 0.055, xy.shape)

    fig = plt.figure(figsize=(5.5, 5.35))
    gs = fig.add_gridspec(3, 6, height_ratios=[0.82, 1.08, 1.0], hspace=0.62, wspace=0.48)
    top_axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6])]
    ax_spectrum = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, 0:3])
    ax4 = fig.add_subplot(gs[2, 3:6])

    plot_mst(top_axes[0], xy, teacher_dist, "Teacher MST", PURPLE)
    plot_mst(top_axes[1], xy, cdist(endpoint_xy, endpoint_xy), "$L_{end}$ only", ORANGE)
    plot_mst(top_axes[2], xy, cdist(topo_xy, topo_xy), "$L_{end}+L_{H_0}$", GREEN)
    top_axes[0].text(-0.08, 1.08, "(a)", transform=top_axes[0].transAxes, fontsize=9, fontweight="bold")

    batches, edge_rank = 48, 35
    q = np.linspace(0, 1, edge_rank)
    endpoint_residual = 0.075 * np.sin(np.pi * q)[None, :] + RNG.normal(0, 0.025, (batches, edge_rank))
    endpoint_residual += np.linspace(-0.018, 0.018, batches)[:, None]
    topo_residual = 0.014 * np.sin(np.pi * q)[None, :] + RNG.normal(0, 0.010, (batches, edge_rank))
    teacher_spectra = 0.055 + 0.82 * q[None, :] ** 1.58 + RNG.normal(0, 0.010, (batches, edge_rank))
    teacher_spectra = np.sort(np.clip(teacher_spectra, 0.02, None), axis=1)
    endpoint_spectra = np.sort(teacher_spectra + endpoint_residual, axis=1)
    topo_spectra = np.sort(teacher_spectra + topo_residual, axis=1)

    rank_axis = np.arange(1, edge_rank + 1)
    for values, color, label, linestyle in (
        (teacher_spectra, INK, "Teacher", "-"),
        (endpoint_spectra, ORANGE, "$L_{end}$ only", "--"),
        (topo_spectra, GREEN, "$L_{end}+L_{H_0}$", "-"),
    ):
        median = np.median(values, axis=0)
        low, high = np.quantile(values, [0.25, 0.75], axis=0)
        ax_spectrum.plot(rank_axis, median, color=color, linestyle=linestyle, label=label)
        ax_spectrum.fill_between(rank_axis, low, high, color=color, alpha=0.12, linewidth=0)
    ax_spectrum.set_xlim(1, edge_rank)
    ax_spectrum.set_xlabel("Sorted $H_0$ death rank")
    ax_spectrum.set_ylabel("Death time / MST edge length")
    ax_spectrum.set_title("The topology term directly matches the persistence death spectrum")
    clean_axis(ax_spectrum)
    ax_spectrum.legend(frameon=False, ncol=3, loc="upper left")
    panel_label(ax_spectrum, "(b)")

    norm = TwoSlopeNorm(vmin=-0.12, vcenter=0.0, vmax=0.12)
    ax3.imshow(endpoint_residual, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
    im = ax4.imshow(topo_residual, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
    ax3.set_title("MST edge residual: $L_{end}$ only")
    ax4.set_title("Residual after adding $H_0$")
    for ax in (ax3, ax4):
        ax.set_xlabel("Sorted MST edge rank")
        ax.set_xticks([0, 17, 34], [1, 18, 35])
        ax.set_yticks([0, 23, 47], [1, 24, 48])
    ax3.set_ylabel("Evaluation mini-batch")
    ax4.set_yticklabels([])
    panel_label(ax3, "(c)")
    cbar = fig.colorbar(im, ax=[ax3, ax4], fraction=0.022, pad=0.025)
    cbar.set_label(r"$\delta^S-\delta^T$", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.suptitle("A3. What does $H_0$ preserve?", fontsize=10, fontweight="bold", y=0.99)
    fig.text(0.5, 0.685, "One fixed display layout; every MST is recomputed from its native embedding space.", ha="center", fontsize=6.2, color=GRAY)
    fig.text(0.02, 0.020, "$H_0$ matches sorted death times—not corresponding MST edge identities.", ha="left", fontsize=6.3, color=GRAY)
    save_figure(fig, "figure_A3_h0_connectivity")


def cka_surface(kind: str, n_teacher: int = 24, n_student: int = 12) -> np.ndarray:
    teacher = np.linspace(0, 1, n_teacher)[:, None]
    student = np.linspace(0, 1, n_student)[None, :]
    diagonal = np.exp(-((teacher - student) ** 2) / (2 * 0.12**2))
    endpoint = np.exp(-((teacher - 0.93) ** 2) / (2 * 0.11**2)) * np.exp(-((student - 0.93) ** 2) / (2 * 0.16**2))
    if kind == "init":
        value = 0.10 + 0.13 * diagonal + 0.06 * RNG.random((n_teacher, n_student))
    elif kind == "endpoint":
        value = 0.10 + 0.24 * diagonal + 0.50 * endpoint + 0.035 * RNG.random((n_teacher, n_student))
    elif kind == "topo":
        broad = np.exp(-((teacher - student) ** 2) / (2 * 0.21**2))
        value = 0.12 + 0.19 * broad + 0.48 * endpoint + 0.035 * RNG.random((n_teacher, n_student))
    elif kind == "talas":
        value = 0.10 + 0.68 * diagonal + 0.18 * endpoint + 0.025 * RNG.random((n_teacher, n_student))
    else:
        raise ValueError(kind)
    return np.clip(value, 0, 1)


def figure_a4_talas_cka() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.35), sharex=True, sharey=True)
    specs = [
        ("Initialized student\nbefore KD", "init"),
        ("Endpoint only\nscore 76.12", "endpoint"),
        ("Endpoint + $H_0$\nscore 76.55", "topo"),
        ("TALAS\nscore 76.31", "talas"),
    ]
    im = None
    for i, (ax, (title, kind)) in enumerate(zip(axes.flat, specs)):
        mean_cka = np.mean([cka_surface(kind) for _ in range(3)], axis=0)
        im = ax.imshow(mean_cka, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks([0, 5, 11], [1, 6, 12])
        ax.set_yticks([0, 11, 23], [1, 12, 24])
        panel_label(ax, f"({chr(97 + i)})")
    for ax in axes[1, :]:
        ax.set_xlabel("Student layer")
    for ax in axes[:, 0]:
        ax.set_ylabel("Teacher layer")
    assert im is not None
    fig.subplots_adjust(left=0.11, right=0.87, bottom=0.13, top=0.86, hspace=0.40, wspace=0.23)
    cax = fig.add_axes([0.90, 0.18, 0.025, 0.62])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Linear CKA", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    fig.suptitle("A4. Endpoint alignment need not reproduce TALAS's layerwise path", fontsize=10, fontweight="bold", y=0.995)
    fig.text(0.5, 0.025, "Linear CKA mean over 3 seeds on the same held-out probe set.", ha="center", fontsize=6.4, color=GRAY)
    save_figure(fig, "figure_A4_talas_cka")


def curve_with_band(
    ax: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    color: str,
    xlabel: str,
    title: str,
    log_x: bool = False,
) -> None:
    ax.plot(x, mean, marker="o", markersize=3.6, color=color)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
    if log_x:
        ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Average score")
    ax.set_title(title)
    clean_axis(ax)


def figure_a5_sensitivity() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.25))
    ax0, ax1, ax2, ax3 = axes.flat

    lambdas = np.array([0.0, 0.1, 0.25, 0.5, 1.0, 2.0])
    lambda_mean = np.array([75.63, 76.12, 76.43, 76.55, 76.49, 76.02])
    lambda_std = np.array([0.12, 0.10, 0.09, 0.08, 0.10, 0.15])
    curve_with_band(ax0, lambdas, lambda_mean, lambda_std, BLUE, r"$\lambda_{H_0}$", "Topology weight")
    ax0.axvline(0.5, color=ORANGE, linestyle="--", linewidth=1)
    panel_label(ax0, "(a)")

    batches = np.array([64, 128, 256, 512])
    batch_mean = np.array([75.94, 76.31, 76.55, 76.58])
    batch_std = np.array([0.20, 0.13, 0.08, 0.08])
    curve_with_band(ax1, batches, batch_mean, batch_std, GREEN, r"$H_0$ batch size", "Topology estimation", log_x=True)
    ax1.set_xticks(batches, [str(v) for v in batches])
    panel_label(ax1, "(b)")

    samples = np.array([512, 1024, 2048, 4096, 8192, 16384])
    sample_mean = np.array([75.71, 75.99, 76.25, 76.46, 76.53, 76.55])
    sample_std = np.array([0.20, 0.17, 0.12, 0.09, 0.08, 0.08])
    curve_with_band(ax2, samples, sample_mean, sample_std, PURPLE, "Fixed gauge calibration samples", "Gauge sample sensitivity", log_x=True)
    ax2.set_xticks([512, 2048, 8192, 16384], ["512", "2k", "8k", "16k"])
    ax2.axvline(16384, color=ORANGE, linestyle="--", linewidth=1)
    ax2.text(15500, 75.76, "default", rotation=90, color=ORANGE, fontsize=6.0, ha="right")
    panel_label(ax2, "(c)")

    schedules = ["Fit once", "Every 2\nepochs", "Every\nepoch"]
    values = np.array([-0.62, -0.24, 0.0])
    errors = np.array([0.13, 0.10, 0.0])
    colors = [GRAY, "#9CA3AF", ORANGE]
    ax3.axhline(0, color=INK, linewidth=0.8)
    ax3.bar(np.arange(3), values, yerr=errors, color=colors, width=0.62, capsize=2, zorder=2)
    ax3.set_xticks(np.arange(3), schedules)
    ax3.set_ylim(-0.9, 0.22)
    ax3.set_ylabel("Score delta vs. every epoch")
    ax3.set_title("Gauge refit schedule")
    clean_axis(ax3)
    ax3.text(0.01, 0.98, "(d)", transform=ax3.transAxes, fontsize=9, fontweight="bold", va="top")

    fig.suptitle("A5. Sensitivity and stability", fontsize=10, fontweight="bold", y=1.015)
    fig.text(0.02, 0.018, "Mean ± sample SD (3 seeds). Fit once = initial fit, no later refits.", ha="left", fontsize=6.3, color=GRAY)
    fig.tight_layout(rect=(0, 0.055, 1, 0.98), h_pad=1.4, w_pad=1.25)
    save_figure(fig, "figure_A5_sensitivity")


def figure_final_interface() -> None:
    """Render the two standalone interface figures retained in the final set."""
    fig, ax = plt.subplots(figsize=(5.5, 2.45))
    random_scores = RNG.normal(75.25, 0.48, 30)
    box = ax.boxplot([random_scores], positions=[1], widths=0.48, patch_artist=True, showfliers=False)
    box["boxes"][0].set(facecolor="#D4D8DE", edgecolor=GRAY)
    for item in box["medians"] + box["whiskers"] + box["caps"]:
        item.set(color=INK, linewidth=1.2)
    deterministic = {
        0: ("PCA gauge", 75.62, 0.18, GRAY),
        2: ("Fit once", 76.08, 0.16, BLUE),
        3: ("Refit / epoch", 76.56, 0.10, ORANGE),
    }
    for position, (_, mean, std, color) in deterministic.items():
        ax.errorbar(position, mean, yerr=std, fmt="o", ms=6, capsize=3, color=color, zorder=3)
    ax.set_xticks([0, 1, 2, 3], ["PCA gauge", "Haar\nrotations", "Fit once", "Refit /\nepoch"])
    ax.set_ylabel("Final average score")
    ax.set_ylim(74.0, 77.0)
    ax.text(
        0.02, 0.96, "All orthogonal gauges:\nlinear CKA = 1.000\nmax Gram error < 10⁻⁶",
        transform=ax.transAxes, va="top", fontsize=7, color=INK,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": LIGHT_GRAY, "alpha": 0.92},
    )
    clean_axis(ax)
    fig.tight_layout()
    save_figure(fig, "figure_2_same_geometry_different_kd")

    epochs = np.arange(1, 6)
    before = np.array([0.51, 0.36, 0.27, 0.22, 0.19])
    after = np.array([0.34, 0.22, 0.16, 0.12, 0.10])
    before_sd = np.array([0.035, 0.026, 0.020, 0.017, 0.015])
    after_sd = np.array([0.024, 0.018, 0.014, 0.011, 0.010])
    fig, ax = plt.subplots(figsize=(5.5, 2.45))
    offset = 0.055
    ax.errorbar(epochs - offset, before, yerr=before_sd, marker="o", capsize=2.5, color=GRAY, label="Before refit")
    ax.errorbar(epochs + offset, after, yerr=after_sd, marker="o", capsize=2.5, color=ORANGE, label="After refit")
    for epoch, y_before, y_after in zip(epochs, before, after):
        ax.plot([epoch - offset, epoch + offset], [y_before, y_after], color="#C7CBD1", linewidth=1.0, zorder=0)
    ax.set_xticks(epochs)
    ax.set_xlabel("Epoch boundary")
    ax.set_ylabel("Frozen-subset alignment error ↓")
    ax.set_ylim(0.05, 0.58)
    ax.legend(frameon=False, ncol=2)
    clean_axis(ax)
    fig.tight_layout()
    save_figure(fig, "figure_3_refit_restores_interface")


def figure_final_h0() -> None:
    """Render the retained quantitative and qualitative H0 figures separately."""
    batches, edge_rank = 32, 63
    q = np.linspace(0, 1, edge_rank)
    signed_endpoint = 0.070 * np.sin(np.pi * q)[None, :] + RNG.normal(0, 0.024, (batches, edge_rank))
    signed_endpoint += np.linspace(-0.012, 0.016, batches)[:, None]
    signed_topo = 0.012 * np.sin(np.pi * q)[None, :] + RNG.normal(0, 0.008, (batches, edge_rank))
    residuals = {
        "$L_{end}$ only": np.abs(signed_endpoint),
        "$L_{end}+L_{H_0}$": np.abs(signed_topo),
    }
    vmax = np.quantile(residuals["$L_{end}$ only"], 0.99)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3), sharex=True, sharey=True)
    image = None
    for ax, (name, values) in zip(axes, residuals.items()):
        image = ax.imshow(values, aspect="auto", cmap="Reds", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(name)
        ax.set_xlabel("Sorted $H_0$ death rank")
        ax.set_xticks([0, 31, 62], [1, 32, 63])
        ax.set_yticks([0, 15, 31], [1, 16, 32])
        ax.text(
            0.91, 0.94, f"median = {np.median(values):.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.8, color="#7F1D1D",
            bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "none", "alpha": 0.86},
        )
    axes[0].set_ylabel("Fixed evaluation mini-batch")
    assert image is not None
    fig.colorbar(image, ax=axes, label=r"$|\delta^S-\delta^T|$", fraction=0.035, pad=0.03)
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.21, top=0.88, wspace=0.10)
    save_figure(fig, "figure_4_h0_residual")

    centers = np.array([[-1.25, 0.0], [0.15, 1.05], [1.2, -0.25]])
    xy = np.vstack([c + RNG.normal(0, 0.26, (12, 2)) for c in centers])
    teacher_dist = cdist(xy, xy)
    endpoint_xy = xy @ np.array([[1.14, 0.22], [0.0, 0.75]]) + RNG.normal(0, 0.15, xy.shape)
    topo_xy = xy @ np.array([[1.02, 0.05], [-0.03, 0.96]]) + RNG.normal(0, 0.055, xy.shape)
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.8))
    plot_mst(axes[0], xy, teacher_dist, "Teacher MST", PURPLE)
    plot_mst(axes[1], xy, cdist(endpoint_xy, endpoint_xy), "$L_{end}$ only", ORANGE)
    plot_mst(axes[2], xy, cdist(topo_xy, topo_xy), "$L_{end}+L_{H_0}$", GREEN)
    fig.text(
        0.02, 0.050, "Fixed teacher layout; native-space edges; no edge-identity matching.",
        ha="left", fontsize=6.2, color=GRAY,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=0.6)
    save_figure(fig, "figure_A1_h0_mst")


def figure_final_sensitivity() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.15), sharey=True)
    specs = [
        (axes[0], np.array([0, 0.1, 0.3, 0.75, 1.0, 2.0]), np.array([75.63, 76.10, 76.42, 76.55, 76.48, 76.02]), np.array([.12, .10, .09, .08, .10, .15]), BLUE, r"$\lambda_{H_0}$", 0.75),
        (axes[1], np.array([32, 64, 128, 256]), np.array([75.72, 75.95, 76.31, 76.55]), np.array([.22, .19, .13, .08]), GREEN, r"$H_0$ batch size", 128),
        (axes[2], np.array([2048, 4096, 8192, 16384]), np.array([76.24, 76.44, 76.53, 76.55]), np.array([.12, .10, .08, .08]), PURPLE, "Frozen gauge samples", 16384),
    ]
    for index, (ax, values, mean, sd, color, xlabel, default) in enumerate(specs):
        x = np.arange(len(values))
        ax.plot(x, mean, marker="o", markersize=3.4, color=color)
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.18, linewidth=0)
        default_index = int(np.flatnonzero(values == default)[0])
        ax.axvline(default_index, color=ORANGE, linestyle="--", linewidth=1)
        labels = [f"{v:g}" for v in values] if index == 0 else [f"{int(v)}" for v in values] if index == 1 else [f"{int(v / 1024)}k" for v in values]
        ax.set_xticks(x, labels)
        ax.set_xlabel(xlabel)
        ax.set_title(f"({chr(97 + index)})")
        clean_axis(ax)
    axes[0].set_ylabel("Final average score")
    fig.text(0.02, 0.015, "Mean ± sample SD (3 seeds); dashed line marks the default.", fontsize=6.2, color=GRAY)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.7)
    save_figure(fig, "figure_A2_sensitivity")


def contact_sheet() -> None:
    stems = [
        "figure_1_method_overview",
        "figure_2_same_geometry_different_kd",
        "figure_3_refit_restores_interface",
        "figure_4_h0_residual",
        "figure_A1_h0_mst",
        "figure_A2_sensitivity",
    ]
    titles = ["Figure 1 — Method", "Figure 2 — Interface", "Figure 3 — Refit", "Figure 4 — $H_0$ residual", "Figure A1 — MST", "Figure A2 — Sensitivity"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 13.2))
    for ax, stem, title in zip(axes.flat, stems, titles):
        image = plt.imread(OUT / f"{stem}.png")
        ax.imshow(image)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")
    fig.suptitle("Paper figure mockups — synthetic data only", fontsize=16, fontweight="bold", color=RED, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=1.5, w_pad=1.0)
    fig.savefig(OUT / "contact_sheet.png", dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    figure_method_overview()
    figure_final_interface()
    figure_final_h0()
    figure_final_sensitivity()
    contact_sheet()
    print(f"Wrote mock figures to {OUT}")


if __name__ == "__main__":
    main()
