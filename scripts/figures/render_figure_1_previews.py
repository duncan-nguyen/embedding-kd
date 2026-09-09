#!/usr/bin/env python3
"""Render synthetic layout previews for Figure 1.

These images are for visual-direction review only.  They deliberately carry a
watermark and must never be used as experimental evidence in the paper.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/embedding-kd-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, ConnectionPatch, FancyBboxPatch
from scipy.sparse.csgraph import minimum_spanning_tree


OUTPUT_DIR = Path("docs/latex_iclr/figures/previews")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
COLORS = np.array(["#477DB3", "#F2A93B", "#E66C9C", "#D64B40", "#7057B1", "#2FA67E"])
INK = "#202124"
MUTED = "#69707D"
PANEL = "#FAFAF8"
TEACHER = "#B45F06"
TALAS = "#596579"
GATE = "#5B43B4"
RNG = np.random.default_rng(17)


def make_clouds(n_per_class: int = 100):
    centers = np.array(
        [[-2.3, 0.7], [2.2, 0.9], [1.0, -1.65], [-1.8, -1.45], [-0.35, 2.2], [2.65, -1.55]]
    )
    labels = np.repeat(np.arange(6), n_per_class)
    teacher = np.vstack(
        [RNG.normal(centers[index], [0.50, 0.45], size=(n_per_class, 2)) for index in range(6)]
    )
    # TALAS is intentionally diffuse only to make the visual direction legible.
    talas = teacher * np.array([0.64, 0.72]) + RNG.normal(0, [0.78, 0.66], teacher.shape)
    talas += np.column_stack([0.18 * teacher[:, 1], -0.12 * teacher[:, 0]])
    gate = teacher * np.array([0.93, 0.96]) + RNG.normal(0, [0.28, 0.25], teacher.shape)
    return labels, teacher, talas, gate


LABELS, TEACHER_XY, TALAS_XY, GATE_XY = make_clouds()


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "figure.dpi": 160,
            "savefig.dpi": 240,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def limits(*clouds):
    points = np.vstack(clouds)
    low = np.quantile(points, 0.005, axis=0)
    high = np.quantile(points, 0.995, axis=0)
    pad = 0.07 * (high - low)
    return (low[0] - pad[0], high[0] + pad[0]), (low[1] - pad[1], high[1] + pad[1])


XLIM, YLIM = limits(TEACHER_XY, TALAS_XY, GATE_XY)


def clean_axis(ax):
    ax.set_facecolor(PANEL)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def centroids(points):
    return np.stack([points[LABELS == label].mean(axis=0) for label in range(6)])


def mst_edges(points):
    means = centroids(points)
    distances = np.linalg.norm(means[:, None] - means[None, :], axis=-1)
    tree = minimum_spanning_tree(distances).toarray()
    edges = np.argwhere((tree + tree.T) > 0)
    return means, edges[edges[:, 0] < edges[:, 1]]


def draw_cloud(ax, points, accent, *, alpha=0.72, mst=True, label_centers=False):
    means, edges = mst_edges(points)
    if mst:
        for left, right in edges:
            ax.plot(means[[left, right], 0], means[[left, right], 1], color="white", lw=3.2, zorder=1)
            ax.plot(means[[left, right], 0], means[[left, right], 1], color=accent, lw=1.2, alpha=0.72, zorder=2)
    for label in range(6):
        mask = LABELS == label
        ax.scatter(points[mask, 0], points[mask, 1], s=8.5, color=COLORS[label], alpha=alpha, lw=0, zorder=3)
        ax.scatter(*means[label], s=36, color=COLORS[label], edgecolor="white", lw=1.0, zorder=4)
        if label_centers:
            ax.text(
                means[label, 0], means[label, 1] + 0.25, CLASS_NAMES[label],
                ha="center", fontsize=5.6, color=INK, weight="bold", zorder=5,
            )
    clean_axis(ax)


def legend(fig, x=0.43, y=0.025):
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[i], markeredgecolor="none", markersize=4.5, label=name)
        for i, name in enumerate(CLASS_NAMES)
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(x, y), ncol=6, frameon=False, fontsize=6.1)


def watermark(fig):
    fig.text(
        0.985, 0.012, "CONCEPT PREVIEW · SYNTHETIC LAYOUT ONLY",
        ha="right", va="bottom", fontsize=5.2, color="#A0A4AB", weight="bold",
    )


def save(fig, name):
    png_path = OUTPUT_DIR / f"{name}.png"
    pdf_path = OUTPUT_DIR / f"{name}.pdf"
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.035, facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.035, facecolor="white")
    plt.close(fig)
    print(png_path)
    print(pdf_path)


def preview_a():
    """Semantic constellations plus dimension-invariant residual textures."""
    fig = plt.figure(figsize=(7.45, 2.55), facecolor="white")
    grid = fig.add_gridspec(
        2, 5, width_ratios=[1, 1, 1, 0.73, 0.025], wspace=0.09, hspace=0.25
    )
    axes = [fig.add_subplot(grid[:, index]) for index in range(3)]
    configs = [
        ("a", "Teacher", "1024-D", TEACHER_XY, TEACHER),
        ("b", "TALAS", "384-D", TALAS_XY, TALAS),
        ("c", "GATE-KD", "384-D", GATE_XY, GATE),
    ]
    for index, (panel, name, dimension, points, accent) in enumerate(configs):
        draw_cloud(axes[index], points, accent)
        axes[index].set_title(
            f"({panel})  {name}", loc="left", color=accent, fontsize=8.6, weight="bold", pad=4
        )
        axes[index].text(
            0.99, 1.015, dimension, transform=axes[index].transAxes,
            ha="right", va="bottom", fontsize=5.5, color="#777C85",
        )

    heat_axes = [fig.add_subplot(grid[row, 3]) for row in range(2)]
    talas_error = np.clip(RNG.gamma(2.1, 0.08, (72, 72)), 0, 0.45)
    gate_error = np.clip(RNG.gamma(1.7, 0.035, (72, 72)), 0, 0.45)
    for ax, matrix, title in zip(heat_axes, [talas_error, gate_error], ["TALAS", "GATE-KD"]):
        image = ax.imshow(matrix, cmap="magma_r", vmin=0, vmax=0.34, interpolation="nearest")
        for boundary in range(12, 72, 12):
            ax.axhline(boundary - 0.5, color="white", lw=0.35, alpha=0.8)
            ax.axvline(boundary - 0.5, color="white", lw=0.35, alpha=0.8)
        ax.set_title(title, fontsize=6.5, weight="bold", pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
    heat_axes[0].text(
        0.5, 1.22, "(d)  Geometry residual", transform=heat_axes[0].transAxes,
        ha="center", fontsize=7.0, weight="bold",
    )
    cax = fig.add_subplot(grid[:, 4])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label(r"$|\Delta\,\mathrm{cos}|$", fontsize=5.6, labelpad=2)
    colorbar.ax.tick_params(labelsize=5.0, length=1.8, width=0.45)
    colorbar.outline.set_linewidth(0.45)
    legend(fig, x=0.43, y=0.005)
    fig.subplots_adjust(left=0.012, right=0.985, top=0.92, bottom=0.14)
    save(fig, "figure_1_preview")


def overlay_panel(ax, student, name, accent, description):
    clean_axis(ax)
    ax.scatter(TEACHER_XY[:, 0], TEACHER_XY[:, 1], s=6, facecolor="none", edgecolor="#BBC0C8", lw=0.35, alpha=0.5, zorder=1)
    thread_idx = np.arange(0, len(LABELS), 13)
    for index in thread_idx:
        ax.plot(
            [TEACHER_XY[index, 0], student[index, 0]], [TEACHER_XY[index, 1], student[index, 1]],
            color=accent, lw=0.45, alpha=0.24, zorder=2,
        )
    for label in range(6):
        mask = LABELS == label
        ax.scatter(student[mask, 0], student[mask, 1], s=10, color=COLORS[label], alpha=0.78, lw=0, zorder=3)
    means, edges = mst_edges(student)
    for left, right in edges:
        ax.plot(means[[left, right], 0], means[[left, right], 1], color=accent, lw=1.5, alpha=0.8, zorder=4)
    for label in range(6):
        ax.scatter(*means[label], s=42, color=COLORS[label], edgecolor="white", lw=1.1, zorder=5)
    ax.set_title(name, loc="left", fontsize=10, color=accent, weight="bold", pad=7)
    ax.text(0.0, 1.01, description, transform=ax.transAxes, fontsize=6.2, color=MUTED)


def preview_b():
    """Direct overlay against a shared ghost teacher reference."""
    fig, axes = plt.subplots(1, 2, figsize=(7.45, 3.0), facecolor="white", gridspec_kw={"wspace": 0.07})
    overlay_panel(axes[0], TALAS_XY, "TALAS", TALAS, "student points over the ghosted teacher map")
    overlay_panel(axes[1], GATE_XY, "GATE-KD", GATE, "shorter correspondence threads indicate recovery")
    axes[0].text(0.02, 0.97, "geometry drifts", transform=axes[0].transAxes, va="top", fontsize=6.4, color=TALAS, weight="bold", bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "none", "alpha": 0.88})
    axes[1].text(0.02, 0.97, "geometry reconnects", transform=axes[1].transAxes, va="top", fontsize=6.4, color=GATE, weight="bold", bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "none", "alpha": 0.88})
    fig.suptitle("One teacher map. Two ways to distill it.", x=0.025, y=0.985, ha="left", fontsize=11, weight="bold", color=INK)
    fig.text(0.025, 0.925, "Gray rings are teacher sentences; colored points are the corresponding student sentences.", fontsize=6.4, color=MUTED)
    method_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#AEB4BD", markersize=4.5, label="teacher reference"),
        Line2D([0], [0], color=GATE, lw=0.8, alpha=0.5, label="same-sentence correspondence"),
    ]
    fig.legend(handles=method_handles, loc="lower left", bbox_to_anchor=(0.02, 0.015), ncol=2, frameon=False, fontsize=6.1)
    watermark(fig)
    fig.subplots_adjust(left=0.025, right=0.985, top=0.80, bottom=0.11)
    save(fig, "figure_1_preview_b_teacher_overlay")


def nearest(points, anchor, k=12):
    distance = np.linalg.norm(points - points[anchor], axis=1)
    return np.argsort(distance)[1 : k + 1]


def local_lens(ax, points, anchor, teacher_neighbors, title, accent):
    own_neighbors = nearest(points, anchor, len(teacher_neighbors))
    union = np.unique(np.concatenate([[anchor], teacher_neighbors, own_neighbors]))
    shifted = points[union] - points[anchor]
    radius = max(np.quantile(np.linalg.norm(shifted, axis=1), 0.9), 0.4)
    ax.set_facecolor(PANEL)
    for index in own_neighbors:
        preserved = index in teacher_neighbors
        ax.plot(
            [0, points[index, 0] - points[anchor, 0]], [0, points[index, 1] - points[anchor, 1]],
            color=GATE if preserved else "#D65A50", lw=1.15 if preserved else 0.8,
            ls="-" if preserved else "--", alpha=0.76, zorder=1,
        )
    for index, point in zip(union, shifted):
        ax.scatter(*point, s=28 if index != anchor else 90, color=COLORS[LABELS[index]], edgecolor="white", lw=0.8, zorder=3)
    ax.scatter(0, 0, s=34, marker="*", color="white", edgecolor=INK, lw=0.55, zorder=4)
    ax.set_xlim(-1.15 * radius, 1.15 * radius); ax.set_ylim(-1.15 * radius, 1.15 * radius)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_title(title, loc="left", fontsize=8.2, color=accent, weight="bold", pad=4)


def preview_c():
    """A global semantic map with a magnified local-neighborhood comparison."""
    anchor_candidates = np.flatnonzero(LABELS == 1)
    joy_center = TEACHER_XY[anchor_candidates].mean(axis=0)
    anchor = anchor_candidates[np.argmin(np.linalg.norm(TEACHER_XY[anchor_candidates] - joy_center, axis=1))]
    teacher_neighbors = nearest(TEACHER_XY, anchor, 12)

    fig = plt.figure(figsize=(7.45, 3.05), facecolor="white")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.62, 1], hspace=0.18, wspace=0.16)
    global_ax = fig.add_subplot(grid[:, 0])
    draw_cloud(global_ax, TEACHER_XY, TEACHER, label_centers=True)
    global_ax.scatter(*TEACHER_XY[anchor], s=135, marker="*", color="white", edgecolor=INK, lw=0.8, zorder=7)
    radius = np.quantile(np.linalg.norm(TEACHER_XY[teacher_neighbors] - TEACHER_XY[anchor], axis=1), 0.9)
    lens = Circle(TEACHER_XY[anchor], radius * 1.25, facecolor="none", edgecolor=TEACHER, lw=1.2, ls=(0, (3, 2)), zorder=6)
    global_ax.add_patch(lens)
    global_ax.set_title("Teacher semantic universe", loc="left", color=TEACHER, fontsize=9.3, weight="bold", pad=5)
    global_ax.text(0.01, 0.98, "zoom into one held-out anchor", transform=global_ax.transAxes, va="top", fontsize=6.0, color=MUTED)

    talas_ax = fig.add_subplot(grid[0, 1])
    gate_ax = fig.add_subplot(grid[1, 1])
    local_lens(talas_ax, TALAS_XY, anchor, teacher_neighbors, "TALAS neighborhood", TALAS)
    local_lens(gate_ax, GATE_XY, anchor, teacher_neighbors, "GATE-KD neighborhood", GATE)
    connection = ConnectionPatch(
        xyA=(TEACHER_XY[anchor, 0] + radius, TEACHER_XY[anchor, 1]), coordsA=global_ax.transData,
        xyB=(-0.04, 0.5), coordsB=talas_ax.transAxes, arrowstyle="-|>", mutation_scale=8,
        color="#9AA0A8", lw=0.8,
    )
    fig.add_artist(connection)
    fig.suptitle("Does the student's local neighborhood tell the same story?", x=0.025, y=0.988, ha="left", fontsize=10.6, weight="bold", color=INK)
    fig.text(0.025, 0.928, "Solid purple: teacher neighbor retained · dashed red: neighbor replaced · star: anchor sentence", fontsize=6.2, color=MUTED)
    legend(fig, x=0.31, y=0.012)
    watermark(fig)
    fig.subplots_adjust(left=0.025, right=0.985, top=0.81, bottom=0.12)
    save(fig, "figure_1_preview_c_neighborhood_lens")


def contact_sheet():
    paths = [
        OUTPUT_DIR / "figure_1_preview_a_constellations.png",
        OUTPUT_DIR / "figure_1_preview_b_teacher_overlay.png",
        OUTPUT_DIR / "figure_1_preview_c_neighborhood_lens.png",
    ]
    images = [plt.imread(path) for path in paths]
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 9.25), facecolor="#ECEDEB")
    for label, ax, image in zip(["A · constellations + residual", "B · teacher overlay", "C · neighborhood lens"], axes, images):
        ax.imshow(image)
        ax.set_title(label, loc="left", fontsize=9, weight="bold", color=INK, pad=4)
        ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.985, bottom=0.015, hspace=0.08)
    save(fig, "figure_1_preview_contact_sheet")


if __name__ == "__main__":
    style()
    preview_a()
