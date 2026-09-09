#!/usr/bin/env python3
"""Render the 15K one-factor-at-a-time sensitivity figure.

The values mirror docs/latex_iclr/tables/sensitivity_15k.tex. Curves show the
absolute IOD, OOD, and overall scores. Error bars are the run-wise sample
standard deviations reported in the table.
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


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "latex_iclr" / "figures"

METRICS = {
    "Avg. IOD": {"color": "#2563A6", "marker": "o"},
    "Avg. OOD": {"color": "#D97706", "marker": "s"},
    "Avg. All": {"color": "#222222", "marker": "D"},
}

PANELS = (
    {
        "title": r"(a) Topology weight $\lambda_{\mathrm{topo}}$",
        "labels": ("0", "0.5", "0.75", "1"),
        "means": {
            "Avg. IOD": (67.54, 68.62, 68.47, 68.14),
            "Avg. OOD": (77.54, 77.87, 77.84, 77.75),
            "Avg. All": (74.21, 74.79, 74.72, 74.55),
        },
        "stds": {
            "Avg. IOD": (0.18, 0.21, 0.13, 0.06),
            "Avg. OOD": (0.07, 0.02, 0.04, 0.03),
            "Avg. All": (0.03, 0.06, 0.04, 0.03),
        },
    },
    {
        "title": r"(b) Training / $H_0$ batch",
        "labels": ("16", "64", "128", "256"),
        "means": {
            "Avg. IOD": (70.28, 69.65, 68.62, 67.38),
            "Avg. OOD": (78.42, 78.10, 77.87, 77.32),
            "Avg. All": (75.71, 75.29, 74.79, 74.01),
        },
        "stds": {
            "Avg. IOD": (0.09, 0.09, 0.21, 0.03),
            "Avg. OOD": (0.09, 0.08, 0.02, 0.07),
            "Avg. All": (0.03, 0.05, 0.06, 0.05),
        },
    },
    {
        "title": "(c) Gauge calibration set",
        "labels": ("2,048", "4,096", "8,192", "14,760"),
        "means": {
            "Avg. IOD": (68.36, 68.52, 68.26, 68.62),
            "Avg. OOD": (77.59, 77.74, 77.78, 77.87),
            "Avg. All": (74.51, 74.67, 74.61, 74.79),
        },
        "stds": {
            "Avg. IOD": (0.11, 0.08, 0.08, 0.21),
            "Avg. OOD": (0.04, 0.05, 0.03, 0.02),
            "Avg. All": (0.06, 0.02, 0.02, 0.06),
        },
    },
)


def render() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.5,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), sharey=True)
    for ax, panel in zip(axes, PANELS, strict=True):
        x = np.arange(len(panel["labels"]))
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.6, zorder=0)

        for metric, style in METRICS.items():
            means = np.asarray(panel["means"][metric], dtype=float)
            stds = np.asarray(panel["stds"][metric], dtype=float)
            ax.errorbar(
                x,
                means,
                yerr=stds,
                color=style["color"],
                marker=style["marker"],
                markersize=3.8,
                markerfacecolor="white",
                markeredgewidth=1.0,
                capsize=2.0,
                elinewidth=0.8,
                label=metric,
                zorder=3,
            )

        ax.set_title(panel["title"], fontweight="bold", pad=6)
        ax.set_xticks(x, panel["labels"])
        ax.set_xlim(-0.35, len(x) - 0.65)
        ax.set_ylim(66.5, 79.4)
        ax.set_yticks(np.arange(68, 80, 2))
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Score (points)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.8,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.19, top=0.78, wspace=0.16)

    for suffix in ("pdf", "png"):
        fig.savefig(
            OUT_DIR / f"sensitivity_15k.{suffix}",
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(fig)


if __name__ == "__main__":
    render()
