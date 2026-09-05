"""Render synthetic previews of the three all-pairs layerwise figures.

The values are deliberately synthetic and the exported previews carry a visible
watermark.  The experiment notebook calls the same renderer without that mark.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
from notebooks._layerwise_analysis import render_all_options


OUT_DIR = PROJECT_DIR / "docs/latex_iclr/figures/layerwise_options"
PAIR_SPECS = [
    ("qwen3_4b_to_bert_base", "Qwen3-4B → BERT-base", 37, 13),
    ("bge_m3_to_minilm_h768", "BGE-M3 → MiniLM-768", 25, 7),
    ("qwen3_0.6b_to_minilm_h384", "Qwen3-0.6B → MiniLM-384", 29, 7),
]


def gaussian(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cx: float,
    cy: float,
    sx: float,
    sy: float,
) -> np.ndarray:
    return np.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))


def illustrative_pair(
    teacher_layers: int,
    student_layers: int,
    *,
    seed: int,
) -> dict[str, object]:
    """Create one deterministic, plausible-looking synthetic pair."""
    rng = np.random.default_rng(seed)
    student = np.linspace(0.0, 1.0, student_layers)[None, :]
    teacher = np.linspace(0.0, 1.0, teacher_layers)[:, None]
    x = np.broadcast_to(student, (teacher_layers, student_layers))
    y = np.broadcast_to(teacher, (teacher_layers, student_layers))

    inherited_band = np.exp(-((y - (0.08 + 0.84 * x)) / 0.18) ** 2)
    init = 0.22 + 0.24 * x + 0.10 * y + 0.13 * inherited_band
    init += rng.normal(0.0, 0.018, size=init.shape)
    init = np.clip(init, 0.0, 1.0)

    ours_gain = 0.025 + 0.075 * x + 0.030 * y
    ours_gain += 0.13 * gaussian(x, y, cx=0.88, cy=0.88, sx=0.25, sy=0.28)
    talas_gain = 0.015 + 0.035 * x + 0.19 * inherited_band
    talas_gain += rng.normal(0.0, 0.012, size=talas_gain.shape)

    ours = np.clip(init + ours_gain, 0.0, 0.98)
    talas = np.clip(init + talas_gain, 0.0, 0.98)

    init_h0 = 0.22 - 0.035 * x - 0.020 * y
    ours_reduction = 0.008 + 0.020 * x + 0.012 * y
    ours_reduction += 0.060 * gaussian(
        x, y, cx=0.88, cy=0.88, sx=0.28, sy=0.30
    )
    talas_reduction = 0.004 + 0.010 * x + 0.070 * inherited_band
    ours_h0 = np.clip(init_h0 - ours_reduction, 0.02, None)
    talas_h0 = np.clip(init_h0 - talas_reduction, 0.02, None)

    ours_proc = np.clip(ours + 0.035, 0.0, 0.98)
    talas_proc = np.clip(talas + 0.025, 0.0, 0.98)
    zero = np.zeros_like(init)

    return {
        "init_cka": init,
        "init_h0": init_h0,
        "ours": {
            "cka_mean": ours,
            "cka_std": zero,
            "delta_cka": ours - init,
            "h0_mean": ours_h0,
            "h0_std": zero,
            "h0_reduction": init_h0 - ours_h0,
            "procrustes_mean": ours_proc,
            "procrustes_std": zero,
        },
        "talas": {
            "cka_mean": talas,
            "cka_std": zero,
            "delta_cka": talas - init,
            "h0_mean": talas_h0,
            "h0_std": zero,
            "h0_reduction": init_h0 - talas_h0,
            "procrustes_mean": talas_proc,
            "procrustes_std": zero,
        },
        "metadata": {"procrustes_rank": 128},
    }


def main() -> None:
    pair_order = [key for key, _, _, _ in PAIR_SPECS]
    pair_labels = {key: label for key, label, _, _ in PAIR_SPECS}
    results = {
        key: illustrative_pair(teacher_layers, student_layers, seed=17 + index)
        for index, (key, _, teacher_layers, student_layers) in enumerate(PAIR_SPECS)
    }
    paths = render_all_options(
        results,
        pair_order=pair_order,
        pair_labels=pair_labels,
        output_dir=OUT_DIR,
        figure_note="ILLUSTRATIVE — SYNTHETIC DATA",
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
