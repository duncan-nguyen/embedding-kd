"""Plot the per-depth diagnostics written by a GeoODE-KD (or TALAS) training run.

Reads `depth_metrics.jsonl` and `step_metrics.jsonl` from one or more run
directories and writes PNGs plus a tidy CSV next to them. The figures are organised
around the three claims in Section 3.12 of the paper rather than around whatever
happens to be logged:

1. the teacher discrepancy should fall smoothly across depth,
2. the relational geometry gap should contract across depth,
3. the student should actually follow the prescribed field, not merely end up near
   the teacher.

With more than one run directory the final epoch of each is overlaid, which is the
GeoODE-vs-baseline comparison figure.

Usage:
    python3 scripts/plot_depth_diagnostics.py runs/<stamp>/geoode
    python3 scripts/plot_depth_diagnostics.py runs/<stamp>/geoode runs/<stamp>/talas
    python3 scripts/plot_depth_diagnostics.py runs/<stamp> --out runs/<stamp>/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Matches the style the run notebook uses for its comparison plots.
plt.style.use("seaborn-v0_8-whitegrid")

# One value per layer (or per transition between layers).
CURVES = {
    "cos_teacher": ("Teacher cosine by depth", "cos(z, tau)", "layer"),
    "gram_gap": ("Relational geometry gap by depth", "||ZZ' - TT'||_F^2 / B^2", "layer"),
    "energy": ("Teacher-conditioned energy by depth", "E(Z, T)", "layer"),
    "dyn_residual": ("ODE consistency residual by transition", "D_cos", "transition"),
    "direction_alignment": (
        "Alignment of the layer update with the field",
        "cos(dz, dt*F)",
        "transition",
    ),
}
SUMMARY_COLUMNS = (
    "cos_first",
    "cos_final",
    "cos_gain",
    "cos_curvature",
    "cos_violations",
    "gram_gap_first",
    "gram_gap_final",
    "gram_gap_contraction",
    "gram_violations",
    "energy_first",
    "energy_final",
    "energy_violations",
    "mean_dyn_residual",
    "mean_alignment",
    "mean_field_norm",
    "mean_step_norm",
    "student_anisotropy",
    "teacher_anisotropy",
)


def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return pd.DataFrame(rows)


def find_runs(paths: list[Path]) -> dict[str, Path]:
    """Accept run directories directly, or a parent holding one per method."""
    runs: dict[str, Path] = {}
    for path in paths:
        if (path / "depth_metrics.jsonl").is_file():
            runs[path.name] = path
            continue
        for child in sorted(path.iterdir()) if path.is_dir() else []:
            if (child / "depth_metrics.jsonl").is_file():
                runs[child.name] = child
    return runs


def long_form(depth: pd.DataFrame, run: str) -> pd.DataFrame:
    """Explode the per-layer lists into one row per (epoch, layer, curve)."""
    frames = []
    for column, (_, _, axis) in CURVES.items():
        if column not in depth:
            continue
        exploded = depth[["epoch", "global_step", column]].explode(column)
        exploded["position"] = exploded.groupby(level=0).cumcount() + 1
        exploded = exploded.rename(columns={column: "value"})
        exploded["curve"] = column
        exploded["axis"] = axis
        exploded["run"] = run
        exploded["value"] = exploded["value"].astype(float)
        frames.append(exploded)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_per_epoch(tidy: pd.DataFrame, run: str, out_dir: Path) -> list[Path]:
    """One figure per curve: depth on x, one line per epoch."""
    written = []
    subset = tidy[tidy["run"] == run]
    epochs = sorted(subset["epoch"].unique())
    colours = plt.cm.viridis([i / max(1, len(epochs) - 1) for i in range(len(epochs))])

    for column, (title, ylabel, axis) in CURVES.items():
        curve = subset[subset["curve"] == column]
        if curve.empty:
            continue
        figure, axes = plt.subplots(figsize=(7.5, 4.2))
        for colour, epoch in zip(colours, epochs):
            means = (
                curve[curve["epoch"] == epoch]
                .groupby("position")["value"]
                .mean()
                .sort_index()
            )
            axes.plot(
                means.index,
                means.to_numpy(),
                marker="o",
                markersize=3.5,
                color=colour,
                linewidth=2.0 if epoch == epochs[-1] else 1.2,
                label=f"epoch {epoch}",
            )
        axes.set(title=f"{title} - {run}", xlabel=axis, ylabel=ylabel)
        axes.legend(frameon=False, ncol=2, fontsize=8)
        figure.tight_layout()
        path = out_dir / f"{run}_depth_{column}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written


def plot_step_profile(depth: pd.DataFrame, run: str, out_dir: Path) -> Path | None:
    """Prescribed step size next to the student's own motion.

    A small ODE residual is only meaningful if the prescribed step is not negligible
    against the layer's native movement, so the two magnitudes belong on one axis.
    """
    if "field_norm" not in depth or depth.empty:
        return None
    last = depth[depth["epoch"] == depth["epoch"].max()]
    field = pd.DataFrame(last["field_norm"].tolist()).mean()
    step = pd.DataFrame(last["step_norm"].tolist()).mean()

    figure, axes = plt.subplots(figsize=(7.5, 4.2))
    positions = range(1, len(field) + 1)
    axes.plot(positions, step.to_numpy(), marker="o", label="student step |dz|")
    axes.plot(
        positions, field.to_numpy(), marker="s", label="prescribed step |dt*F|"
    )
    axes.set(
        title=f"Prescribed vs realized step size - {run} (final epoch)",
        xlabel="transition",
        ylabel="mean row norm",
        yscale="log",
    )
    axes.legend(frameon=False)
    figure.tight_layout()
    path = out_dir / f"{run}_depth_step_norms.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_training_progress(depth: pd.DataFrame, run: str, out_dir: Path) -> Path | None:
    """Endpoint quality and depth behaviour as training proceeds."""
    if depth.empty:
        return None
    tracked = [
        ("cos_final", "cos(z_L, tau)"),
        ("gram_gap_final", "relational gap at L"),
        ("mean_alignment", "mean field alignment"),
        ("student_anisotropy", "student anisotropy"),
    ]
    available = [(column, label) for column, label in tracked if column in depth]
    figure, axes = plt.subplots(
        len(available), 1, figsize=(7.5, 2.1 * len(available)), sharex=True
    )
    axes = axes if len(available) > 1 else [axes]
    for axis, (column, label) in zip(axes, available):
        axis.plot(depth["global_step"], depth[column], linewidth=1.2)
        axis.set_ylabel(label, fontsize=8)
    axes[0].set_title(f"Depth diagnostics over training - {run}")
    axes[-1].set_xlabel("global step")
    figure.tight_layout()
    path = out_dir / f"{run}_depth_progress.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_losses(run_dir: Path, run: str, out_dir: Path) -> Path | None:
    steps = read_jsonl(run_dir / "step_metrics.jsonl")
    columns = [c for c in ("loss_end", "loss_dyn", "loss_ctr") if c in steps]
    if steps.empty or not columns:
        return None
    window = max(1, len(steps) // 100)
    figure, axes = plt.subplots(figsize=(7.5, 4.2))
    for column in columns:
        axes.plot(
            steps["global_step"],
            steps[column].rolling(window, min_periods=1).mean(),
            label=column,
            linewidth=1.3,
        )
    axes.set(
        title=f"Loss components - {run}",
        xlabel="global step",
        ylabel=f"value (rolling mean, window {window})",
    )
    axes.legend(frameon=False)
    figure.tight_layout()
    path = out_dir / f"{run}_loss_components.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_comparison(tidy: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Final-epoch curves of every run on shared axes."""
    written = []
    final = tidy[tidy["epoch"] == tidy.groupby("run")["epoch"].transform("max")]
    for column, (title, ylabel, axis) in CURVES.items():
        curve = final[final["curve"] == column]
        if curve["run"].nunique() < 2:
            continue
        figure, axes = plt.subplots(figsize=(7.5, 4.2))
        for run, frame in curve.groupby("run"):
            means = frame.groupby("position")["value"].mean().sort_index()
            axes.plot(means.index, means.to_numpy(), marker="o", label=run)
        axes.set(title=f"{title} (final epoch)", xlabel=axis, ylabel=ylabel)
        axes.legend(frameon=False)
        figure.tight_layout()
        path = out_dir / f"comparison_depth_{column}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written


def summarise(depth: pd.DataFrame, run: str) -> pd.DataFrame:
    columns = [c for c in SUMMARY_COLUMNS if c in depth]
    table = depth.groupby("epoch")[columns].mean().reset_index()
    table.insert(0, "run", run)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="Run directory (or a parent containing one directory per method)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory for the figures (default: alongside the first run)",
    )
    args = parser.parse_args()

    runs = find_runs(args.run_dirs)
    if not runs:
        raise SystemExit(
            "No depth_metrics.jsonl found. Train with --depth_log_every > 0 "
            "(default 50 for the talas and geoode methods)."
        )

    out_dir = args.out or (next(iter(runs.values())).parent / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    tidy_frames, summaries, written = [], [], []
    for run, run_dir in runs.items():
        depth = read_jsonl(run_dir / "depth_metrics.jsonl")
        if depth.empty:
            continue
        tidy = long_form(depth, run)
        tidy_frames.append(tidy)
        summaries.append(summarise(depth, run))
        written.extend(plot_per_epoch(tidy, run, out_dir))
        written.extend(
            path
            for path in (
                plot_step_profile(depth, run, out_dir),
                plot_training_progress(depth, run, out_dir),
                plot_losses(run_dir, run, out_dir),
            )
            if path is not None
        )

    if tidy_frames:
        tidy_all = pd.concat(tidy_frames, ignore_index=True)
        written.extend(plot_comparison(tidy_all, out_dir))
        tidy_all.to_csv(out_dir / "depth_curves.csv", index=False)

    if summaries:
        summary = pd.concat(summaries, ignore_index=True)
        summary_path = out_dir / "depth_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"\nWrote {summary_path}")

    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
