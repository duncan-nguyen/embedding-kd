# Table conventions

Each file contains one complete `table` float and preserves the provenance of
its values in leading comments.

Register tables in `main_tables.tex` or `appendix_tables.tex`; `main.tex`
loads both manifests automatically.

- Use `[t]`, with `\caption` followed immediately by `\label`.
- Use `\tablebody` for ordinary tables and `\widetablebody` for dense results.
- Use `booktabs`; do not add vertical rules or manual vertical spacing.
- Report uncertainty with `\meanstd{mean}{std}`. In dense task grids, use
  `\meanstdstacked{mean}{std}` to preserve a readable font size.
- Use `\tablebest{...}` and `\tablesecond{...}` consistently with the caption.
- Prefer short headers and put units or metric definitions in the caption.
- Use `\resizebox{\linewidth}{!}{...}` only when a wide task-level table cannot
  remain legible with natural spacing.

## Arm vocabulary

- Coordinate treatments: `As reduced`, `Haar-random`, and
  `Student-conditioned`.
- Selection signals: `Shuffled correspondence`, `Unrelated student`, and
  `Matched student`.
- Selection schedules: `Per-batch selection`, `Initial selection only`,
  `One refresh`, and `Epoch-wise selection`.
- Endpoint targets: `No endpoint`, `PCA-reduced`, and `Student-conditioned`.
- Structural objectives: `None`, `Gram matching`, `$H_0$ persistence`, and
  `kNN-distance matching`.
- Use `Student (pre-KD)` for the pre-distillation student baseline.
