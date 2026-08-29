# Experiments and figures — protocol for the ICLR 2027 submission

Companion to `docs/paper_outline.md`. Every experiment is tied to a claim (C1 interface · C2 structure-for-free · C3 depth · G gauge · M main results) and to the figure or table it feeds. Expectations are marked **E**; nothing here asserts a result.

---

## 0. Prerequisites — do these before any new run

### 0.1 Probe set and per-run dump (non-negotiable)
Every training run, regardless of arm, writes at the end of every epoch (and at step 0):
- `probe_final.pt` — final-layer pooled, **unnormalised** student embeddings on a fixed probe set;
- `probe_layers.pt` — pooled state of every layer on the same probe set;
- `train_summary.json` — final endpoint loss, final InfoNCE loss, learned-map parameters if any (W), gauge/projection metadata (retained energy, participation ratio, arm name).

Probe set (fixed once, seeded): 4,096 corpus sentences + every evaluation sentence (all nine tasks + retrieval queries and a fixed 4,096-document sample per retrieval corpus). Also cache, once per teacher: the full teacher embeddings on the probe set, and every *target* variant (PCA-k, random-k, MRL-prefix-k, with and without R) on the probe set.

Why: every ladder rung, every depth profile, every residual spectrum and every visualisation below is then a post-hoc computation. Changing a metric never triggers retraining.

*Implementation (2026-08-29).* The dump is produced from the per-epoch student weights (`--save_every 1 --weights_dir`) right after each arm finishes, by `notebooks/audit_runs_qwen_minilm.ipynb` (cell 6) — same content, no change to the training loop. Probe set and dedup: `src/probe_set.py`; every metric below: `src/structural_audit.py`; tables and figures: `notebooks/audit_analysis_qwen_minilm.ipynb`.

### 0.2 Corpus deduplication
Remove from the training corpus every sentence that appears in any evaluation split (exact match after whitespace/case normalisation); keep a manifest. All new runs use the deduplicated corpus. Report a clean-subset robustness table for anything trained on the old corpus.

### 0.3 Matched-hyperparameter protocol
One learning rate, batch size, epoch count, optimiser (AdamW) and λ for every arm and every baseline. Two extra arms for TALAS: (a) TALAS with AdamW, (b) ours on TALAS's small-corpus setting. Three seeds for every arm that supports a claim; one seed is enough only for the go/no-go pass.

---

## 1. Experiments by claim

### C1 — The interface (Table 2, Figures 2, 6, 7, 8)

| Arm | What it is | Implementation status | Expectation |
|---|---|---|---|
| `pca__procrustes` | PCA to $d_S$, one Procrustes rotation to student init, frozen | available | best or tied-best |
| `pca__none` | PCA only | available | ≈ ours on shallow students |
| `pca__random_rot` | PCA + Haar rotation (≥3 draws) | available | ≤ `pca__none`; draws give the null band for G |
| `random__none` | Haar orthonormal subspace, same rank (≥3 draws) | available | competitive but below PCA; gap tracks angular distortion |
| `mrl_prefix` | first $d_S$ coordinates of the teacher | available (`--projection_type mrl_prefix`) | between random and PCA for MRL-trained teachers; ≈ random otherwise |
| `learned_t2s` (lr ×1, ×5) | trainable $W\in\mathbb{R}^{d_T\times d_S}$ on the teacher side | available (`--learned_projector_lr_scale`) | lowest training loss, lowest score, collapsed rank |
| `learned_s2t` (lr ×1, ×5) | trainable $W\in\mathbb{R}^{d_S\times d_T}$ lifting the student | available | close to ours, not above |
| `procrustes_per_batch` | orthogonal re-alignment re-solved every step | to write (~40 lines) | ≈ ours or slightly below; no dissociation |
| `pca__mse` | PCA target + MSE (sentence-transformers ≤5.4 recipe) | available (`--endpoint_loss mse --lambda_ctr 0 --no-gauge_align`) | below cosine + InfoNCE |
| `simcse_only` | no teacher | available | floor |

Also sweep $d_S$ for the fixed interfaces on the H384 student by *padding/truncating the target* (k ∈ {64, 128, 256, 384}) to obtain a distortion-vs-score curve (Figure 7).

Per-arm post-hoc quantities (from the dump): target angular distortion (Gram distance to the full teacher), retained energy, final endpoint loss, effective rank of the student's final embeddings (RankMe-style), principal angles between span(W) and the PCA basis per epoch (learned arms), residual spectrum (Observation 3.1).

Decision rule for C1: the fixed interfaces must beat `learned_t2s` and `procrustes_per_batch` by a clear margin with the learned arm at *lower* training loss; the distortion-vs-score relation across {PCA, MRL-prefix, random, PCA-k} must be monotone.

### C2 — Structure for free (Figure 3, Figures 9–11)

Arms (all on top of `pca__procrustes` unless stated): ours · `+gram` (RKD/SP-style pairwise-similarity term at the last layer) · `+lasd` (TALAS-style relational self-distillation between adjacent layers) · `+anchor_k` (TALAS-style cosine anchoring of the upper k layers) · `learned_t2s` · `learned_s2t` · `simcse_only` (floor) · PCA target itself vs teacher (ceiling).

Ladder rungs, computed teacher-vs-student on the probe set:
1. cosine to target (mean, and distribution);
2. Gram distance, linear CKA, Procrustes distance (Kornblith 2019; Ding 2021);
3. k-NN overlap at k ∈ {1, 10, 50} (N2O), and *mutual* k-NN, each at probe sizes N ∈ {1k, 4k, 16k, full corpus sample};
4. H0 persistence barcode distance (MST edge distribution, Wasserstein-1) and RTD (Barannikov 2022) on a 2k subsample, averaged over draws.

Decision rule for C2: `+gram`, `+lasd`, `+anchor_k` within the seed band of ours on score **and on every rung**; `learned_t2s` above ours on rung 1 and below on rungs 2–4.

### C3 — Depth (Figure 4, Figure 12)

Arms on all three students: ours · `freeze_lower` (train only the top two blocks) · `+anchor_k` · `+lasd`.
Post-hoc from `probe_layers.pt`: per-layer linear CKA and Procrustes distance to the *step-0* state of the same layer; per-layer cosine to the target; per-layer intrinsic dimension (TwoNN) over epochs.
Decision rule: `freeze_lower` within noise of ours on shallow students; on BERT-base report whichever way it falls and condition C3 accordingly.

### G — Gauge as an optimisation effect (Figure 5)

- Null band: `pca__random_rot` with ≥3 draws, 3 seeds.
- Interpolation: $Q(\theta)=R\,\exp(\theta\log(R^\top Q_{rand}))$, θ ∈ {0, ¼, ½, ¾, 1}, one seed first, then 3 seeds on the two extreme and middle points.
- Rank-one control: replace $R$ by the Householder reflection mapping the mean target direction onto the mean initial-student direction.
- Per-pair predictor: participation ratio of the cross-covariance between PCA targets and initial student states, logged at target-construction time.
Expectation: curve in θ monotone and clearly non-flat only where the participation ratio is high; the rank-one control recovers the gain where it is near one.

### M — Main results and external validity (Table 1, Table 3, Figure 13)

- Table 1: all baselines + ours, 3 pairs, 3 seeds, matched HP, deduplicated corpus; clean-subset column.
- Matched-TALAS arms (AdamW; small-corpus setting).
- Retrieval trio + MTEB-v2 subset (Table 3).
- Corpus-size sweep (15k → 200k) for ours and the strongest baseline.
- MRL-style truncation curve of the distilled student's embedding (Figure 13).
- Ceiling analysis: evaluate every *target* variant directly on the STS tasks (no student) — the score the student is chasing.

---

## 2. Figures — geometry made visible

Each figure lists: what it shows · how it is computed · the expected pattern · which claim it serves. Aim for one figure per claim in the main text; the rest go to the appendix.

**Figure 1 — Interface map (schematic).** Two spaces of different width, the five interface families as arrows between them, published methods placed on the arrows. *Serves:* §2 taxonomy.

**Figure 2 — Dissociation scatter.** x = final endpoint loss, y = downstream average, one marker per interface arm (error bars over seeds); marker size = effective rank of the student's final embeddings. **E:** fixed interfaces cluster top-right; `learned_t2s` bottom-left with the smallest marker. *Serves:* C1, Prop. 4.

**Figure 3 — Structural ladder.** x = rung (cosine → Gram/CKA → k-NN overlap → H0/RTD), y = teacher-agreement normalised so the PCA target = 1 and SimCSE-only = 0; one line per arm. **E:** ours, `+gram`, `+lasd`, `+anchor_k` overlap; `learned_t2s` starts high and drops; `learned_s2t` tracks ours. *Serves:* C2 (central figure).

**Figure 4 — Depth heatmap.** Rows = layers, columns = epochs, colour = CKA to the step-0 state (one panel per student; ours vs `+anchor_k`). **E:** lower rows stay at 1.0 in every panel; only the top rows change. *Serves:* C3.

**Figure 5 — Gauge interpolation.** x = θ from Procrustes (0) to random rotation (1), y = downstream average and epoch-1 endpoint loss; one curve per pair, annotated with the participation ratio; horizontal band = random-rotation null. **E:** slope grows with participation ratio. *Serves:* G.

**Figure 6 — Teacher spectrum and what each interface keeps.** Left: singular-value decay of the centred teacher with $d_S$ markers; right: retained energy vs angular distortion for PCA-k, random-k, MRL-prefix-k. **E:** random keeps little energy but small angular distortion; PCA keeps both. *Serves:* Props. 1–2, explains "random is competitive".

**Figure 7 — Distortion predicts transfer.** x = angular (Gram) distortion of the target, y = downstream average, points = {PCA-k, random-k, MRL-prefix-k, learned arms}. **E:** monotone for fixed interfaces; learned t→s off the curve (low distortion of the *fitted* map, low score). *Serves:* C1.

**Figure 8 — Pairwise-cosine agreement.** 2-D density of (teacher cosine, student cosine) over random probe pairs, one panel per arm. **E:** a tight diagonal for fixed interfaces; a diagonal for `learned_s2t`; a compressed/flat cloud for `learned_t2s`. The most direct picture of "Gram transfers for free". *Serves:* C2.

**Figure 9 — Residual spectrum.** Singular values of the residual $Z-\mathcal{T}$ against those of $\mathcal{T}$, per arm. **E:** a few dominant residual directions for fixed interfaces (shared offset), a flat high-rank residual for `learned_t2s`. *Serves:* Observation 3.1.

**Figure 10 — 2-D embeddings of the same sentences under teacher / target / student / learned t→s student.** UMAP or t-SNE fitted on the *teacher* and reused (transform) for the others so panels are comparable; colour by class on a labelled subset (e.g. intent classes or emotion labels); overlay the k-NN graph edges that are preserved vs lost. **E:** cluster layout preserved by the PCA target and by our student; merged/collapsed clusters for `learned_t2s`. Use as an illustration only, never as evidence (state this in the caption). *Serves:* C2, intuition.

**Figure 11 — k-NN overlap vs probe size.** x = N ∈ {1k, 4k, 16k, full}, y = k-NN overlap with the teacher, curves per arm, plus the PCA-target ceiling. **E:** overlap decays slowly with N for fixed interfaces and remains ordered as in Figure 3. *Serves:* C2 robustness (pre-empts the scale artefact).

**Figure 12 — Per-layer intrinsic dimension and cosine-to-target over depth.** Two panels: TwoNN ID by layer at step 0 and at the end; cosine-to-target by layer. **E:** both change only in the top blocks. *Serves:* C3 (appendix).

**Figure 13 — Truncation curve of the distilled student.** Score vs retained leading dimensions (MRL-style) for ours vs `learned_s2t` vs SimCSE-only. **E:** graceful degradation; answers LEAF's claim that PCA-based targets destroy truncation robustness. *Serves:* M.

**Figure 14 — Anisotropy histograms.** Distribution of pairwise cosines for teacher, PCA target, ours, `learned_t2s`, SimCSE-only. **E:** ours inherits the teacher's spread; `learned_t2s` concentrates near a single direction. *Serves:* Prop. 4 intuition (appendix).

Visualisation rules: same probe subset in every panel; UMAP/t-SNE fitted once on the teacher and applied by transform; all agreement metrics reported with seed bands; every 2-D plot carries a caption saying it is illustrative.

---

## 3. Tables

- **Table 1** — main results, 3 pairs × 9 tasks + retrieval, baselines + ours, mean ± std over 3 seeds, clean-subset column.
- **Table 2** — interface ablation: arm · retained energy · target angular distortion · final endpoint loss · effective rank · downstream average (mean ± std).
- **Table 3** — retrieval and MTEB-v2 subset.
- **Table 4** — ladder values per arm (numbers behind Figure 3).
- **Table 5** — depth: full vs frozen-lower-stack per student.
- **Table 6** — gauge: null band, interpolation points, rank-one control, participation ratio per pair.
- **Table 7** — matched-HP protocol vs TALAS (AdamW / small corpus).

---

## 4. Statistical protocol

- Three seeds for every reported number; report mean ± std and, for headline comparisons, a paired test across tasks.
- Null bands from ≥3 Haar draws for every random-rotation / random-subspace arm.
- Every claim of "no effect" must state the seed band it is measured against and be shown on *both* the score and the ladder.
- Report the clean 7-task subset alongside the full nine tasks for anything touched by the old corpus.

---

## 5. Run budget (order of execution)

1. Week 1 (smallest pair, one seed): all C1 arms, `+gram`, `+lasd`, `+anchor_k`, `freeze_lower`, `pca__mse`, `simcse_only`, matched-TALAS. → go/no-go.
2. Week 2: three seeds for the eight arms that carry claims on all three pairs; gauge interpolation and rank-one control on the smallest pair, then on BERT-base.
3. Week 3: retrieval, MTEB-v2 subset, corpus sweep, truncation curve; all figures from the dumps.
4. Week 4: writing only; no new runs except seed top-ups.

Compute rule: prefer breadth of arms at one seed for the decision, then depth of seeds only where a claim rests.

---

## 6. Where things run (smallest pair)

| | |
|---|---|
| Run every arm of §1 at matched HP, dedup corpus, probe dump | `notebooks/audit_runs_qwen_minilm.ipynb` |
| Tables 2, 4–7 and Figures 2–14 from the dumps | `notebooks/audit_analysis_qwen_minilm.ipynb` |
| Prerequisites (§0.1–0.2) | `src/probe_set.py` |
| Metrics (ladder, depth, spectra, ceilings, truncation) | `src/structural_audit.py` |
| Main results table (all baselines, Table 1) | `notebooks/main_tables.ipynb` |

Arms still without code (listed in the run notebook's plan as `needs_code`, skipped at run time): `procrustes_per_batch`, PCA-k target truncation (Figure 7), `+gram` / `+lasd` / `+anchor_k`, `freeze_lower`, gauge interpolation $Q(\theta)$ and the rank-one control, TALAS with AdamW.
