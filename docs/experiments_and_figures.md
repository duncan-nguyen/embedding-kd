# Experiments and figures — minimal protocol for the ICLR 2027 submission

Companion to `docs/paper_outline.md`. The paper is *geometric analysis → method → state of the art*. Four experiments, three main-text figures, three main-text tables. Every item answers one question raised by the analysis; expectations are marked **E**.

---

## 0. Prerequisites

**Probe set and per-run dump.** Fixed probe set (seeded): 4,096 corpus sentences + every evaluation sentence of the nine tasks. Every run saves, at step 0 and at the end of every epoch, the final-layer pooled unnormalised student embeddings on the probe set and a `train_summary.json` (final endpoint loss, final InfoNCE loss, interface metadata: retained energy, target angular distortion, participation ratio, arm). Cache once per teacher: the teacher's probe embeddings and every target variant. Everything in Experiment 2 and every figure is then post-hoc.

**Corpus.** Deduplicated against every evaluation split (exact match after normalisation), with a manifest. Every reported run uses it.

**Protocol.** One set of hyperparameters (AdamW, lr, batch, epochs, λ) for every arm and baseline; three seeds; mean ± std; paired test across tasks for headline comparisons. Null bands from ≥3 Haar draws wherever a random subspace or rotation appears.

---

## 1. Measurements the analysis relies on (CPU, no training) → Figure 1, Table A1

| What | How | Question it answers |
|---|---|---|
| Teacher spectrum | singular values of the centred teacher on the probe set | is the teacher redundant enough for a rank-$d_S$ image? (Prop. 1) |
| Energy vs angular distortion | PCA-k, random-k (≥3 draws), k ∈ {64,…,$d_T$}: retained energy, Gram distance to the full teacher, mean \|Δcos\| | does a random subspace keep angles while discarding energy? (Prop. 2) |
| Target ceiling | each target variant evaluated directly on STS, no student | how early does the coordinate-level score saturate? (Measurement 3.1) |
| Student-init geometry | per student at step 0: mean pairwise cosine, effective rank, participation ratio of the cross-covariance with PCA targets, cosine reached by the least-squares teacher-side map | is the shortcut of Prop. 4 open at initialisation? which pair should show a gauge effect? |

---

## 2. Experiment 1 — the interface decides (training) → Table 2, Figure 2

One arm per interface family of the taxonomy. Same objective, corpus, initialisation and hyperparameters; only the interface changes. Smallest pair first (go/no-go), then all three pairs (Table A4).

| Arm | Family | Code | **E** |
|---|---|---|---|
| `pca__procrustes` | fixed principal subspace (ours) | available | best or tied |
| `random__none` | fixed random subspace (≥3 draws) | available | competitive, below ours by a margin that tracks angular distortion |
| `learned_t2s` | learned teacher→student (lr ×1 and ×5) | available | lowest training loss, lowest score, collapsed effective rank |
| `learned_s2t` | learned student→teacher (lr ×1 and ×5) | available | close to ours, not above |
| `procrustes_per_batch` | per-step orthogonal re-alignment | ~40 lines | ≈ ours, no dissociation |

Logged per arm: final endpoint loss, effective rank of the student's final embeddings, target angular distortion, downstream average — the columns of Table 2 and the axes of Figure 2.

**Go/no-go (end of week 1, smallest pair, one seed):** *go* if ours beats `learned_t2s` and `procrustes_per_batch` by a clear margin with `learned_t2s` at lower training loss; *no-go* if every arm is within noise of every other — then the analysis has no contrast to explain.

---

## 3. Experiment 2 — recipe ablation, one design choice at a time (training) → Table 3

Smallest pair, interface fixed to ours unless the row changes it. One row per line of the §4 design table.

| Row | Flag | Design choice tested | **E** |
|---|---|---|---|
| cosine → MSE | `--endpoint_loss mse` | cosine, not MSE (sbert recipe) | below ours |
| no InfoNCE | `--lambda_ctr 0` | InfoNCE as anti-collapse term | lower effective rank and score |
| + Gram term | `rkd`-style pairwise term added | no relational term (Prop. 3) | within noise |
| no gauge | `--no-gauge_align` | one Procrustes rotation | within noise on shallow students |
| gauge refit every epoch | `--gauge_refit_every 1` | rotation fixed once, never refit | within noise |
| no teacher | `--lambda_end 0` | floor (SimCSE in the same harness) | floor |

Sanity variants for Table A5 (appendix): non-orthonormal Gaussian vs orthonormal random subspace at the same seed (`random_gaussian` vs `random`); PCA with teacher mean removed at apply time (`pca_full`); uncentred SVD (`svd`); contrastive view dropout vs paired sentence. **E:** all within noise — Prop. 2 concerns the subspace, not the basis; the mean direction is one dimension of $d_S$.

---

## 4. Experiment 3 — what the student inherits (post-hoc, no new runs) → Figure 3, Table A2

From the saved probe embeddings of Experiments 1–2, for ours, `learned_t2s`, `learned_s2t`, no-teacher:
- pairwise-cosine agreement with the teacher (Spearman over probe pairs; 2-D density);
- k-NN overlap with the teacher at k ∈ {1,10,50}, at probe sizes N ∈ {1k, 4k, 16k} (pre-empts the known mutual-kNN scale artefact);
- residual spectrum of $Z-\mathcal{T}$ and its effective rank.

Two reference lines: the PCA target vs the full teacher (ceiling) and an independently trained encoder of the same size vs the teacher (what a good student shares by default).

**E:** ours near the ceiling and far above the independent encoder; `learned_t2s` higher on cosine-to-target yet lower on every teacher-agreement metric, with a high-rank residual (Observation 3.2); the + Gram row of Experiment 2 indistinguishable from ours (Prop. 3).

---

## 5. Experiment 4 — state of the art (training) → Table 1

Ours vs Stella, CDM, DSKD, EMO, TALAS on the three pairs, nine tasks, three seeds, shared hyperparameters. Appendix version adds per-task ± std and two matched-protocol rows: TALAS with AdamW, and ours on TALAS's small-corpus setting.

---

## 6. Gauge (training; appendix) → Figure A1, Table A3

`pca__none` vs `pca__procrustes` vs `pca__random_rot` (≥3 draws = null band) on all three pairs; interpolation $Q(\theta)=R\exp(\theta\log(R^\top Q_{rand}))$, θ ∈ {0, ½, 1}, on the smallest pair and on BERT-base. **E:** monotone where the participation ratio (Table A1) is high, flat where it is near one.

---

## 7. Figures

**Main text**

**Figure 1 — What a rank-$d_S$ image of the teacher keeps.** (a) singular-value decay with $d_S$ markers; (b) retained energy vs angular distortion for PCA-k and random-k; (c) STS ceiling vs k. *Twist:* **E:** a random subspace holding a small fraction of the energy already has a ceiling close to the teacher's — angles, not energy, are what matter.

**Figure 2 — Dissociation.** x = final endpoint loss, y = downstream average, one marker per arm of Experiments 1–2 (seed bars), marker size = effective rank. *Twist:* **E:** the learned teacher-side arm is bottom-left — best loss, worst score, smallest rank.

**Figure 3 — What the student inherits.** Row (a): 2-D density of (teacher cosine, student cosine); row (b): residual spectrum. Columns: ours, `learned_t2s`, `learned_s2t`, `simcse_only`; ceiling and independent-encoder references as overlays. *Twist:* **E:** the arm closest to its targets pointwise is the one furthest from the teacher pairwise.

**Appendix**

**Figure A1 — Gauge.** Downstream vs θ per pair, null band, participation ratio annotated.

**Figure A2 — 2-D embeddings (illustrative).** UMAP fitted on the teacher and applied by transform to the PCA target, ours and `learned_t2s`; colour by class on a labelled subset. Caption states it is illustrative; Figure 3 is the evidence.

Rules: same probe subset in every panel; reduction fitted once on the teacher; seed bands on every agreement metric.

---

## 8. Tables

**Main text**
- **Table 1** — main results: 3 pairs × 9 tasks, baselines + ours, mean over 3 seeds, per-pair average.
- **Table 2** — Experiment 1 on the smallest pair: interface family · target angular distortion · final endpoint loss · effective rank · downstream average (mean ± std).
- **Table 3** — Experiment 2: one row per §4 design choice · final endpoint loss · effective rank · downstream average (mean ± std).

**Appendix**
- **Table 1 (appendix version)** — per-task ± std; matched-protocol rows.
- **Table A1** — student-init geometry per student (feeds Prop. 4 and the gauge prediction).
- **Table A2** — inheritance metrics per arm (numbers behind Figure 3).
- **Table A3** — gauge: null band, interpolation points, participation ratio.
- **Table A4** — Experiment 1 on the other two pairs, same columns as Table 2.
- **Table A5** — sanity variants of Experiment 2.

---

## 9. Order of execution

1. Week 1 — §1 measurements (CPU); Experiments 1 and 2 on the smallest pair, one seed → go/no-go.
2. Week 2 — three seeds: Experiment 1 on all pairs, Experiment 2 on the smallest; Experiment 4 baselines at shared hyperparameters; matched-protocol rows.
3. Week 3 — Experiment 3 and all figures from the dumps; gauge appendix; Table A5.
4. Week 4 — writing; seed top-ups only.
