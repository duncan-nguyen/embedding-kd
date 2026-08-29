# Paper outline — ICLR 2027

## Headline

**Primary title**
> **Fix the Interface, Not the Student: A Geometric Analysis of Cross-Dimensional Embedding Distillation**

Selling point in one line: *when a 1024–2560-d embedding teacher is distilled into a 384–768-d encoder, the geometry of the connection between the two spaces — not the loss family, not the depth schedule — decides what can transfer; the analysis predicts that a frozen near-isometric image of the teacher is all a student needs, and the method it implies, with no projector, no relational term and no inference-time parameters, sets the state of the art.*

Alternative titles:
- *Random Subspaces Are Almost Enough: The Geometry of Embedding Distillation Targets* — leads with the most surprising predicted result.
- *The Interface Is the Method* — shortest; pair with a descriptive subtitle.
- *What a Small Encoder Can Inherit: Geometry-Guided Embedding Distillation*.

Why this sells at ICLR: (i) it reverses the direction the field is taking in 2026 — LEAF, jina-v5, TALAS and sentence-transformers v5.5 all *add* adaptivity — by analysing the geometry first and letting the analysis dictate the method; (ii) the analysis chains published results (Eckart–Young, Johnson–Lindenstrauss, the Procrustes bound, the projector-shortcut theorem) into one setting and produces testable predictions before any training run; (iii) the method has zero inference-time parameters and one loss weight, yet matches or beats the strongest multi-layer baseline — the "the simple thing is the right thing, and here is why" genre that reviewers reward when the analysis is airtight.

Deadlines: abstract 2026-09-18, full 2026-09-25 (AoE). 9 pages, ≥3 seeds on every claim, mandatory AI-use disclosure.

**Page budget (main text):** 3 tables (Table 1 main results, Table 2 interface families, Table 3 recipe ablation) and 3 figures (Figure 1 what a rank-$d_S$ image keeps, Figure 2 dissociation, Figure 3 geometry inheritance). Appendix: Table A1 student-init geometry, A2 inheritance metrics, A3 gauge, A4 Table 2 on the other two pairs, A5 sanity variants, appendix version of Table 1; Figures A1 gauge, A2 geometry-inheritance visual: Gram heatmaps under the teacher ordering + kNN-retention map on the teacher layout.

---

## 0. The argument in one paragraph (the chain every section must follow)

Distilling a large embedding model into a small encoder of lower width is usually approached by choosing *what to supervise*: which layers, which relations, which loss. We approach it by first asking *how the two spaces are connected*. A teacher vector must be turned into a target that lives in the student's space, and we analyse the geometry of this **endpoint interface**: which invariants of the teacher a rank-$d_S$ image can preserve, what fixes the residual rotational freedom, and what a learnable map does to the objective. The analysis yields three predictions: (1) a **fixed, near-isometric linear image** of the teacher is sufficient — the principal subspace is the least-distorting such image, but even a random orthonormal subspace, which discards most of the teacher's energy yet preserves angles, should already be competitive, so what must be preserved is *angular structure*, not variance; (2) with such an interface, pointwise cosine matching already carries the teacher's pairwise structure, so relational terms are redundant; (3) a learnable teacher-side projector has a degenerate near-minimiser next to the student's initialisation and is therefore a shortcut, whereas a student-side projector is merely unnecessary. These predictions dictate a method — PCA to $d_S$, one Procrustes rotation to the student's initialisation, freeze, cosine + InfoNCE at the last layer — whose every design choice is traceable to an analysis result. The experiments verify the predictions with controlled interface ablations and show the resulting method sets the state of the art.

Each section states: claim → argument and supporting prior work/theory → expected evidence → the figure/table it owns. No results are asserted here; expectations are marked **E**.

---

## 1. Introduction (≈1 page)

**Hook — the field is adding adaptivity without analysing the geometry.** Learned projectors kept at inference (LEAF, ACL 2026); learned student-side maps, with the frozen teacher-side variant reported as "less effective" and the unfrozen one as collapsing, without explanation (jina-embeddings-v5, SIGIR 2026); multi-layer anchoring plus relational self-distillation with sharpness-aware optimisation (TALAS, ACL 2026); a learnable `EmbedDistillLoss(projection_dim)` replacing the decade-old frozen-PCA recipe in sentence-transformers v5.5.0. Each is justified by benchmark deltas; none asks what a rank-$d_S$ image of the teacher can preserve, or what a learnable map does to the objective.

**Reframing.** Teacher and student have different widths. The interface that maps one into the other has its own geometry, and that geometry determines whether any downstream loss can transfer structure at all. We analyse it first and let the method follow.

**Contributions.**
1. **A geometric analysis of the endpoint interface.** We formalise the interface between two embedding spaces of different width, give a taxonomy that places every existing method in one table, and — by instantiating Eckart–Young, Johnson–Lindenstrauss, the Procrustes bound and the projector-shortcut theorem in this setting, together with measurements of real teacher and student geometry — derive three predictions: a fixed near-isometric image suffices and its ordering is governed by angular distortion rather than retained variance (so a random subspace is competitive); relational terms are redundant given such an image; learned teacher-side projectors are shortcuts while student-side ones are unnecessary.
2. **A method dictated by the analysis.** PCA to $d_S$ + one Procrustes rotation to the student's initialisation, frozen; cosine + InfoNCE at the last layer. No projector, no relational term, no intermediate-layer term, no inference-time parameters, one loss weight — each omission is an analysis result, not a tuning decision.
3. **State of the art with the simplest method in the comparison.** **E:** at matched data and optimiser the method matches or exceeds the strongest multi-layer baseline (TALAS) across three teacher→student pairs and nine sentence-level tasks, and the controlled interface ablations confirm the predicted ordering.

**Explicitly not a contribution:** PCA targets per se (sentence-transformers 2020–v5.4; HPD 2022). The contribution is the analysis that explains *when and why* a frozen isometric target is the right interface, and the method and predictions that follow from it.

---

## 2. Problem setting: the endpoint interface (≈0.75 page)

**2.1 Setup.** Frozen teacher $f_T: x\mapsto t\in\mathbb{R}^{d_T}$ (cached once); student $f_S$ with final pooled state $z\in\mathbb{R}^{d_S}$, $d_S<d_T$. An *interface* is either a teacher-side map $\Pi$ giving targets $\tau=\mathrm{norm}(\Pi(t))\in S^{d_S-1}$ or a student-side lift $\psi(z)\in\mathbb{R}^{d_T}$. Training minimises $\lambda_{end}\,\mathbb{E}[1-\langle z,\tau\rangle]+\lambda_{ctr}\,\mathcal{L}_{InfoNCE}$ (or the lifted analogue).

**2.2 What "comparable" must mean.** Downstream use is cosine-metric, so representations are defined only up to the orthogonal group (Kornblith et al. 2019; Williams et al. 2021; Han 2607.03572 on representation equivalence classes). A comparable interface must (i) preserve the teacher's orthogonally-invariant structure — pairwise cosines — as well as a rank-$d_S$ image can, and (ii) fix the residual $O(d_S)$ gauge. This gives the taxonomy:

| Interface family | Angular structure preserved? | Gauge fixed by | Where it appears |
|---|---|---|---|
| Fixed orthonormal, principal subspace | up to tail energy (Prop. 1) | Procrustes to student init (ours) / arbitrary (sbert ≤5.4, HPD teacher side) | **ours**, sbert PCA recipe, HPD |
| Fixed orthonormal, random subspace | up to JL distortion (Prop. 2) | arbitrary | jina-v5 "frozen" arm; TCS random control (vision) |
| Learned teacher→student | no guarantee; shortcut exists (Prop. 4) | co-adapts every step | sbert v5.5 EmbedDistillLoss, EMO, jina-v5 "teacher-side unfrozen" |
| Learned student→teacher | student lifts into teacher space; $\psi$ absorbs residual | co-adapts | TALAS $W_l$, LEAF $W_{out}$, jina-v5 $\psi$ |
| Per-step orthogonal re-alignment | by construction | re-solved per batch | EdgePoint2; Bhattarai's Procrustes loss |

(The taxonomy table is the figure for this section; no separate schematic.)

---

## 3. Geometric analysis (≈2 pages; proofs in App. A) — the core of the paper

Every statement is elementary or a direct instantiation of a published result — say so. The value is in stringing them together for this setting, measuring the quantities they depend on for real teachers and students, and turning them into predictions *before* training.

### 3.1 What a rank-$d_S$ image of the teacher can keep

**Prop. 1 (Eckart–Young for targets).** For centred teacher matrix $\bar T$ and orthonormal $P\in\mathbb{R}^{d_T\times d_S}$, the top-$d_S$ right-singular basis minimises $\|\bar TPP^\top\bar T^\top-\bar T\bar T^\top\|_F$ over rank-$d_S$ orthogonal projections; with a lemma bounding the effect of post-projection renormalisation, the target Gram deviates from the teacher Gram by at most the tail-energy ratio. **E:** for LLM-based embedding teachers the spectrum decays fast enough that the PCA target at $d_S=384$ is a near-isometric image.

**Prop. 2 (a random orthonormal subspace is also an interface).** For Haar-distributed $P$ and unit $x,y$: $\mathbb{E}\langle Px,Py\rangle=(d_S/d_T)\langle x,y\rangle$ and the renormalised cosine concentrates around $\cos(x,y)$ with deviation $O(\sqrt{(1-\cos^2)/d_S})$ (Johnson–Lindenstrauss; Dasgupta–Gupta). A random subspace keeps a $d_S/d_T$ fraction of the energy but preserves *angles* to first order. **E:** a random fixed projection is therefore competitive, and the quantity that predicts transfer is angular distortion of the target, not retained variance.

**Measurement 3.1 (Figure 1).** On each teacher: singular-value decay; retained energy and angular (Gram) distortion of PCA-$k$ vs random-$k$ vs leading-$k$ coordinates for $k$ from 64 to $d_T$; and the *ceiling* score of each target evaluated directly on STS. **E:** energy and angular distortion separate sharply for random subspaces (little energy, small distortion); ceiling scores saturate early for both — which is itself a prediction that coordinate-level (STS) scores will not separate interfaces much, while neighbourhood-sensitive tasks will.

### 3.2 Why pointwise matching is enough — no relational term

**Prop. 3 (gauge equivalence).** (a) On the sphere $1-\langle z,\tau\rangle=\|z-\tau\|^2/2$, hence $|\langle z_i,z_j\rangle-\langle\tau_i,\tau_j\rangle|\le\sqrt{2\ell_i}+\sqrt{2\ell_j}$: every Gram/RKD/similarity-preserving discrepancy is bounded by the endpoint loss. (b) Conversely (Maystre et al. 2510.13406; Schönemann 1966), approximate Gram preservation implies an *orthogonal* map with small pointwise error, and orthogonal maps dominate unconstrained linear ones. Therefore, **for a fixed orthonormal interface, pointwise matching in a fixed gauge and relational matching are the same statement up to the gauge; a relational loss cannot carry information the endpoint loss lacks.** This is the reason the method has no relational term.

**Observation 3.2 (where the bound is loose).** At practical loss values the bound in 3(a) is vacuous, yet **E:** the student's Gram will track the teacher's far more closely than the bound allows — possible only if the residual $Z-\mathcal{T}$ is dominated by a few shared directions (a global anisotropy offset). Measurable: the residual spectrum. **E:** low-rank for fixed interfaces, high-rank for learned teacher-side ones.

### 3.3 What a learnable map does to the objective

**Prop. 4 (learned teacher-side projectors are shortcuts).** Instantiation of Bhattarai et al. 2509.25253, Thm 2: for non-orthogonal $W$, zero projected loss does not imply Gram matching. When the student initialisation is strongly anisotropic — the normal case for pretrained small encoders — the least-squares $W$ from targets to initial student states already attains near-perfect cosine with near-zero Gram correlation: a degenerate near-minimiser adjacent to initialisation, reachable by gradient descent (dimensional-collapse mechanism of Jing et al. 2110.09348). **E:** this explains jina-v5's unexplained collapse of the "teacher-side unfrozen" arm and predicts learned t→s to be the worst interface. A student-side lift $\psi$ has no such shortcut at initialisation (it can only match the teacher's mean direction); **E:** student-side learning is not harmful, merely unnecessary — it adds parameters and a moving target without adding preserved structure.

**Measurement 3.3 (Table A1).** Geometry of the student *at initialisation*: mean pairwise cosine, effective rank, and the participation ratio of the cross-covariance between PCA targets and initial student states, per student. **E:** small pretrained encoders start nearly one-dimensional (the shortcut of Prop. 4 is wide open); deeper students start with richer geometry.

### 3.4 Fixing the gauge

**Identification argument.** Arms {Procrustes $R$ to the student init, no rotation, random rotation} share *every* orthogonally-invariant property of the target by construction (Gram, energy, ceiling score). Any downstream difference between them is therefore an optimisation-path effect from the student's initialisation: geometry is rotation-invariant, pointwise optimisation is not. **E:** the effect is small when the initial student geometry is nearly one-dimensional (participation ratio ≈ 1; $R$ reduces to mean alignment) and larger when it is richer. This justifies fixing the gauge once, at initialisation, and never re-solving it.

**What the analysis does not claim (state it):** no optimality of PCA under the cosine loss (only under Gram/MSE); no derivation that the residual is low-rank (measured); no claim that the gauge effect is large.

---

## 4. Method: what the analysis dictates (≈0.5 page)

$P_T=P_{PCA}R$; $R$ = orthogonal Procrustes solution aligning PCA targets to the student's initial pooled states, fitted once on a sample and frozen; $\tau_i=\mathrm{norm}(t_iP_T)$; $\mathcal{L}=\mathcal{L}_{end}+\lambda\,\mathcal{L}_{InfoNCE}$ at the final layer; AdamW. Algorithm box, ≤8 lines.

| Design choice | Dictated by |
|---|---|
| Teacher-side, frozen | Prop. 4 (learned teacher-side = shortcut); Prop. 3 (fixed orthonormal ⇒ structure carried by pointwise loss) |
| Principal subspace rather than random | Prop. 1 (least angular distortion); random is the control that shows *why* |
| One Procrustes rotation to the student init, never refit | §3.4 identification argument |
| Cosine, not MSE | targets are unit-norm; Prop. 3 is stated on the sphere |
| InfoNCE | anti-collapse regulariser; pointwise matching alone leaves the tail of the student spectrum unconstrained (Jing et al.; cf. contrastive uniformity) |
| No relational term | Prop. 3 |
| Last layer only | the interface is defined at the endpoint; last-layer-only supervision is supported by Ko et al. 2302.01530 and 2502.04499 — a design choice, not a claim |
| No inference-time parameters | nothing learned lives between student and target |

---

## 5. Experiments (≈3 pages) — see `docs/experiments_and_figures.md`

Order follows the argument: the analysis made predictions → verify them under control → show that every design choice of the method is one of those predictions → then show the method is the strongest. Four experiments; three tables and three figures in the main text.

**5.1 Setup.** Three teacher→student pairs spanning teacher family and student width/depth: Qwen3-Embedding-4B → BERT-base; BGE-M3 → MiniLMv2-L6-H768; Qwen3-Embedding-0.6B → MiniLMv2-L6-H384. Unlabeled corpus deduplicated against every evaluation set. Nine sentence-level tasks (classification F1, pair-classification AP, STS Spearman). One set of hyperparameters (AdamW, lr, batch, epochs, λ) shared by every arm and baseline; three seeds everywhere.

**5.2 Experiment 1 — the interface decides (Table 2, Figure 2).** One arm per interface family of §2, everything else fixed: fixed principal subspace (ours) · fixed random subspace · learned teacher→student · learned student→teacher · per-step orthogonal re-alignment. **E:** ours ≥ learned s→t ≳ random ≫ learned t→s (Props. 1, 2, 4), per-step re-alignment ≈ ours, and a monotone relation between target angular distortion and score across the fixed interfaces. **Figure 2 — the twist:** final endpoint loss against downstream score; **E:** the learned teacher-side arm reaches the *lowest* loss and the *lowest* score with a collapsed effective rank — the shortcut of Prop. 4 made visible. Answers sbert v5.5, jina-v5's projection ablation and LEAF's untested rejection of PCA in one table.

**5.3 Experiment 2 — every design choice of §4, removed one at a time (Table 3).** Smallest pair, interface fixed to ours unless the row changes it; one row per line of the §4 table: cosine → MSE (the sentence-transformers recipe) · no InfoNCE · + a Gram term · no gauge (PCA only) · gauge refit every epoch · no teacher (SimCSE in the same harness, the floor). **E:** MSE below cosine (targets are unit-norm; Prop. 3 lives on the sphere); no-InfoNCE loses effective rank and score (tail of the spectrum unconstrained); + Gram within noise (Prop. 3); no-gauge and refit-gauge within noise on shallow students (§3.4: nothing to re-align when the participation ratio ≈ 1). Each expectation is a consequence of the analysis — that is the point of the table. Sanity variants (non-orthonormal random basis, centring choices, contrastive view) go to Table A5.

**5.4 Experiment 3 — what the student actually inherits (Figure 3).** No new runs: from the saved embeddings of Experiments 1–2, compare teacher and student pairwise structure for ours, learned t→s, learned s→t and the no-teacher floor, with two reference lines — the PCA target itself (ceiling) and an independently trained encoder of the same size (what a good student shares with this teacher by default). **E:** ours near the ceiling and far above the independent encoder; learned t→s *closer to its targets pointwise* yet *further from the teacher pairwise*, with a high-rank residual (Observation 3.2). The geometric face of the dissociation in Figure 2.

**5.5 Experiment 4 — the method the analysis dictates is the strongest (Table 1).** Ours vs Stella, CDM, DSKD, EMO, TALAS on all three pairs. **E:** matches or exceeds the strongest multi-layer baseline on the two shallow students, within noise on BERT-base, with no inference-time parameters and one loss weight. Two matched-protocol rows in the appendix version (TALAS with AdamW; ours on TALAS's small-corpus setting) so no gain can be attributed to optimiser or data.

**5.6 Gauge (one paragraph; App. B).** PCA + Procrustes vs PCA + random rotation (null band), and the interpolation between the Procrustes solution and a random rotation. **E:** a monotone curve only where the student's initial geometry is rich (participation ratio ≫ 1), flat where it is near one — the identification argument of §3.4.

## 6. Related work (≈0.75 page; may sit after §2)

- *Sentence-embedding distillation:* Reimers & Gurevych 2004.09813; HPD 2203.07687; DistillCSE; Jasper/Stella 2412.19048; EMO (EMNLP 2025); TALAS 2606.21851; LEAF 2509.12539; jina-v5 2602.15547; EmbeddingGemma 2509.20354; sentence-transformers v5.5.0; Model2Vec; Wada et al. 2506.04624.
- *Projectors and shortcuts in feature distillation:* Miles & Mikolajczyk 2303.11098; Chen et al. 2310.17183; VkD 2403.06213; Bhattarai et al. 2509.25253; TCS 2412.09388; LELP 2409.20449; Han 2607.03572.
- *Relational distillation:* RKD 1904.05068; SP 1907.09682; CRD 1910.10699; Space Similarity 2409.13939; VRM 2502.20760 — cited to position Prop. 3.
- *Geometry of embedding spaces:* anisotropy (Ethayarajh 1909.00512; Godey et al. 2401.12143); whitening/ABTT (1702.01417; 2103.15316); dimensional collapse (2110.09348); intrinsic dimension and redundancy (Tsukagoshi & Sasano 2506.01435; Kisako et al. 2606.01074); caveats on spectrum vs quality (Kulkarni et al. 2602.20433; Nastase & Merlo 2509.01606); cross-model alignment (Platonic 2405.07987; vec2vec 2505.12540; Maystre et al. 2510.13406; Structure Retention 2605.22202).
- *Last-layer-only supervision (design support, one sentence):* Ko et al. 2302.01530; 2502.04499.
- *Theory of distillation in regression:* Phuong & Lampert 2019; Harutyunyan et al. 2301.12245; Mobahi et al. 2002.05715; Wu et al. 2606.01292.

Positioning sentence: *We propose no new loss; we analyse the geometry of the interface between teacher and student, derive from it which design choices can and cannot transfer structure, and show that the method the analysis dictates is also the strongest.*

---

## 7. Discussion and limitations (≈0.5 page)

- The bound in Prop. 3(a) is vacuous at practical loss values; the low-rank residual is an empirical regularity we measure, not derive.
- The gauge effect is expected to be small for students with near-one-dimensional initial geometry; it is a finding about *where* gauge matters, not a universal gain.
- Retained variance is not a quality predictor (Kulkarni et al.); we use angular distortion of the *target* to predict *transferability* inside a controlled pipeline, not to compare models (Nastase & Merlo).
- Coordinate-level scores (STS) saturate early for any fixed interface; the separation between interfaces is expected on classification and pair tasks, which depend on neighbourhood structure — we report all nine.
- English only; three pairs; cosine-metric downstream tasks only.

## 8. Conclusion (≈0.25 page)

The hard part of cross-dimensional embedding distillation is the geometry of the endpoint interface. Analyse it, and the method writes itself: a frozen near-isometric image of the teacher, gauge-aligned to the student's initialisation, matched pointwise. The adaptivity the field is currently adding is either a shortcut or adds nothing the interface did not already carry.

---

## Appendices
A. Proofs (Props. 1–4; renormalisation lemma; JL constants).
B. Target construction; student-init geometry (Table A1); gauge experiments (Figure A1, Table A3).
C. Appendix version of Table 1 (per-task ± std, matched-protocol rows); Table 2 on the other two pairs (Table A4); sanity variants (Table A5).
D. Geometry-inheritance metrics: definitions, probe set, probe-size sweep for k-NN (Table A2); Figure A2 — Gram heatmaps of the same sentences ordered by the teacher, and a retention map (teacher 2-D layout, colour = per-sentence kNN@10 overlap); both rotation-invariant, no cross-space transform.
E. Matched-hyperparameter protocol vs TALAS.
F. AI-use disclosure.

---

## Logic-chain check (each arrow must be supported before submission)

1. Interfaces differ only in (angular structure preserved, gauge) → taxonomy (§2). *By definition.*
2. PCA is the least-distorting fixed image; random preserves angles within JL (Props. 1–2; Measurement 3.1) → **Table 2 ordering** and **distortion-vs-score relation** (Figure 1 + Figure 2).
3. Fixed orthonormal interface ⇒ relational terms redundant (Prop. 3) → **Gram-term arm within noise** (Table 2) and **student inherits pairwise structure** (Figure 3).
4. Learned teacher-side interface has a shortcut at initialisation (Prop. 4; Measurement 3.3) → **dissociation + rank collapse** (Figure 2).
5. Gauge differences are optimisation effects (§3.4) → **null band + interpolation curve** (Figure A1).
6. Therefore the method of §4 → **state of the art at matched data/optimiser** (Table 1), with **each design choice individually justified** (Table 3).

**Go/no-go rule (end of week 1, smallest pair, one seed):** *Go* if the fixed interface beats the learned teacher-side interface and per-step re-alignment by a clear margin with the learned arm at lower training loss, and the Gram-term arm sits within noise of ours. *No-go* if learned, random and Gram-term arms are all within noise of ours with nothing dissociating — then the analysis has no contrast to explain and the paper should be redirected to an ACL/EMNLP analysis track.
