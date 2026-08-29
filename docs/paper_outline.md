# Paper outline — ICLR 2027

## Headline

**Primary title**
> **Fix the Interface, Not the Student: Structure Transfers for Free in Cross-Dimensional Embedding Distillation**

Selling point in one line: *the only design decision that matters when distilling a 1024–2560-d embedding teacher into a 384–768-d encoder is how the two spaces are connected — and once that connection is a frozen near-isometry, even a random subspace works, relational losses become redundant, and layer-wise supervision has nothing left to act on.*

Alternative titles (same story, different emphasis):
- *Random Subspaces Are Almost Enough: What Actually Transfers in Embedding Distillation* — leads with the most surprising expected result.
- *The Interface Is the Method: Frozen Isometric Targets Make Relational and Layer-wise Distillation Redundant* — leads with the negative results.
- *Distillation Is an Interface Problem* — shortest; pair with a descriptive subtitle.

Why this sells at ICLR: (i) it reverses the direction the field is moving in 2026 (LEAF, jina-v5, TALAS, sentence-transformers v5.5 all add adaptivity) with a controlled audit rather than a new loss; (ii) it has a theory chain built from published results, instantiated and stress-tested in one setting; (iii) it produces a recipe with zero inference-time parameters and one loss weight — the "simple baseline that was hiding in plain sight" genre that reviewers reward when the analysis is airtight.

Deadlines: abstract 2026-09-18, full 2026-09-25 (AoE). 9 pages, ≥3 seeds on every claim, mandatory AI-use disclosure.

---

## 0. The argument in one paragraph (the chain every section must follow)

Distilling a large embedding model into a small encoder of lower width is usually treated as a question of *what to supervise* (which layers, which relations, which loss). We argue it is first a question of *how the two spaces are connected*: before any loss is written, a teacher vector must be turned into a target that lives in the student's space, and this **endpoint interface** is the load-bearing design decision. We show that the right interface is a **fixed, near-isometric linear image of the teacher inside the student's space**. The top-$d_S$ principal subspace is the least-distorting such image, but a random orthonormal subspace — a Johnson–Lindenstrauss image that discards most of the teacher's energy yet preserves angles — is expected to be already competitive, which reveals that what must be preserved is *angular structure*, not variance. Once the interface is fixed and isometric, cosine regression to it (with InfoNCE against collapse) transfers the teacher's relational and neighbourhood structure jointly: every rung of a structural ladder (coordinates → Gram/CKA → k-nearest-neighbours → connectivity) moves together, so explicit relational terms are redundant. Adaptive freedom on the *teacher side* (a learned teacher→student projector) is provably a shortcut — it can minimise the loss at initialisation without transferring structure; adaptive freedom on the *student side* is harmless but unnecessary. Finally, because endpoint supervision moves only the top blocks of the student, intermediate-layer supervision and relational self-distillation across layers have no substrate. The paper is an audit whose conclusion is a minimal recipe.

Each section below states: claim → argument and supporting prior work/theory → expected evidence → the figure/table it owns. No results are asserted here; expectations are marked **E**.

---

## 1. Introduction (≈1 page)

**Hook — the field is adding adaptivity without measuring what transfers.** Learned projectors kept at inference (LEAF, ACL 2026); learned student-side maps with the frozen teacher-side variant reported as "less effective" (jina-embeddings-v5, SIGIR 2026); multi-layer anchoring plus relational self-distillation with sharpness-aware optimisation (TALAS, ACL 2026); a learnable `EmbedDistillLoss(projection_dim)` replacing the decade-old frozen-PCA recipe in sentence-transformers v5.5.0. All are justified by benchmark deltas; none measures which structure of the teacher the student inherits.

**Reframing.** Teacher and student have different widths. The interface that maps one into the other is a design decision with its own theory (which invariants a rank-$d_S$ image can preserve, and what fixes the residual rotational freedom), and it determines whether any downstream loss can transfer structure at all.

**Contributions.**
1. **The interface is the decision.** We formalise the endpoint interface between two embedding spaces of different width, give a taxonomy that places every existing method in one table, and prove — by instantiating Eckart–Young, Johnson–Lindenstrauss, the Procrustes bound and the projector-shortcut theorem in this setting — that a *fixed, near-isometric* image of the teacher is sufficient while a learned teacher-side projector is a shortcut. **E:** interface ordering is predicted by the angular distortion of the target, not by retained variance, so even a random subspace is competitive.
2. **Structure transfers for free; depth and relations are not the lever.** A structural audit (coordinates → Gram/CKA → k-NN → connectivity, teacher vs student) shows that pointwise cosine + InfoNCE on a fixed isometric interface already transfers relational and neighbourhood structure, and that only the top blocks of the student move. **E:** relational terms and multi-layer anchoring move no rung of the ladder and no layer of the lower stack.
3. **A zero-parameter recipe that sets the state of the art.** PCA to $d_S$ + one Procrustes rotation to the student's initialisation, frozen, cosine + InfoNCE at the last layer — no inference-time parameters, one loss weight. **E:** at matched data and optimiser it matches or exceeds the strongest multi-layer baseline (TALAS) across three teacher→student pairs, nine sentence tasks and retrieval, while being the simplest method in the comparison.

**Explicitly not a contribution:** PCA targets per se (sentence-transformers 2020–v5.4; HPD 2022). The contribution is the audit, the theory that organises it, and the three claims that contradict current practice.

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

**Figure 1 (schematic).** The two spaces, the five interface families, and where each published method sits. This is the map the paper fills in.

---

## 3. Theory: why a fixed isometric interface is enough (≈1.25 pages; proofs in App. A)

Every statement is elementary or a direct instantiation of a published result — say so. The value is in stringing them together for this setting and in measuring where the chain is loose.

**Prop. 1 (Eckart–Young for targets).** For centred teacher matrix $\bar T$ and orthonormal $P\in\mathbb{R}^{d_T\times d_S}$, the top-$d_S$ right-singular basis minimises $\|\bar TPP^\top\bar T^\top-\bar T\bar T^\top\|_F$ over rank-$d_S$ orthogonal projections. With a lemma bounding the effect of post-projection renormalisation, the target Gram deviates from the teacher Gram by at most the tail-energy ratio. **E:** for LLM-based embedding teachers the tail beyond $d_S=384$ is small, so the PCA target is a near-isometric image.

**Prop. 2 (a random orthonormal subspace is also an interface).** For Haar-distributed $P$ and unit $x,y$, $\mathbb{E}\langle Px,Py\rangle=(d_S/d_T)\langle x,y\rangle$ and the renormalised cosine concentrates around $\cos(x,y)$ with deviation $O(\sqrt{(1-\cos^2)/d_S})$ (Johnson–Lindenstrauss; Dasgupta–Gupta). A random subspace keeps only a $d_S/d_T$ fraction of the energy but preserves *angles* to first order. **E:** this is why a random fixed projection is competitive, and why the predictive quantity for transfer is angular distortion of the target, not retained variance.

**Prop. 3 (gauge equivalence ⇒ relational terms are redundant).** (a) On the sphere, $1-\langle z,\tau\rangle=\|z-\tau\|^2/2$, hence $|\langle z_i,z_j\rangle-\langle\tau_i,\tau_j\rangle|\le\sqrt{2\ell_i}+\sqrt{2\ell_j}$: every Gram/RKD/similarity-preserving discrepancy is bounded by the endpoint loss. (b) Conversely (Maystre et al. 2510.13406; Schönemann 1966), approximate Gram preservation implies the existence of an *orthogonal* map with small pointwise error, and orthogonal maps dominate unconstrained linear ones. Therefore, **for a fixed orthonormal interface, pointwise matching in a fixed gauge and relational matching are the same statement up to the gauge; a relational loss cannot carry information the endpoint loss lacks.** The gauge is fixed once by $R$.

**Observation 3.1 (where the bound is loose — the interesting regime).** At practical loss values the bound in 3(a) is vacuous, yet **E:** the student's Gram will track the teacher's far more closely than the bound allows. That is only possible if the residual $Z-\mathcal{T}$ is far from isotropic — a shared low-rank component (e.g. a global anisotropy offset). This is a measurable prediction: **E:** the residual spectrum is dominated by a few directions for fixed interfaces and is high-rank for learned teacher-side interfaces.

**Prop. 4 (learned teacher-side projectors are shortcuts).** Instantiation of Bhattarai et al. 2509.25253, Thm 2: for a non-orthogonal $W$, zero projected loss does not imply Gram matching. When the student initialisation is strongly anisotropic (the case for pretrained small encoders), the least-squares $W$ from targets to initial student states already attains near-perfect cosine with near-zero Gram correlation — a degenerate near-minimiser adjacent to initialisation, reachable by gradient descent (dimensional-collapse mechanism of Jing et al. 2110.09348). **E:** this explains jina-v5's unexplained "teacher-side unfrozen collapses", and predicts learned t→s to be the worst interface. A student-side lift $\psi$ has no such shortcut at initialisation (it can only match the teacher's mean direction), **E:** hence student-side learning is not harmful, merely unnecessary.

**Identification argument for the gauge.** Arms {Procrustes $R$, no rotation, random $Q$} share *every* orthogonally-invariant property of the target by construction (Gram, energy, ceiling score, all ladder rungs). Any downstream difference between them is therefore an optimisation-path effect from the student's initialisation: geometry is rotation-invariant, pointwise optimisation is not. **E:** the effect is small for students whose initial states are nearly one-dimensional (the cross-covariance participation ratio predicts it) and larger for students with richer initial geometry.

**What the theory does not claim (state it):** no optimality of PCA under the cosine loss (only under Gram/MSE); no derivation that the residual is low-rank (measured); no claim that the gauge effect is large.

---

## 4. The recipe (≈0.5 page)

$P_T=P_{PCA}R$; $R$ = orthogonal Procrustes solution aligning PCA targets to the student's initial pooled states, fitted once on a sample and frozen; $\tau_i=\mathrm{norm}(t_iP_T)$; $\mathcal{L}=\mathcal{L}_{end}+\lambda\,\mathcal{L}_{InfoNCE}$ at the final layer only; AdamW. Explicitly list what is absent: projector at inference, relational term, intermediate-layer term, sharpness-aware optimiser. Algorithm box, ≤8 lines.

---

## 5. Experiments (≈4 pages) — see `docs/experiments_and_figures.md` for the full protocol

**5.1 Setup.** Three teacher→student pairs spanning teacher family and student depth/width: Qwen3-Embedding-4B → BERT-base; BGE-M3 → MiniLMv2-L6-H768; Qwen3-Embedding-0.6B → MiniLMv2-L6-H384. Unlabeled corpus deduplicated against every evaluation set. Nine sentence-level tasks (classification F1, pair-classification AP, STS Spearman) + retrieval (ArguAna, FiQA, SCIDOCS) + an MTEB-v2 subset. Baselines: Stella, CDM, DSKD, EMO, TALAS, PCA+MSE (the sentence-transformers ≤5.4 recipe), SimCSE-only. Matched hyperparameters across all arms; TALAS additionally run without its sharpness-aware optimiser, and ours additionally on TALAS's small-corpus setting. Three seeds everywhere a claim rests.

**5.2 Main results (Table 1).** **E:** ours matches or exceeds the strongest multi-layer baseline on the two shallow students and is within noise on BERT-base; recovers most of the teacher's average with no inference-time parameters.

**5.3 Interface ablation — C1 (Table 2, Figure 2).** Same objective, corpus, initialisation; only the interface changes: PCA·R · PCA · PCA + random rotation · random subspace · MRL-prefix (leading dimensions) · learned t→s · learned s→t · per-step Procrustes · PCA + MSE. **E:** ordering PCA·R ≥ PCA ≥ learned s→t ≳ random ≫ learned t→s, and a monotone relation between target angular distortion and downstream score. **Figure 2 — dissociation:** final endpoint loss vs downstream score, one point per arm; **E:** learned t→s sits bottom-left (lowest loss, lowest score), with collapsed effective rank of the student space. Answers sbert v5.5, jina-v5's projection ablation, LEAF's untested rejection of PCA.

**5.4 Structural audit ladder — C2 (Figure 3, the central figure).** Rungs, teacher vs student on a fixed probe set: (1) cosine to target; (2) Gram distance / linear CKA / Procrustes distance; (3) k-NN overlap (N2O), reported across probe sizes to pre-empt the scale artefact of mutual-kNN; (4) H0 persistence / RTD. Ceiling = PCA target vs full teacher; floor = SimCSE-only. Arms: ours · + Gram/RKD term · + layer-wise relational self-distillation (TALAS-style) · + multi-layer anchoring · learned t→s · learned s→t. **E:** relational and layer-wise arms coincide with ours on every rung and on score; learned t→s is higher on rung 1 and lower on rungs 2–4. First such audit for sentence-embedding distillation; arbitrates the pro-relational (Bhattarai, TALAS, EMO) and pointwise (LEAF, jina-v5) camps with metrics instead of benchmark deltas.

**5.5 Depth — C3 (Figure 4).** Per-layer distance to the *initial* state and per-layer cosine to the target, over training, for ours / lower-stack frozen / + multi-layer anchoring / + layer-wise self-distillation, on all three students. **E:** only the top blocks move; freezing the lower stack matches full training within noise on shallow students; on the 12-layer student report honestly whether anchoring buys anything and, if so, state C3 with a depth condition. Extends Ko et al. 2302.01530, 2502.04499 and 2605.11513 to embedding students.

**5.6 The gauge as a measured optimisation effect (Figure 5; a finding, not a headline).** Null band over random rotations; interpolation $Q(\theta)$ between the Procrustes solution and a random rotation; a rank-one (mean-alignment) control; the cross-covariance participation ratio per pair as the predictor. **E:** monotone curve in $\theta$ where the participation ratio is high, flat where it is near one.

**5.7 Retrieval, scale, truncation (short).** Retrieval trio + MTEB-v2 subset; corpus-size sweep; MRL-style truncation curve of the distilled student (answers LEAF's "PCA destroys MRL").

---

## 6. Related work (≈0.75 page; may sit after §2)

- *Sentence-embedding distillation:* Reimers & Gurevych 2004.09813; HPD 2203.07687; DistillCSE; Jasper/Stella 2412.19048; EMO (EMNLP 2025); TALAS 2606.21851; LEAF 2509.12539; jina-v5 2602.15547; EmbeddingGemma 2509.20354; sentence-transformers v5.5.0; Model2Vec; Wada et al. 2506.04624.
- *Projectors and shortcuts in feature distillation:* Miles & Mikolajczyk 2303.11098; Chen et al. 2310.17183; VkD 2403.06213; Bhattarai et al. 2509.25253; TCS 2412.09388; LELP 2409.20449; Han 2607.03572.
- *Relational distillation and its measurement:* RKD 1904.05068; SP 1907.09682; CRD 1910.10699; G-CRD 2111.04964; Space Similarity 2409.13939; VRM 2502.20760.
- *Geometry of embedding spaces:* anisotropy (Ethayarajh 1909.00512; Godey et al. 2401.12143); whitening/ABTT (1702.01417; 2103.15316); dimensional collapse (2110.09348); intrinsic dimension and redundancy (Tsukagoshi & Sasano 2506.01435; Kisako et al. 2606.01074); caveats on spectrum vs quality (Kulkarni et al. 2602.20433; Nastase & Merlo 2509.01606); cross-model alignment (Platonic 2405.07987; vec2vec 2505.12540; Maystre et al. 2510.13406; Structure Retention 2605.22202).
- *Depth and layer selection:* Ko et al. 2302.01530; 2502.04499; 2605.11513; 2DMSE/ESE 2402.14776.
- *Theory of distillation in regression:* Phuong & Lampert 2019; Harutyunyan et al. 2301.12245; Mobahi et al. 2002.05715; Wu et al. 2606.01292.

Positioning sentence: *We propose no new loss and no new similarity measure; we identify the interface as the decisive design choice, prove when it makes relational and layer-wise terms redundant, and measure what the student inherits.*

---

## 7. Discussion and limitations (≈0.5 page)

- The bound in Prop. 3(a) is vacuous at practical loss values; the low-rank residual is an empirical regularity we measure, not derive.
- The gauge effect is expected to be small for students with near-one-dimensional initial geometry; it is a finding about *where* gauge matters, not a universal gain.
- If multi-layer anchoring helps the 12-layer student, C3 holds only for shallow students — report it.
- Retained variance is not a quality predictor (Kulkarni et al.); we use angular distortion of the *target* to predict *transferability* inside a controlled pipeline, not to compare models (Nastase & Merlo).
- k-NN agreement depends on probe size (2604.18572); we report a sweep.
- English only; three pairs; cosine-metric downstream tasks only.

## 8. Conclusion (≈0.25 page)

The hard part of cross-dimensional embedding distillation is the endpoint interface. Fix it well — a frozen near-isometric image of the teacher, gauge-aligned to the student's initialisation — and structure transfers on its own; the adaptivity the field is currently adding is either a shortcut or has nothing to act on.

---

## Appendices
A. Proofs (Props. 1–4; renormalisation lemma; JL constants).
B. Target construction; participation-ratio diagnostic; Procrustes fit details.
C. Full per-task tables, seeds, clean-subset robustness, retrieval, MTEB-v2 subset.
D. Ladder metrics: definitions, probe set, probe-size sweep, RTD settings.
E. Depth profiles for all pairs.
F. Matched-hyperparameter protocol vs TALAS.
G. AI-use disclosure.

---

## Logic-chain check (each arrow must be supported before submission)

1. Interfaces differ only in (angular structure preserved, gauge) → taxonomy (§2). *By definition.*
2. PCA is the least-distorting fixed image; random preserves angles within JL (Props. 1–2) → **Table 2 ordering** and **distortion-vs-score relation**.
3. Fixed orthonormal interface ⇒ relational terms redundant (Prop. 3) → **Figure 3: relational arms move no rung**.
4. Learned teacher-side interface has a shortcut at initialisation (Prop. 4) → **Figure 2 dissociation + rank collapse**.
5. Endpoint supervision moves only top blocks → **Figure 4 + frozen-lower-stack arm**.
6. Therefore layer-wise anchoring / self-distillation are inert → **arms in 5.4/5.5 coincide with ours**.
7. Gauge differences are optimisation effects (identification argument) → **Figure 5 null band + interpolation curve**.

**Go/no-go rule (end of week 1, smallest pair, one seed):** *Go* if the fixed interface beats the learned teacher-side interface and per-step re-alignment by a clear margin with the learned arm at lower training loss, and the relational/layer-wise arms sit within noise of ours on score and on every rung. *No-go* if learned arms, random subspace and relational arms are all within noise of ours with nothing dissociating — then the paper is an audit without a contrast and should be redirected to an ACL/EMNLP analysis track.
