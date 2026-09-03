# Paper Outline

## 1. Motivation: The Endpoint Is the Bottleneck

Embedding distillation faces two coupled problems: teacher and student endpoint
spaces are not directly comparable, while many existing methods introduce learned
projectors, intermediate-layer supervision, or self-distillation to manage transfer
through the representation hierarchy.

**Hypothesis.** The primary bottleneck is the formulation of a comparable endpoint,
not the path the student takes through its intermediate layers.

## 2. Core Idea: Intrinsic--Extrinsic Decomposition

We decompose teacher knowledge into two complementary transfer channels:

- **Extrinsic interface:** how teacher representations are expressed in coordinates
  that the student can match pointwise.
- **Intrinsic structure:** coordinate-invariant properties of the representation,
  such as its persistent connectivity signature.

Coordinate-dependent knowledge must pass through a shared endpoint interface;
coordinate-invariant structure can be transferred directly between the native
teacher and student spaces.

## 3. Extrinsic Interface

Let the endpoint target used in epoch $e$ be

$$
\tau_i^{(e)}=\operatorname{norm}(t_i P_{\mathrm{PCA}}R^{(e)}).
$$

- PCA is fitted once on the full cached teacher corpus and remains fixed. It selects
  the teacher subspace that can pass through the dimensional bottleneck.
- Before training, orthogonal Procrustes fits $R^{(0)}$ against the initialized
  student using a fixed calibration subset of at most 16,384 corpus sentences.
  After every epoch, $R^{(e)}$ is refitted in closed form against the current
  student on that same subset.
- Every refit changes only the coordinate orientation of the endpoint targets:
  because $R^{(e)}$ is orthogonal, their pairwise geometry and topology remain
  unchanged.
- The endpoint loss $L_{\mathrm{end}}^{(e)}$ anchors each student representation to
  the currently aligned teacher target.

The PCA subspace is frozen, while Procrustes maintains endpoint comparability by
alternating with student optimization once per epoch. The refit is an exact
closed-form coordinate update, not a learned projector, and adds no parameters at
training or inference.

## 4. Intrinsic Structure Transfer

The topological term $L_{H_0}$ matches the zero-dimensional persistent signatures
of student batches and the original, unprojected teacher batches. Because these
signatures depend on pairwise distances rather than coordinates, teacher and
student need neither the same dimensionality nor a shared orientation.

This transfers multiscale connectivity structure directly in the native spaces,
without intermediate-layer guidance or layer-wise propagation.

## 5. Minimal Distillation Objective

$$
L^{(e)}=L_{\mathrm{end}}^{(e)}+\lambda L_{H_0}.
$$

The method uses only two terminal objectives: pointwise endpoint matching and
coordinate-invariant topology matching. It requires no learned projector,
contrastive or self-distillation objective, intermediate supervision, or
inference-time parameters. Between epochs, the Procrustes block is updated exactly
while the PCA subspace and teacher geometry stay fixed.

## 6. Analysis

### 6.1 Is the Bottleneck Compression or Comparability?

Compare learned projection, random projection, PCA, and PCA+Procrustes while
separating subspace selection from orientation. Low-dimensional targets retain
useful teacher structure, but transfer quality depends strongly on whether the
endpoint is compatible with the student.

**Takeaway:** Compression is not the dominant bottleneck; endpoint comparability is.

### 6.2 Same Geometry, Different Optimization

Apply different fixed orthogonal rotations to the same PCA targets. Their pairwise
geometry and topology are identical, yet they can induce different optimization
trajectories and downstream performance. Compare no rotation, random rotations,
one-off Procrustes, and the method's epoch-wise Procrustes refit to isolate the
benefit of continually restoring endpoint comparability.

- **Figure 2:** one standalone distribution plot of downstream scores across Haar
  rotations, with point estimates for the PCA gauge, one-off Procrustes, and
  epoch-wise refitting. Report CKA and Gram error as numeric invariance controls.
- **Figure 3:** one standalone pre/post-refit curve on the frozen calibration
  subset, showing the instantaneous change at each epoch boundary.

**Takeaway:** Intrinsic equivalence does not imply extrinsic trainability; periodic
closed-form realignment maintains a usable extrinsic interface as the student
changes.

### 6.3 Which Signals Require an Interface?

Compare no-teacher, endpoint-only, $H_0$-only, and combined supervision, then
compare $H_0$ computed from the original teacher with $H_0$ computed from its PCA
image. Pointwise correspondence requires shared coordinates, whereas persistent
structural information can be matched directly across different dimensions. Then
test whether intermediate-layer or self-distillation supervision adds value once
both terminal signals are properly specified.

- **Figure 4:** two absolute death-time residual maps, endpoint-only versus
  endpoint+$H_0$, evaluated on the same fixed mini-batches with one shared scale.

**Takeaway:** Pointwise knowledge requires an interface; coordinate-invariant
structure bypasses it; neither requires intermediate-layer coordination.

## 7. Main Results and Takeaway

Compare against state-of-the-art embedding-distillation methods under matched data,
optimization, and caching protocols. Report downstream quality together with
training throughput, wall-clock time, peak memory, and inference-time parameters.

The target result is a simpler method that reaches state-of-the-art performance
while training approximately $2\times$ faster than TALAS.

## 8. Appendix Plan

The appendix keeps only two required visual artifacts. Projection variants and
refit schedules move to tables so that evidence is not duplicated across figures.

### Figure A1: Qualitative $H_0$ Visualization

Show three MSTs on one held-out batch: teacher, endpoint-only, and
endpoint+$H_0$. Use one fixed teacher-derived two-dimensional layout for display,
but recompute each edge set from distances in its native embedding space. State
explicitly that $L_{H_0}$ matches sorted death times, not corresponding MST edge
identities. This figure is illustrative; Figure 4 is the quantitative evidence.

### Figure A2: Sensitivity

Use one compact row of three ordered curves with mean $\pm$ sample standard
deviation over three seeds: topology weight $\lambda$, $H_0$ batch size, and the
fixed gauge-calibration sample size. Mark the default recipe. PCA remains fitted
once on the full teacher cache and only one factor changes at a time.

### Figure A3 (Optional): Ours vs. TALAS Layerwise CKA

If the pattern is stable across seeds and at least two teacher--student pairs, show
only endpoint+$H_0$ and TALAS heatmaps on the same held-out probe set and shared
color scale. Otherwise omit this descriptive figure. Downstream results, rather
than CKA, carry the claim that intermediate supervision is unnecessary.

### Supporting Tables

- **Table A1:** projection/interface ablation replacing the former PCA figure,
  including retained energy, Gram error, $k$-NN overlap, and downstream score;
- **Table A2:** exact per-setting values behind Figure A2, including downstream
  score, throughput, refit time, and peak memory;
- **Table A3:** per-seed values and random-projection/rotation draw variance behind
  the main analyses;
- **Table A4:** complete downstream results and efficiency measurements, including
  throughput, wall-clock time, peak memory, preprocessing cost, and inference-time
  parameters.

## Story in One Line

**A fixed PCA subspace retains usable teacher structure $\rightarrow$ epoch-wise
orthogonal refitting keeps the endpoint comparable without changing its intrinsic
geometry $\rightarrow$ coordinate-invariant topology transfers directly from the
native teacher space $\rightarrow$ no need to manage the intermediate hierarchy
$\rightarrow$ simpler, faster, and better distillation.**
