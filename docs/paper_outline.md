Đúng. Nếu mục tiêu là **ngắn, không lặp, mỗi section có một chức năng riêng**, tôi sẽ chốt outline như này.

# 1. Introduction

Chỉ **4 đoạn ngắn**.

* **P1 — Problem:** cross-architecture embedding KD có width mismatch và coordinate ambiguity.
* **P2 — Gap:** projection tạo một equivalence class \(\{YR\}\); các representative giữ cùng geometry nhưng không nhất thiết là cùng optimization target.
* **P3 — Solution:** GATE-KD chọn một **student-conditioned global representative**, rồi dùng \(H_0\) để support native-space structure.
* **P4 — Contributions/results:** 3 contributions + 2–3 headline numbers.

Không giải thích theory, invariant loss, projector, PCA ở Introduction. Chỉ đặt problem và answer.

---

# 2. Related Work

Không cần subsection nếu muốn cực gọn. Ba paragraph:

1. **Embedding distillation:** endpoint, projector, hidden-state/layer alignment.
2. **Representation equivalence & invariant alignment:** TCS, Procrustes/invariant KD, equivalence-class work.
3. **Structural distillation:** relational / geometric / topology.

Kết section bằng đúng gap:

> Prior work studies how to align or quotient equivalent representations; we study **which equivalent representative should be used as a persistent optimization target for a particular student**.

Sau câu này không nhắc lại literature gap nữa trong paper.

---

# 3. GATE-KD

## 3.1 Student-space target and gauge ambiguity

Làm hai việc cùng lúc:

* teacher endpoint → student-width projection;
* formalize orbit:

$$
[Y]=\{YR:R\in O(d)\}.
$$

Sau đó establish một fact duy nhất:

> same Gram geometry, different pointwise optimization objective.

Không cần riêng một “problem formulation section”.

---

## 3.2 Student-conditioned gauge fixing

Đây là **core subsection**.

$$
R^\star=\arg\min_{R\in O(d)}\|YR-Z_0\|_F^2.
$$

Bao gồm luôn theory chính:

* Procrustes = minimum-displacement representative;
* global gauge tạo một coherent representation;
* batchwise invariant matching cho phép independent local gauges và không identify cross-batch geometry;
* learned map có thể absorb deformation thay vì constrain deployed representation.

Tức **theory nằm cùng chỗ với mechanism mà nó giải thích**, không tạo Theory section riêng.

Nếu có gauge-stability proposition thì để cuối subsection này.

---

## 3.3 Native-space structural support

Giải thích \(H_0\) và joint loss:

$$
L=L_{\text{end}}+\lambda L_{H_0}.
$$

Hierarchy rõ:

* aligned endpoint = sample-level supervision;
* \(H_0\) = native-space structural support.

Paper hiện tại đã cho thấy endpoint mang phần lớn gain và \(H_0\) bổ sung thêm improvement, nên framing này phù hợp evidence. 

Kết section bằng algorithm.

---

# 4. Experiments

Chỉ **3 subsections**.

## 4.1 Experimental Setup

Models, corpus, metrics, baselines, training fairness.

Không analysis ở đây.

---

## 4.2 What Determines a Good Distillation Target?

Gom toàn bộ controlled experiments vào **một scientific sequence**:

**Orientation → global coherence → stability → subspace.**

Cụ thể theo thứ tự:

1. **Same geometry, different orientation:** aligned ≫ unaligned/random.
2. **Global gauge vs invariant matching:** fixed global > per-minibatch invariant.
3. **Fixed vs refit:** \(R_0\) gần repeated refitting.
4. **Subspace sensitivity:** aligned random subspace gần PCA.

Không cần 4 subsubsections; dùng 4 bold questions trong prose.

Kết luận duy nhất của subsection:

> **The dominant factor is selecting a stable student-conditioned representative, rather than repeatedly estimating alignment or maximizing retained variance.**

Evidence hiện tại support rất trực tiếp conclusion này.  

---

## 4.3 Distillation Performance

Chỉ practical validation:

* main 3 teacher–student × 9 tasks;
* endpoint vs endpoint+\(H_0\);
* structural controls;
* BEIR retrieval nếu còn space.

Không nhắc lại “why gauge works” ở đây.

Narrative:

> Section 4.2 establishes the mechanism; Section 4.3 asks whether it translates into competitive distillation.

Full method hiện đạt highest numerical average ở cả ba configurations, với margin rõ nhất ở 22M student. 

---

# 5. Discussion & Conclusion

Không subsection.

Chỉ 3 đoạn:

* **Implication:** information equivalence does not imply optimization equivalence.
* **Scope:** selected gauge là student-relative, không phải canonical coordinates; theory nói về coherence/identifiability, không claim giải thích toàn bộ nonlinear training.
* **Conclusion:** one-paragraph recap.

Không nhắc lại experiments từng cái một.

---

# Flow cuối cùng

**Introduction:** vấn đề là gì?
→ **Related Work:** gap nằm đâu?
→ **Method:** representative được chọn như thế nào và tại sao coherent?
→ **Experiments:** mechanism có thật không, rồi có giúp performance không?
→ **Discussion:** principle rộng hơn là gì?

Và mỗi ý chỉ xuất hiện **một nơi chính**:

* **equivalence-class gap:** Intro + positioning cuối Related Work
* **theory:** chỉ Section 3.2
* **mechanistic evidence:** chỉ Section 4.2
* **benchmark performance/H0 gain:** chỉ Section 4.3
* **broader implication:** chỉ Discussion

Tôi nghĩ đây là bản outline sạch nhất và gần style một ICLR paper mạnh hơn.
