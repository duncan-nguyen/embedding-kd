# Outline — "Frozen Spectral Targets" (ICLR 2027)

Bản 2026-08-28. Thay thế story GeoODE-KD / L_dyn / L_vel. Cơ sở: memo bản 3
"Kiểm định PCA + Procrustes"
(https://claude.ai/code/artifact/8383fd00-48d5-4ed5-9898-533b5b981ace).

Deadline: abstract 18/09/2026, full 25/09/2026 (AoE). 9 trang main text.

---

## 0. Một câu thesis

> Trong cross-dimensional embedding distillation, mọi độ tự do *thích ứng* của
> target map (projector học, Procrustes giải lại mỗi batch) đều bị dùng để giảm
> loss thay vì dạy student. Map đúng là closed-form, cố định trước khi train:
> `P_T = P_PCA · R` — subspace từ teacher (Eckart–Young), orientation từ student
> init (Schönemann) — và chỉ cần cosine endpoint + InfoNCE là đủ thắng các
> learned-projector method, với 0 tham số thêm và 0 hyperparameter mới.

Tiêu đề ứng viên (chọn 1):
1. **Frozen Spectral Targets: Cross-Dimensional Embedding Distillation Without a Projector**
2. Don't Learn the Target Map: Closed-Form Teacher Targets for Embedding Distillation
3. How Adaptive Should the Target Map Be? Fixed Spectral Targets for Cross-Dimensional Distillation

Ba contribution (giữ đúng 3, mỗi cái có 1 bằng chứng cụ thể):
- **C1 (recipe + kết quả):** target map cố định closed-form + cosine + InfoNCE
  thắng learned-projector baselines (TALAS, HPD, learned s→t / t→s, per-batch
  Procrustes) ở hyperparameter matched, trên 3 cặp teacher→student, 3 seed.
- **C2 (hiện tượng shortcut, đo được):** khi target map thích ứng, train
  distillation loss giảm mà downstream giảm theo — dissociation curve theo mức
  thích ứng. Neo lý thuyết: Bhattarai 2509.25253 Thm 2; minh hoạ trong setting
  mình: least-squares t→s đạt cos 0.995 ngay tại init vì student init suy biến.
- **C3 (phân tách subspace / orientation):** Stiefel = Grassmannian × O(d_S).
  Subspace phải lấy từ teacher (PCA giữ 91.8% var, Gram distortion 3% @384);
  orientation chỉ đáng fix khi student init có cấu trúc — participation ratio
  của cross-covariance dự đoán trước điều đó (MiniLM ≈ 1 → R vô hại nhưng vô
  dụng; BERT-base 38–52 → R có tác dụng).

Những gì **không** xuất hiện trong bài: L_dyn, L_vel, vector field, depth
schedule, "trajectory supervision", "bảo toàn không gian pretrained của student".

---

## 1. Introduction (~1 trang)

1. Bối cảnh: teacher embedding 1024–2560-d (Qwen3-Embedding, BGE-M3) → student
   384/768-d. Bắt buộc phải có target map. Thực hành hiện tại: học nó
   (TALAS `W_l ∈ R^{d_S×d_T}`, LEAF `W_out`, jina-v5 ψ, Jasper FC, DistilCSE M;
   sentence-transformers v5.5 vừa đổi PCA cố định → projection học).
2. Câu hỏi: target map nên có bao nhiêu độ tự do thích ứng?
3. Quan sát (Fig. 1 — dissociation): thích ứng càng nhiều, train loss càng thấp,
   downstream càng thấp. Projector là shortcut (Bhattarai Thm 2, VkD).
4. Đề xuất: đóng băng map ở nghiệm closed-form hai nhân tử. Không tham số, không
   tunable, inference là student gốc.
5. Contribution C1–C3.

Câu phải tránh: "we preserve the student's pretrained space" (sai trên MiniLM).

## 2. Related Work (~0.6 trang, gộp vào cuối intro hoặc section riêng)

- **Projector trong feature KD:** Miles & Mikolajczyk 2303.11098 (projector mã
  hoá thông tin — lập trường ngược), PEFD 2210.15274, Chen 2310.17183
  ("teacher-side projector kém hơn" — với projector *học*), VkD 2403.06213
  (orthogonal > free), Bhattarai 2509.25253, Flex-KD 2507.10155.
- **Embedding distillation:** Reimers 2020, DistilCSE 2112.05638, HPD 2203.07687
  (PCA teacher + learned student projection — baseline bắt buộc), Jasper
  2412.19048, LEAF 2509.12539, jina-v5 2602.15547, TALAS 2606.21851, EMO
  (EMNLP 2025), sentence-transformers recipe (2020–v5.4).
- **Procrustes / gauge:** Schönemann 1966; EdgePoint2 2504.17280 & CLIDD
  2601.09230 (Procrustes per-batch); KCD 2103.16844 (permute teacher về student
  init); 2607.03572 (equivalence class, ridge W học); Gauge Freedom 2603.06774,
  Holonomy 2601.21653; Williams 2110.14739, Kornblith 1905.00414.
- **Last-layer-only là đủ:** CR-ILD 2302.01530, GLMD 2306.06625, Yu/Wen/Mou
  2502.04499, TALAS Fig. 3 (đỉnh K=2, không có row K=1).

## 3. Method: Frozen Spectral Targets (~1.5 trang)

### 3.1 Setup
Cache `τ̃_i = norm(f_T(x_i))` một lần. Student `z_i = norm(Pool(h_i^{(L)}))`.
Chỉ layer cuối được supervise. (Tái sử dụng `method.tex` §Teacher Target
Construction, Eq. native-teacher, student-layer-embedding, bỏ ký hiệu ℓ.)

### 3.2 Nhân tử 1 — subspace từ teacher (Eckart–Young)
`P_PCA ∈ R^{d_T×d_S}`, fit trên cache. Mệnh đề 1: trong mọi rank-d_S map, PCA
scores tối thiểu `‖G_T − G_k‖_F` với sai số `(Σ_{i>k} σ_i^4)^{1/2}`
(Eckart–Young–Mirsky; classical MDS). Lemma nhỏ: bound cosine sau chuẩn hoá theo
per-row norm (tự chứng minh, appendix). Bảng phổ: 91.8% @384, 99.2% @768.

### 3.3 Nhân tử 2 — orientation từ student init (Schönemann)
Nhận xét: cosine endpoint loss **không** bất biến O(d_S); mọi metric downstream
**bất biến** O(d_S) → orientation là gauge. Fix một lần:
`R* = UVᵀ, UΣVᵀ = svd(T_PCAᵀ Z_0)`. Đóng băng. Ghi rõ: R không đổi Gram, không
đổi retained variance; chi phí = 1 SVD `d_S×d_S`.
(Tái sử dụng `eq:gauge-alignment`, `eq:projected-teacher`.)

### 3.4 Phổ thích ứng (định nghĩa các arm — dùng lại ở §5)
| mức | map | ai làm |
|---|---|---|
| học joint, s→t | `W ∈ R^{d_S×d_T}` | TALAS, LEAF, jina-v5 |
| học joint, t→s | `W ∈ R^{d_T×d_S}` | EMO, sbert v5.5 |
| closed-form mỗi batch | Procrustes per-batch | EdgePoint2, Bhattarai |
| refit mỗi epoch | `gauge_refit_every=1` | (ablation) |
| **cố định tại init** | `P_PCA R` | **ours** |
| cố định, không gauge | `P_PCA` | sbert ≤5.4, HPD (+ learned s) |
| cố định, gauge ngẫu nhiên | `P_PCA Q`, Q~Haar | (control) |

### 3.5 Objective
`L = mean_i (1 − ⟨z_i, τ_i⟩) + λ_ctr · InfoNCE(dropout views)`.
λ_ctr là hyperparameter duy nhất, chọn trên validation, dùng chung cho mọi arm.

## 4. Why adaptivity is a shortcut (~0.8 trang — phần "analysis" C2 + C3)

- **Prop. 2 (Bhattarai Thm 2, phát biểu lại):** projected-MSE/cosine = 0 với
  projector `W` không right-orthogonal ⇏ Gram(student) = Gram(teacher).
- **Minh hoạ tại init:** bảng cos đạt được bởi (a) Procrustes, (b) least-squares
  t→s, (c) least-squares s→t trên student *chưa train*: MiniLM 0.55 / 0.995 /
  0.55; BERT-base 0.69 / — / —. (b) = 0.995 vì student init anisotropy 0.98 —
  projector học sẽ bắt đầu từ nghiệm suy biến.
- **Grassmannian × fibre:** `St(d_T,d_S)/O(d_S) = Gr(d_S,d_T)`. Learned
  projector di chuyển trên Gr (đổi thông tin giữ lại) *và* trên fibre; per-batch
  Procrustes chỉ trên fibre nhưng liên tục. Cả hai đều làm target đuổi student.
- **Khi nào R có ý nghĩa:** định nghĩa participation ratio
  `PR = (Σσ_i)²/Σσ_i²` của `T_PCAᵀ Z_0`. PR ≈ 1 → R chỉ khớp hai vector trung
  bình (σ₁ = ‖mean τ‖‖mean z‖, kiểm chứng số 0.534 ≈ √0.286·√0.979). Dự đoán:
  ablation R ≈ 0 trên MiniLM, > 0 trên BERT-base. §5.4 kiểm chứng.

## 5. Experiments (~3 trang)

### 5.1 Setup
3 cặp (Qwen3-0.6B→MiniLM-H384, BGE-M3→MiniLM-H768, Qwen3-4B→BERT-base);
corpus `train_100k` **đã dedup SICK/STS-B**; 9 task + 3 retrieval + subset
MTEB-v2 nhỏ; 3 seed, mean±std; **cùng lr/batch/epoch cho mọi method** (nêu
rõ; nêu cả kết quả ở HP gốc của TALAS trong appendix). Baselines: SimCSE-only
(sàn), TALAS, HPD, EMO, DistilCSE-style, MiniLM/RKD-style relational,
PCA+MSE (sbert recipe).

### 5.2 Main results (Table 1)
3 block × {Teacher, Student base, baselines, ours}. Bold/underline như bảng
hiện tại. Nếu pair (a) vẫn thua TALAS ở HP matched: báo cáo thẳng, phân tích ở
§6.

### 5.3 Adaptivity spectrum (Fig. 1 + Table 2) — figure trung tâm
Trục x: 7 arm của §3.4 theo mức thích ứng giảm dần. Hai đường: train
distillation loss cuối (giảm theo thích ứng) và AVG downstream (đỉnh ở fixed).
Chạy trên pair (c) và (a). Đây là bằng chứng C2.

### 5.4 Subspace vs orientation (Table 3)
- Ceiling: đánh giá thẳng `τ` (teacher đã chiếu, không student) trên benchmark
  ở d ∈ {128, 256, 384, 512, 768} — PCA gần lossless ở 2×.
- R ablation: `P_PCA R` vs `P_PCA` vs `P_PCA Q` (random) trên 3 cặp, kèm cột
  PR tại init. Kỳ vọng: gain(R) tương quan với PR.
- (Phụ lục) learning curve: R có tăng tốc hội tụ trên BERT-base không.

### 5.5 Scale & robustness
Data sweep 15k→100k→200k (script sẵn); protocol 15k của TALAS; MRL-truncation +
int8 của student sau distill (trả lời LEAF); benchmark sạch vs đủ (contamination
row).

## 6. Discussion / Limitations (~0.4 trang)
- Student sâu (BERT-base 12 layer): multi-layer anchoring của TALAS có thể vẫn
  hơn — frozen targets là về *target*, orthogonal với việc chọn layer; có thể
  cắm vào TALAS (thí nghiệm phụ nếu kịp: TALAS với `W_l` thay bằng `P_PCA R`).
- R là lợi ích phụ thuộc student; với student suy biến nó vô hại nhưng vô dụng.
- Theory về "gauge từ init" là implicit-bias, không phải định lý (OFT/LoRA nói
  rotation rẻ) — nói rõ, không claim quá.

## 7. Conclusion (~0.2 trang)

## Appendix
A. Chứng minh Prop. 1 + lemma cosine sau chuẩn hoá.
B. Prop. 2 (phát biểu lại Bhattarai) + ví dụ suy biến tại init.
C. Chi tiết fit P_PCA, R; pseudo-code 10 dòng; chi phí.
D. Diagnostic đầy đủ (phổ teacher, PR từng student, cos trước/sau R).
E. Hyperparameter, HP-sensitivity (λ_ctr), kết quả ở HP gốc của từng baseline.
F. Bảng đầy đủ per-task, per-seed; contamination; retrieval; MTEB subset.
G. AI-use disclosure (bắt buộc ICLR 2027).

---

## Danh sách figure / table

| # | Nội dung | Nguồn số | Trạng thái |
|---|---|---|---|
| Fig 1 | Dissociation: train loss vs downstream theo mức thích ứng | grid §5.3 | **chưa chạy** |
| Fig 2 | Phổ teacher + Gram distortion theo d | diagnostic (chạy lại trên cache 100k) | có bản 4k |
| Fig 3 | Singular values của `T_PCAᵀZ_0` cho 3 student (log-scale) | diagnostic | có bản 4k |
| Tab 1 | Main results 3 cặp, 3 seed, HP matched | rerun | **chưa** |
| Tab 2 | 7 arm × {train loss, AVG clean, AVG all} | grid | **chưa** |
| Tab 3 | Ceiling + R ablation + PR | grid + eval script | **chưa** |
| Tab 4 | Scale sweep, retrieval, MRL/int8 | scripts sẵn | **chưa** |

## Claim → bằng chứng → rủi ro

| Claim | Bằng chứng cần | Nếu sai |
|---|---|---|
| Fixed > learned ở HP matched | Tab 1, Tab 2 arm 1/2 vs 4 | **paper sập** — no-go |
| Fixed > per-batch Procrustes | Tab 2 arm 3 vs 4 | mất câu trả lời cho Bhattarai; vẫn sống nếu ≈ |
| Dissociation | Fig 1 | C2 thành "quan sát", không phải finding |
| PCA gần lossless | Fig 2, ceiling | an toàn (đã có số) |
| R gain ↔ PR | Tab 3 | nếu R ≈ 0 mọi nơi: C3 thành "R vô hại", bài vẫn đứng trên C1+C2 |
| Thắng TALAS pair (a) | Tab 1 | báo cáo thật; §6 |

## Việc phải làm trước khi viết (thứ tự)

1. Dedup corpus; tính lại PCA/PR trên cache 100k thật (GPU).
2. Viết 5 arm còn thiếu: learned s→t (last layer), learned t→s, per-batch
   Procrustes, random-R, PCA+MSE. Mỗi arm là 1 criterion/flag nhỏ.
3. Grid pair (c) ở HP matched + 2 run sửa confound → **go/no-go**.
4. Nếu go: 3 seed × 3 cặp × {arm 1, 3, 4, 5, TALAS, HPD, SimCSE, PCA+MSE}.
5. Ceiling eval, retrieval, MTEB subset, scale sweep, MRL/int8.
6. Viết: §3–4 trước (không phụ thuộc số), §5 sau, intro cuối.

## Map sang latex hiện tại

| File | Giữ | Bỏ / viết lại |
|---|---|---|
| `sections/abstract.tex` | — | viết lại toàn bộ |
| `sections/introduction.tex` | đoạn 1 (bối cảnh) | mọi thứ về trajectory/transition; 3 contribution |
| `sections/motivation.tex` | — | bỏ hẳn (motivation cũ là về depth) |
| `sections/method.tex` | §Teacher Target Construction (Eq. 1–8, gauge, projected-teacher) | §Vector Field, §Depth Schedule, §Layer-Wise Flow, §Training Objective (rút còn 1 công thức) |
| `appendices/derivations.tex` | — | thay bằng Prop. 1, lemma cosine, Prop. 2 |
| `appendices/target_construction.tex` | khung 3 subsection | điền nội dung (PCA, gauge, **bỏ** Gram-consistent fitting nếu không dùng) |
| `tables/main_results.tex` | format | số mới (HP matched, seed) |
| `references.bib` | schonemann1966, hinton, reimers, talas, truong2025emo, xu2023distillcse, zhang2024jasper | thêm ~25 entry từ memo bản 3; bỏ ODE/flow citations |
