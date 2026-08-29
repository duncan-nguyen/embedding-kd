# Ablation study

Danh sách mọi ablation gọi được bằng flag trong repo, kèm câu hỏi mỗi cái trả lời
và ý nghĩa của từng kết quả. Cột **default** là giá trị chạy khi không truyền gì —
tức là recipe, còn mọi dòng khác là ablation. Kết quả ở §6 (run 2026-08-29,
cặp Qwen3-0.6B → MiniLM-H384, **3/6 cell**).

Objective của `geoode` là **`L_end + L_ctr`** (`λ_end = 1`, `λ_ctr = 0.5`) và không
có gì khác — không term nào đọc layer trung gian. Khớp với
`docs/latex_iclr/sections/experiments.tex` và caption
`docs/latex_iclr/tables/main_results.tex`.

Entry point:

| | |
|---|---|
| Lưới target map | `scripts/run_target_map_ablation.py` (dry-run → `--execute` → `--collect`) |
| Trong notebook | `notebooks/audit_runs_qwen_minilm.ipynb` (mọi arm của `docs/experiments_and_figures.md`, matched HP, probe dump) + `notebooks/audit_analysis_qwen_minilm.ipynb` (bảng/figure post-hoc) |
| Ablation lẻ | truyền flag thẳng cho `main.py --method geoode` |
| Test của các arm | `tests/test_target_map_ablations.py` |
| Cache teacher | `--cache_dir` (mặc định của script: `runs/teacher_cache`) — dùng chung cho mọi cell, mọi lưới và cả 8 method của notebook, nên teacher chỉ chạy một lần cho mỗi (teacher, pooling, corpus) |

---

## 1. Target map `P_T = P_PCA · R` — lưới hai nhân tố

Nhóm chính. Mọi cell chỉ khác nhau ở target map: cùng corpus, cùng cache teacher,
cùng objective, cùng lịch train, cùng student init.

### 1.1 Nhân tố subspace — `--projection_type`, `--pca_center_fit`, `--pca_subtract_mean`

Kiểm định claim Eckart–Young: subspace phổ của teacher là thứ mang tín hiệu.

| arm | flag | hỏi gì |
|---|---|---|
| `pca` **(default)** | `--projection_type pca --pca_center_fit --no-pca_subtract_mean` | recipe: hướng chọn sau khi trừ mean, nhưng mean vẫn còn khi apply |
| `pca_full` | thêm `--pca_subtract_mean` | biến đổi PCA sách giáo khoa; bỏ luôn thành phần chung của teacher |
| `svd` | `--no-pca_center_fit` | uncentered SVD: hướng đầu tiên được phép chính là vector mean của teacher |
| `random` | `--projection_type random` | subspace Haar orthonormal cùng rank — giữ đúng contract của PCA, bỏ phổ |
| `random_gaussian` | `--projection_type random_gaussian` | Johnson–Lindenstrauss: **cùng subspace** với `random` ở cùng seed, bỏ thêm tính orthonormal |
| `mrl_prefix` | `--projection_type mrl_prefix` | `d_S` toạ độ đầu của teacher (Matryoshka prefix): orthonormal như PCA, subspace chọn theo *thứ tự toạ độ* — kỳ vọng giữa random và PCA với teacher train MRL, ≈ random với teacher khác |
| `learned_t2s` | `--projection_type learned_t2s` | `W ∈ R^{d_T×d_S}` học cùng student, chiếu teacher xuống (EMO, sbert v5.5) |
| `learned_s2t` | `--projection_type learned_s2t` | `W ∈ R^{d_S×d_T}` học cùng student, chiếu student lên không gian teacher (TALAS, LEAF, jina-v5) |

- `random` và `random_gaussian` ở cùng `--projection_seed` bốc từ cùng một ma trận
  Gaussian nên span cùng subspace: so hai arm này là cô lập đúng tính orthonormal.
- `--projection_seed k` là draw thứ k. Độ tản giữa các draw là **null band**;
  khoảng cách PCA-vs-random nhỏ hơn band đó thì không đọc được thành kết luận.
- Run in ra `explained_energy` kèm mốc `d_S/d_T` mà một subspace ngẫu nhiên giữ
  được, nên đọc được ngay arm này giữ lại nhiều hay ít hơn mức tình cờ.

**Nếu `pca ≈ random`:** subspace phổ không phải thứ mang tín hiệu, §Eckart–Young
của bài không còn cơ sở.

**Hai arm learned là loại control khác.** Chúng không *chọn* subspace nào cả — chúng
để map được học. Không có gauge đi kèm (gauge định hướng một basis đóng băng, map
học không có basis cố định nào để định hướng), và tính ngẫu nhiên của chúng nằm ở
khởi tạo `W` nên biến thiên theo `--seeds` chứ không theo `--draws`. `W` chỉ tồn tại
lúc train: model đem đi dùng vẫn là student encoder trần, nên thứ thay đổi là
*giám sát* chứ không phải artefact. Không term nào khác của objective bị đụng — cụ
thể `L_ctr` vẫn đọc embedding `d_S` chưa map của student, để hai arm chỉ khác một
thứ duy nhất.

Hai hướng không thay thế nhau được: `t2s` phải vứt `d_T − d_S` chiều và được tự chọn
vứt chiều nào theo cái gì làm loss nhỏ — đúng shortcut mà công thức đóng băng phản
đối (Bhattarai 2509.25253 Thm 2). `s2t` không vứt gì của teacher, nhưng `d_T − d_S`
chiều dư cho map chỗ trống để hấp thụ sai số mà student không bao giờ phải học.

`--learned_projector_lr_scale` (mặc định `1.0`) đặt lr của `W` theo bội số lr
student. Nó tồn tại để baseline được tune chứ không bị dựng thành bù nhìn.

### 1.2 Nhân tố orientation — `--gauge_align`, `--gauge_rotation`

Kiểm định claim Schönemann: gauge khớp student init tốt hơn gauge tuỳ ý.

| arm | flag | hỏi gì |
|---|---|---|
| `procrustes` **(default)** | `--gauge_align --gauge_rotation procrustes` | recipe: `R = UVᵀ` khớp student *chưa train*, fit một lần rồi đóng băng |
| `none` | `--no-gauge_align` | bỏ hẳn `R`. **Target map của nó là prior art**: `P_PCA` không gauge đúng bằng recipe sentence-transformers ≤ v5.4 và phần teacher-side của HPD (2203.07687) — nên đây là baseline bắt buộc, không chỉ là ablation. Khác biệt còn lại là loss (ta cosine, sbert MSE); arm đầy đủ `PCA + MSE` nằm ở §5 |
| `random` | `--gauge_align --gauge_rotation random` | `Q ~ Haar`, cùng chi phí, không dùng thông tin nào của student |

`none` **không** phải "không có gauge": basis PCA tự nó đã là một orientation tuỳ ý
(chọn theo hiệp phương sai teacher, không nhìn student). Nên `none` là *một điểm*
trong không gian orientation, không phải gốc trung tính — và `procrustes > none`
một mình không phân biệt được "R là orientation *đúng*" với "R là *một*
orientation". Cột `random` với nhiều draw (`--gauge_random_seed`) mới cho phân phối
null: story về gauge đúng thì `none` phải nằm *trong* phân phối đó, `procrustes`
nằm ngoài.

Cả hai rotation đều trực giao nên Gram matrix, retained variance và mọi đại lượng
bất biến `O(d_S)` không đổi — chỉ endpoint cosine loss đổi. Đây là ablation
một-nhân-tố sạch nhất trong lưới.

**Chẩn đoán đi kèm:** run có gauge ghi `participation_ratio` của cross-covariance
`T_PCAᵀZ₀`. `PR ≈ 1` nghĩa là ma trận này hạng 1 và `R` chỉ xoay một vector mean
lên vector mean kia — khi đó ablation `R` **được dự đoán** là null, không phải
nhiễu. Arm `random` còn ghi `cos_procrustes` (số Procrustes *sẽ* đạt được) trong
cùng dòng log để so trực tiếp. Run `none` bỏ hẳn forward pass student-init nên các
cột `cos_*`/`PR` của dòng đó trống; PR là tính chất của cặp (target, student init)
nên đọc từ dòng khác cùng cặp.

### 1.3 Gauge động — `--gauge_refit_every N`

Fit lại `R` theo student *hiện tại* sau mỗi N epoch (alternating exact minimisation
trên `O(d_S)`). `0` (default) = đóng băng cả run. Đây là nấc "thích ứng vừa" giữa
map cố định và Procrustes mỗi batch (EdgePoint2 2504.17280). Không dùng chung được
với `--gauge_rotation random` (bị từ chối ngay lúc dựng target: gauge ngẫu nhiên
không có gì để refit, và bước refit chỉ là bước giảm với Procrustes).

### 1.4 Lưới `requested` — ours với 4 control

6 run, mặc định của script. Đọc theo *hàng một claim*, không phải theo lưới:

| cell | là gì | trả lời câu hỏi |
|---|---|---|
| `pca__procrustes` | **ours** | — |
| `pca__none` | PCA only | gauge có làm gì không? |
| `pca__random` | PCA + xoay trực giao ngẫu nhiên | gauge có làm gì *mang thông tin* không? |
| `random__none` | random projection | subspace có làm gì không? |
| `learned_t2s__none` | learned projection t→s | map học có đáng hơn map đóng băng không? |
| `learned_s2t__none` | learned projection s→t | như trên, hướng ngược lại |

`full` = mọi tổ hợp subspace × gauge = 20 ô (6 × 3 cộng 2 arm learned không có cột
gauge). Nhân thêm `--draws` (chỉ với arm ngẫu nhiên) và `--seeds`.

```bash
python3 scripts/run_target_map_ablation.py --pair qwen3_4b_to_bert_base            # xem plan
python3 scripts/run_target_map_ablation.py --pair qwen3_4b_to_bert_base --execute  # chạy
python3 scripts/run_target_map_ablation.py --pair qwen3_4b_to_bert_base --collect  # đọc lại
```

`--collect` liệt kê **mọi cell đã plan**: cell chưa chạy hiện `missing`, nên bảng
cũng là danh sách việc còn lại.

---

## 2. Objective

Objective chỉ có hai term, và cả hai đều chỉ đọc layer cuối.

| flag | default | arm | hỏi gì |
|---|---|---|---|
| `--lambda_ctr` | `0.5` | `0` / khác | `0` bỏ InfoNCE → đo phần tín hiệu thuần từ teacher. Đây là weight duy nhất được tune |
| `--lambda_end` | `1.0` | `0` | bỏ endpoint → chỉ còn regulariser, về thực chất là `simcse` chạy trong khung geoode |

Không có flag nào khác chạm vào loss. `include_embedding_layer` (§3) chỉ đổi state
nào được báo cáo ở `cos_first`, không đổi gradient: `L_end` chỉ đọc `states[-1]`.
Bất biến đó được chốt bằng
`test_geoode_kd.py::test_the_supervised_depth_is_the_last_layer_alone`.

---

## 3. Chỉ chỉnh được trong config (chưa có CLI flag)

Sửa ở `config/geoode_config.py` hoặc truyền qua `GeoODEConfig(**kwargs)`.

| field | default | hỏi gì |
|---|---|---|
| `contrastive_view` | `dropout` | `pair` dùng câu thứ hai của cặp làm view thứ hai thay vì hai dropout mask |
| `contrastive_temperature` | `0.05` | `τ_c` của InfoNCE |
| `gauge_align_samples` | `16384` | số câu fit Procrustes; phải `>> d_S` để cross-covariance đủ điều kiện |
| `include_embedding_layer` | `False` | bật = coi output embedding là state độ sâu 0. **Không đổi loss**, chỉ đổi state mà `cos_first` báo cáo: `L_end` chỉ đọc `states[-1]` |

`student_pooling` có CLI (`--student_pooling mean`) và luôn có tác dụng — nó quyết
định vector nào bị supervise, và phải khớp pooling mà code evaluation dùng.

---

## 4. Ablation của các method khác

| method | flag | default | hỏi gì |
|---|---|---|---|
| `rkd` | `--normalize_student` | bật | tắt = ablation Euclid thô: đo quan hệ của student không chiếu lên mặt cầu mà teacher cache và mọi benchmark cosine đang sống |
| `rkd` | `--w_dist` / `--w_angle` | `25` / `50` | tách riêng thế năng khoảng cách và thế năng góc |
| `simcse` | `--simcse_view` | `dropout` | `pair` dùng câu ghép đôi làm positive |

`simcse` (không teacher) và `rkd` (teacher chỉ giám sát quan hệ ở layer cuối) là
hai mốc để đọc mọi method còn lại: phần vượt lên trên `simcse` chính là thứ tín
hiệu teacher mang lại.

---

## 5. Arm còn thiếu code

| arm | là gì | ai làm |
|---|---|---|
| Procrustes mỗi batch | giải lại closed-form theo từng batch — nấc thích ứng giữa `gauge_refit_every` và map học | EdgePoint2 2504.17280, Bhattarai 2509.25253 |
| ~~PCA + MSE~~ | **đã có**: `--endpoint_loss mse --lambda_ctr 0 --no-gauge_align` (dòng `pca_mse` của `main_tables.ipynb`, arm `pca__mse` của audit notebook) | — |

(learned s→t và learned t→s đã có code, xem §1.1.)

Cùng với hai arm learned, đây là bằng chứng cho luận điểm "mọi độ tự do thích ứng
của target map bị dùng để giảm loss thay vì dạy student": khi map thích ứng, train
loss giảm mà downstream giảm theo.

---

## 6. Kết quả

**Run 2026-08-29** `qwen3_0.6b_to_minilm_h384_100k_all_methods_20260829-144415`.
Qwen3-Embedding-0.6B (1024-d) → MiniLMv2-L6-H384, corpus `train_100k` (102,361
dòng), 5 epoch, batch 64, lr 5e-5, seed 42, objective `L_end + L_ctr`
(`λ_ctr = 0.5`). `eval_retrieval = False`, nên **AVG(ALL) ở đây là trung bình 9
benchmark câu, không có retrieval** — không so thẳng được với AVG(ALL) của bảng
main results.

> Trạng thái: 3/6 cell xong, `random__none` đang chạy, hai arm learned chưa tới.
> **1 seed, 1 draw** — chưa có null band, xem §6.2 trước khi đọc thành kết luận.

### 6.1 Lưới `requested`

| cell | AVG(ALL) | AVG(IOD) | AVG(OOD) | Δ vs ours | train `loss_end` | `cos_final` |
|---|---|---|---|---|---|---|
| `pca__procrustes` (**ours**) | **77.22** | 70.48 | 80.58 | 0 | **0.1921** | **0.8079** |
| `pca__none` | 76.77 | 70.01 | 80.15 | −0.45 | 0.2074 | 0.7926 |
| `pca__random` | 76.53 | 69.66 | 79.96 | −0.69 | 0.2099 | 0.7901 |
| `random__none` | *đang chạy* | | | | | |
| `learned_t2s__none` | *chưa chạy* | | | | | |
| `learned_s2t__none` | *chưa chạy* | | | | | |

Per-benchmark (primary metric, ×100):

| cell | banking77 | emotion | tweet | mrpc | scitail | wic | sick | sts12 | stsb |
|---|---|---|---|---|---|---|---|---|---|
| `pca__procrustes` | 91.52 | 66.50 | 74.55 | 85.28 | 82.58 | 67.15 | 77.18 | 72.38 | 77.80 |
| `pca__none` | 90.82 | 65.46 | 73.79 | 85.75 | 81.97 | 67.00 | 76.85 | 71.74 | 77.55 |
| `pca__random` | 90.77 | 64.35 | 73.41 | 85.43 | 81.56 | 67.05 | 76.84 | 71.76 | 77.57 |

Chẩn đoán của target map, đo trong chính các run trên:

| đại lượng | giá trị | nghĩa |
|---|---|---|
| `explained_energy` PCA 1024→384 | **91.8%** | subspace phổ giữ gần hết năng lượng cache |
| `explained_energy` random 1024→384 | **38.1%** | đúng mức tình cờ `d_S/d_T = 37.5%` |
| cos(z, τ) tại init, không gauge | +0.031 | student init gần như trực giao với target PCA |
| cos(z, τ) tại init, sau Procrustes | **+0.620** | gauge kéo target về đúng chỗ student đang đứng |
| cos(z, τ) tại init, sau `Q ~ Haar` | +0.013 | xoay ngẫu nhiên không mang thông tin nào, đúng như thiết kế |
| participation ratio của `T_PCAᵀZ₀` | **1.37 / 384** | top singular share 0.853 |

### 6.2 Đọc được gì, chưa đọc được gì

**Đọc được ngay — thứ tự đúng như story dự đoán.**
`procrustes (77.22) > none (76.77) > random (76.53)`, đúng thứ tự đó ở cả ba mức
AVG. Ở mức benchmark: ours cao nhất ở **8/9** (mrpc là ngoại lệ duy nhất, 85.28 vs
85.75/85.43), còn thứ tự ba bậc đầy đủ giữ ở 5/9 — bốn benchmark còn lại
(mrpc, wic, sts12, stsb) đảo `none` với `random`, mà chênh lệch ở đó chỉ 0.02–0.32
điểm. Gauge Procrustes cũng cho `loss_end` thấp nhất
(0.1921 vs 0.2074/0.2099) và `cos_final` cao nhất — nó làm endpoint loss dễ hơn
*và* downstream tốt hơn cùng lúc. Đây **không phải** dissociation, và đúng như
kỳ vọng: gauge cố định tại init không phải một độ tự do thích ứng nên không có
shortcut nào để đi.

**Đọc được ngay — `pca__random` thấp hơn `pca__none`.** Xoay ngẫu nhiên *tệ hơn*
là không xoay gì. Nghĩa là basis PCA không phải một orientation tuỳ ý trung tính:
nó đã tình cờ gần student hơn một draw Haar (cos init +0.031 vs +0.013). Điều này
làm cho `procrustes > none` khó đọc hơn chứ không dễ hơn — vì `none` không phải
điểm giữa của phân phối null.

**CHƯA đọc được — khoảng cách quá nhỏ so với bằng chứng đang có.** +0.45 và +0.69
trên 1 seed, 1 draw `Q`, không có null band. Theo đúng hai điều kiện ở dưới, chưa
được viết thành "gauge có tác dụng". Với `PR = 1.37 / 384` (top singular share
0.853) thì cross-covariance gần hạng 1 — `R` gần như chỉ xoay một vector mean lên
vector mean kia, nên **một hiệu ứng bé ở cặp này là dự đoán trước, không phải thí
nghiệm hỏng** (§6.3).

Việc phải làm để khoá lại kết luận gauge:

1. `--draws 3` trở lên cho `pca__random` → std giữa các draw là null band. Nếu
   +0.69 không vượt band thì nhân tố orientation là null trên cặp này.
2. `--seeds 1 2 3`. ICLR sample-reject review dẫn thẳng "results not statistically
   significant".
3. Chạy cặp `qwen3_4b_to_bert_base`: PR ở đó được dự đoán 38–52, tức là chỗ duy
   nhất gauge có nhiều chiều thật để khớp. Nếu gauge có tác dụng ở đâu thì là ở đó.

| so sánh | claim | trạng thái |
|---|---|---|
| ours vs `pca__random` | gauge Procrustes mang thông tin | +0.69, **chưa đủ bằng chứng** (1 draw) |
| ours vs `pca__none` | gauge có tác dụng | +0.45, **chưa đủ bằng chứng** (1 seed) |
| `pca__none` vs `random__none` | subspace phổ mang tín hiệu | đang chạy |
| ours vs `learned_*` | map đóng băng thắng map thích ứng | chưa chạy |

Với `learned_*`, thêm một điều kiện thứ ba trước khi đọc thành "map học thua":
`learned_projector_lr_scale` phải được quét ít nhất một lần (ví dụ `1.0` và `5.0`).
Một baseline chỉ chạy ở một lr là baseline chưa được tune.

### 6.3 Dự đoán ghi trước khi chạy, và nó ra sao

| dự đoán | kết quả |
|---|---|
| MiniLM: `procrustes ≈ none ≈ random` vì PR ≈ 1 | **PR đo được 1.37/384** (memo cũ ước 1.05 trên 4k câu CPU; đây là fit thật trên 16,384 câu). Spread 0.69 điểm — nhỏ, đúng hướng dự đoán, nhưng chưa phân định được với nhiễu |
| `pca__none` ≫ `random__none` | `explained_energy` **91.8% vs 38.1%** đúng như dự đoán; điểm downstream đang chạy |
| BERT-base: `procrustes > random` (PR 38–52) | chưa chạy — đây là cặp quyết định của C3 |
| `learned_t2s` thắng train loss, thua downstream | chưa chạy; cần log `loss_end` cuối cạnh AVG để dựng figure dissociation |

### 6.4 Số cần lấy kèm

Ngoài AVG, ba thứ này phải lấy trong cùng lần chạy vì không dựng lại được sau:

- **train `loss_end` cuối** của từng cell — trục thứ hai của figure dissociation.
  Đã có cho 3 cell (bảng §6.1).
- **`participation_ratio`** mỗi cặp (chỉ có ở dòng có gauge).
- **`explained_energy`** mỗi arm subspace.

Sinh lại bảng:

```bash
python3 scripts/run_target_map_ablation.py --pair <pair> --execute
python3 scripts/run_target_map_ablation.py --pair <pair> --collect
```

---

## 7. Cơ chế: file nào ghi gì

- Mọi run ghi `metrics.jsonl` (record cuối = final test, nhận ra bằng chỗ nó
  *thiếu* block `train`) và `teacher_projection.pt` (map đã fit, `explained_energy`,
  `gauge_stats`, arm đã dùng).
- `--collect` gộp thành `target_map_ablation.csv` + bảng text; cell 10 của notebook
  vẽ thêm bar chart có error bar theo `--draws`.
- Corpus mặc định `train_100k` **chưa dedup** SICK/STS-B. Nhiễm benchmark làm mọi
  arm cao lên như nhau nên không lệch phép so *trong* lưới, nhưng làm số tuyệt đối
  cao lên — đừng trích số của lưới ra ngoài bối cảnh ablation.
- Teacher chỉ chạy một lần cho mỗi (teacher, pooling, corpus): `--cache_dir` đặt
  cache ngoài thư mục run, nên lưới thứ hai và mọi lần chạy lại đều không phải
  encode lại corpus.
