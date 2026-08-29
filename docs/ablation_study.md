# Ablation study

Danh sách mọi ablation gọi được bằng flag trong repo, kèm câu hỏi mỗi cái trả lời
và ý nghĩa của từng kết quả. Cột **default** là giá trị chạy khi không truyền gì —
tức là recipe, còn mọi dòng khác là ablation.

Objective mặc định của `geoode` là **`L_end + L_ctr`** (`λ_end = 1`, `λ_ctr = 0.5`,
`λ_vel = λ_desc = 0`), khớp với `docs/latex_iclr/sections/experiments.tex` và caption
`docs/latex_iclr/tables/main_results.tex`.

Entry point:

| | |
|---|---|
| Lưới target map | `scripts/run_target_map_ablation.py` (dry-run → `--execute` → `--collect`) |
| Trong notebook | `test_mdd.ipynb` cell 9–11, bật bằng `RUN_ABLATION` ở cell 1 |
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

`learned_projector_lr_scale` (config, mặc định `1.0`) đặt lr của `W` theo bội số lr
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

`full` = mọi tổ hợp subspace × gauge = 17 ô (5 × 3 cộng 2 arm learned không có cột
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

### 2.1 Đổi gradient

| flag | default | arm | hỏi gì |
|---|---|---|---|
| `--lambda_vel` | `0.0` | `> 0` | bật `L_evel`: giám sát *hướng* của từng transition layer thay vì chỉ ghim layer cuối. Đã đo trên (student 22M, `train_100k`): nằm trong khoảng nhiễu của `λ_vel = 0` — đó là lý do default là 0 |
| `--lambda_desc` | `0.0` | `> 0` | thêm ràng buộc giảm yếu: nửa sâu của các transition bị phạt chỉ khi *làm tăng* `E_sem`. Chạy độc lập được, nhưng nó được thiết kế để đi kèm `λ_vel > 0` — nó vá đúng chỗ `L_evel` bỏ trống (chỉ ràng buộc hướng, không ràng buộc độ lớn) |
| `--lambda_ctr` | `0.5` | `0` / khác | `0` bỏ InfoNCE → đo phần tín hiệu thuần từ teacher. Đây là weight duy nhất được tune |
| `--lambda_end` | `1.0` | `0` | bỏ endpoint → chỉ còn regulariser, về thực chất là `simcse` chạy trong khung geoode |

`loss_vel` và `loss_desc` **luôn được tính và log** kể cả ở weight 0, nên một run
default vẫn báo cáo hai term đó đo được gì.

### 2.2 Chỉ đổi diagnostic, KHÔNG đổi gradient

Các flag dưới đây trông như ablation của objective nhưng trong code hiện tại
**không có đường nào tới loss**. Chạy chúng ở objective mặc định sẽ cho kết quả
downstream *giống hệt* — đừng đốt GPU vào đó rồi đọc thành "ablation null".

| flag | default | thực sự ảnh hưởng cái gì |
|---|---|---|
| `--alpha` / `--beta` | `1.0` / `1.0` | chỉ vào `energy()` và `vector_field()`, mà hai hàm này chỉ được gọi trong `depth_report` và trong khối `no_grad` ghi `energy_first`/`energy_last`. Không term loss nào dùng `alpha`/`beta`: `L_end` là cosine trần, `L_evel` chỉ dùng log map, `L_desc-sem` dùng `E_sem` chưa nhân trọng số |
| `--guidance_schedule` / `--guidance_power` | `linear` / `1.0` | `s(t)` chỉ vào `step_fraction` → `euler_step` và `depth_report`. `L_evel` so hướng transition với geodesic tới teacher, không dùng lịch độ sâu |
| `--relational_target` | `native` | quyết định Gram nào được truyền làm `teacher_gram`, nhưng `velocity_loss` mở đầu bằng `del teacher_gram` (hướng mục tiêu là instance-wise) và chỗ dùng còn lại là khối `no_grad` diagnostic |

Nói cách khác, `E_geo` hiện là **đại lượng đo**, không phải thành phần được tối
ưu. Muốn `E_geo` thành ablation thật thì phải thêm một loss term đọc nó — đấy là
việc viết code, không phải việc chạy flag.

Hai field config `stop_grad_target` và `include_embedding_layer` (§3) cũng nằm
trong nhóm này khi chạy ở objective mặc định: cả hai chỉ vào `L_evel`/`L_desc-sem`.

Bất biến này được chốt bằng test
`test_geoode_kd.py::test_the_default_objective_ignores_the_flow_hyperparameters` —
nếu sau này có loss term đọc `E_geo` hay lịch `s(t)` thì test đỏ và bảng này phải
sửa theo.

---

## 3. Chỉ chỉnh được trong config (chưa có CLI flag)

Sửa ở `config/geoode_config.py` hoặc truyền qua `GeoODEConfig(**kwargs)`.

| field | default | hỏi gì |
|---|---|---|
| `contrastive_view` | `dropout` | `pair` dùng câu thứ hai của cặp làm view thứ hai thay vì hai dropout mask |
| `contrastive_temperature` | `0.05` | `τ_c` của InfoNCE |
| `gauge_align_samples` | `16384` | số câu fit Procrustes; phải `>> d_S` để cross-covariance đủ điều kiện |
| `stop_grad_target` | `True` | tắt = ablation "full gradient dynamics": bỏ `sg[·]` khỏi hướng mục tiêu và khỏi năng lượng layer trước, để hai term đó có thể bị hạ bằng cách làm hỏng chính state chúng đo. **Chỉ có tác dụng khi `λ_vel > 0` hoặc `λ_desc > 0`** |
| `include_embedding_layer` | `False` | bật = coi output embedding là state độ sâu 0. Cũng **chỉ có tác dụng khi `λ_vel > 0` hoặc `λ_desc > 0`**: `L_end` chỉ đọc `states[-1]` |

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
| PCA + MSE | recipe sentence-transformers ≤ v5.4: cùng target map với `pca__none` nhưng loss MSE thay vì cosine | — |

(learned s→t và learned t→s đã có code, xem §1.1.)

Cùng với hai arm learned, đây là bằng chứng cho luận điểm "mọi độ tự do thích ứng
của target map bị dùng để giảm loss thay vì dạy student": khi map thích ứng, train
loss giảm mà downstream giảm theo.

---

## 6. Đọc kết quả

- Mọi run ghi `metrics.jsonl` (record cuối = final test, nhận ra bằng chỗ nó
  *thiếu* block `train`) và `teacher_projection.pt` (map đã fit, `explained_energy`,
  `gauge_stats`, arm đã dùng).
- `--collect` gộp thành `target_map_ablation.csv` + bảng text; cell 11 của notebook
  vẽ thêm bar chart có error bar theo `--draws`.
- Corpus mặc định `train_100k` **chưa dedup** SICK/STS-B. Nhiễm benchmark làm mọi
  arm cao lên như nhau nên không lệch phép so *trong* lưới, nhưng làm số tuyệt đối
  cao lên — đừng trích số của lưới ra ngoài bối cảnh ablation.
