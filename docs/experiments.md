Mình sẽ chốt thành **4 ablation tables**, tất cả chạy trên config (c), và giữ main results như hiện tại.

Các giá trị `Avg.` bên dưới là **mean ± sample std** trên ba seed 42, 43, 44, ở thang điểm phần trăm. Retained energy và các thống kê được lấy từ các bảng tổng hợp trong `runs/`; chữ đậm đánh dấu cấu hình/phương án được nhấn mạnh trong thiết kế, không nhất thiết là giá trị lớn nhất của từng cột.

### Table A1 — Dimension reduction × coordinate selection

| Reduction        | Retained energy ↑ | As reduced     | Haar-random    | **Student-conditioned** |
| ---------------- | ----------------: | -------------: | -------------: | -------------------------: |
| PCA-64           |             0.593 |   71.49 ± 0.08 |   72.39 ± 0.13 |               73.70 ± 0.10 |
| PCA-128          |             0.739 |   72.22 ± 0.04 |   73.03 ± 0.22 |               74.52 ± 0.01 |
| PCA-256          |             0.873 |   72.48 ± 0.01 |   73.08 ± 0.30 |               74.79 ± 0.03 |
| PCA-384          |             0.928 |   72.39 ± 0.03 |   73.01 ± 0.27 |           **74.86 ± 0.06** |
| Random subspace (1) |          0.381 |   73.10 ± 0.08 |   72.79 ± 0.32 |               74.50 ± 0.03 |
| Random subspace (2) |          0.376 |   72.92 ± 0.11 |   72.85 ± 0.33 |               74.64 ± 0.04 |
| Random subspace (3) |          0.374 |   72.65 ± 0.11 |   72.85 ± 0.16 |               74.47 ± 0.07 |
| SVD (uncentered) |             0.945 |   72.84 ± 0.07 |   73.21 ± 0.15 |               74.86 ± 0.03 |

**Mục đích:** chứng minh dimension reduction và coordinate selection là hai quyết định riêng; Haar cho thấy arbitrary rotation không đủ.

---

### Table A2 — What information should coordinate selection use?

| Selection signal       |          Avg. ↑ |
| ---------------------- | --------------: |
| As reduced             |    72.39 ± 0.03 |
| Haar-random            |    73.01 ± 0.27 |
| Shuffled correspondence|    74.01 ± 0.21 |
| Unrelated student      |    73.15 ± 0.20 |
| **Matched student**    | **74.86 ± 0.06**|

**Mục đích:** chứng minh chữ **student-conditioned** thật sự cần thiết, không phải chỉ do có thêm một orthogonal transform.

---

### Table A3 — Global vs local representative selection

| Representative strategy    |           Avg. ↑ |
| -------------------------- |  --------------: |
| As reduced                 |     72.39 ± 0.03 |
| Haar-random                |     73.01 ± 0.27 |
| Per-batch selection        |     72.71 ± 0.07 |
| Initial selection only     |     74.61 ± 0.04 |
| One refresh                |     74.68 ± 0.01 |
| **Epoch-wise selection**   | **74.86 ± 0.06** |

**Mục đích:** justify design choice **corpus-level representative**, đồng thời cho thấy phần lớn gain đến từ initial selection; refresh chỉ là refinement.

---

### Table A4 — Component ablation of GATE-KD

| Endpoint target                 | Structural objective |        Avg. ↑ |
| ------------------------------- | ------------------ | --------------: |
| No endpoint                     | None                 |    52.66 ± 0.02 |
| PCA-reduced                     | None                 |    69.29 ± 0.07 |
| **Student-conditioned**         | None                 |    74.25 ± 0.01 |
| No endpoint                     | \(H_0\) persistence  |    64.21 ± 0.48 |
| Student-conditioned             | Gram matching        |    73.84 ± 0.03 |
| Student-conditioned             | kNN-distance matching|    75.01 ± 0.02 |
| **Student-conditioned**         | **\(H_0\) persistence** |**74.86 ± 0.06** |

**Mục đích:** kiểm tra core gain từ student-conditioned endpoint và so sánh các structural objectives. Kết quả hiện tại cho thấy kNN-distance matching đạt điểm cao nhất, còn \(H_0\) persistence cải thiện endpoint-only nhưng không phải structural objective tốt nhất trong sweep này.

---

### Sensitivity S1 — Structural weight \(\lambda_{H_0}\)

| \(\lambda_{H_0}\) |          Avg. ↑ |
| ----------------: | --------------: |
|                 0 |    74.25 ± 0.01 |
|               0.1 |    74.54 ± 0.03 |
|               0.3 |    74.81 ± 0.05 |
|               0.5 |**74.86 ± 0.06** |
|               1.0 |    74.63 ± 0.06 |

Đây là sensitivity quan trọng nhất để đánh giá mức độ phụ thuộc vào tuning structural term. Trong sweep hiện tại, score nằm trong khoảng 74.25–74.86; \(\lambda_{H_0}=0.5\) cao nhất, còn bỏ \(H_0\) làm giảm 0.61 điểm.

---

### Sensitivity S2 — Representative fit-set size

| Fit-set size |          Avg. ↑ |
| -----------: | --------------: |
|        2,048 |    74.65 ± 0.05 |
|        4,096 |    74.78 ± 0.05 |
|        8,192 |    74.75 ± 0.01 |
|       14,760 |**74.86 ± 0.06** |

Các fit-set từ 2,048 đến 14,760 mẫu đều nằm trong khoảng 74.65–74.86; dùng 2,048 mẫu chỉ thấp hơn full corpus 0.21 điểm. Kết quả này cho thấy coordinate selection không cần estimate trên toàn corpus mới hoạt động tốt.
