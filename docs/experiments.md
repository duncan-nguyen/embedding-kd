Mình sẽ chốt thành **4 ablation tables**, tất cả chạy trên config (c), và giữ main results như hiện tại.

### Table A1 — Dimension reduction × coordinate selection

| Dimension reduction | Retained energy ↑ | As reduced |  Haar | **Student-selected** |
| ------------------- | ----------------: | ---------: | ----: | -------------------: |
| PCA-64              |             0.593 |        ... |   ... |                72.57 |
| PCA-128             |               ... |        ... |   ... |                73.77 |
| PCA-256             |               ... |        ... |   ... |                74.21 |
| PCA-384             |             0.928 |      69.35 | 70.37 |            **74.28** |
| Random subspace #1  |             0.381 |        ... |   ... |                74.45 |
| Random subspace #2  |             0.376 |        ... |   ... |                74.66 |
| Random subspace #3  |             0.374 |        ... |   ... |                74.46 |
| Uncentered SVD      |             0.945 |        ... |   ... |                74.60 |

**Mục đích:** chứng minh dimension reduction và coordinate selection là hai quyết định riêng; Haar cho thấy arbitrary rotation không đủ.

---

### Table A2 — What information should coordinate selection use?

| Selection signal                  | Retained energy ↑ |    Avg. ↑ |
| --------------------------------- | ----------------: | --------: |
| Teacher frame / no selection      |             0.928 |     69.35 |
| Haar random                       |             0.928 |     70.37 |
| Shuffled student correspondence   |             0.928 |       ... |
| Random / unrelated student signal |             0.928 |       ... |
| **Matched pretrained student**    |         **0.928** | **74.28** |

**Mục đích:** chứng minh chữ **student-conditioned** thật sự cần thiết, không phải chỉ do có thêm một orthogonal transform.

---

### Table A3 — Global vs local representative selection

| Representative strategy         | Retained energy ↑ |    Avg. ↑ |
| ------------------------------- | ----------------: | --------: |
| Teacher frame / no selection    |             0.928 |     69.35 |
| Haar fixed representative       |             0.928 |     70.37 |
| Per-minibatch selection         |             0.928 |       ... |
| Global selection, initial only  |             0.928 |     74.63 |
| Global selection + one refresh  |             0.928 |     74.64 |
| **Global epoch-wise selection** |         **0.928** | **74.79** |

**Mục đích:** justify design choice **corpus-level representative**, đồng thời cho thấy phần lớn gain đến từ initial selection; refresh chỉ là refinement.

---

### Table A4 — Component ablation of GATE-KD

| Endpoint target               | Structural support               | Retained energy ↑ |    Avg. ↑ |
| ----------------------------- | -------------------------------- | ----------------: | --------: |
| None                          | None                             |                 — |     52.62 |
| Teacher-frame endpoint        | None                             |             0.928 |     69.35 |
| **Student-selected endpoint** | None                             |             0.928 |     74.28 |
| None                          | \(H_0\)                          |                 — |       ... |
| Student-selected endpoint     | Gram / invariant structural loss |             0.928 |       ... |
| Student-selected endpoint     | NN-distance structural loss      |             0.928 |       ... |
| **Student-selected endpoint** | **\(H_0\)**                      |         **0.928** | **74.79** |

**Mục đích:** chứng minh core gain đến từ coordinate-selected endpoint; structural term chỉ là complementary support.

---

### Sensitivity S1 — Structural weight \(\lambda_{H_0}\)

| \(\lambda_{H_0}\) |    Avg. ↑ |
| ----------------: | --------: |
|                 0 |  existing |
|               0.1 |       ... |
|               0.3 |       ... |
|               0.5 | **74.79** |
|               1.0 |       ... |

Đây là sensitivity quan trọng nhất vì chứng minh result không phụ thuộc tuning structural term. Current paper đã có claim rằng across các giá trị sweep, score vẫn trên strongest baseline, kể cả \(\lambda_{H_0}=0\). 

---

### Sensitivity S2 — Representative fit-set size

| Fit-set size |    Avg. ↑ |
| -----------: | --------: |
|        2,048 |     74.51 |
|        4,096 |     74.67 |
|        8,192 |     74.61 |
|       14,760 | **74.79** |

Cái này giữ nguyên rất tốt. Nó cho thấy coordinate selection không cần estimate trên toàn corpus mới hoạt động. 
