# GeoODE-KD

Cho teacher đóng băng $f_T:\mathcal{X}\rightarrow\mathbb{R}^{d_T}$ và student $f_S:\mathcal{X}\rightarrow\mathbb{R}^{d_S}$, với $d_T>d_S$. GeoODE-KD chuyển embedding teacher về không gian student bằng một ánh xạ cố định, sau đó distill trực tiếp tại layer cuối.

## 1. Tạo target teacher

Teacher được chạy một lần trên toàn bộ tập train và embedding câu cuối được chuẩn hóa, cache lại:

\[
\tilde{t}_i=\operatorname{norm}(f_T(x_i)).
\]

Từ cache, lấy $d_S$ principal directions bằng PCA để tạo $P_{\mathrm{PCA}}\in\mathbb{R}^{d_T\times d_S}$. PCA được fit trên dữ liệu đã trừ mean; theo cấu hình mặc định, mean không bị trừ khi áp dụng phép chiếu.

Do hướng của PCA là tùy ý, GeoODE-KD căn chỉnh target với student chưa train bằng Orthogonal Procrustes. Với $T_{\mathrm{PCA}}$ là target PCA đã chuẩn hóa và $Z_0$ là embedding ban đầu của student trên cùng một tập mẫu:

\[
R^*=\arg\min_{R\in O(d_S)}\|T_{\mathrm{PCA}}R-Z_0\|_F,
\qquad
R^*=UV^\top,
\]

trong đó $U\Sigma V^\top=\operatorname{SVD}(T_{\mathrm{PCA}}^\top Z_0)$. Target cuối cùng là

\[
\tau_i=\operatorname{norm}(\tilde{t}_iP_{\mathrm{PCA}}R^*).
\]

$P_{\mathrm{PCA}}$, $R^*$ và toàn bộ $\tau_i$ được đóng băng trong suốt quá trình train.

## 2. Objective

GeoODE-KD chỉ giám sát embedding ở Transformer layer cuối. Mặc định student dùng CLS pooling và chuẩn hóa L2:

\[
z_i=\operatorname{norm}(\operatorname{Pool}(h_i^{(L)})).
\]

Loss gồm hai thành phần:

\[
\mathcal{L}_{\mathrm{end}}
=\frac{1}{B}\sum_i\left(1-z_i^\top\tau_i\right),
\]

\[
\mathcal{L}
=\lambda_{\mathrm{end}}\mathcal{L}_{\mathrm{end}}
+\lambda_{\mathrm{ctr}}\mathcal{L}_{\mathrm{InfoNCE}}(Z,Z'),
\]

trong đó $Z'$ là lần encode thứ hai của cùng câu với dropout độc lập. Cấu hình mặc định dùng $\lambda_{\mathrm{end}}=1$, $\lambda_{\mathrm{ctr}}=0.5$ và temperature $0.05$.

PCA giữ lại cấu trúc chính của teacher, Procrustes chỉ sửa hệ tọa độ mà không làm đổi pairwise cosine, còn InfoNCE giúp duy trì tính phân biệt và hạn chế collapse. Method chính không có learned projector, intermediate-layer loss, Gram loss hay topological loss; khi inference chỉ giữ nguyên student encoder.
