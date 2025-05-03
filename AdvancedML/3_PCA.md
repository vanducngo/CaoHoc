**Phân Tích Thành Phần Chính (Principal Component Analysis - PCA)**

**I. Giới Thiệu và Mục Tiêu**

1.  **Bối cảnh:**
    *   Trong bài trước, chúng ta đã thấy dữ liệu có thể được biểu diễn như một đám mây điểm (các cá thể) trong không gian `p` chiều (không gian đặc trưng) hoặc một đám mây vector (các biến) trong không gian `n` chiều (không gian cá thể).
    *   Bài giảng này tập trung vào góc nhìn **đám mây điểm trong không gian `p` chiều**.
2.  **Vấn đề:** Dữ liệu thực tế thường có rất nhiều chiều (`p` lớn), gây khó khăn cho việc:
    *   Trực quan hóa (khó hình dung không gian > 3 chiều).
    *   Phân tích và xây dựng mô hình (nguyền rủa số chiều - curse of dimensionality).
    *   Lưu trữ và tính toán.
3.  **Câu hỏi:** Liệu có cách nào để tạo ra một **biểu diễn dữ liệu ở số chiều thấp hơn (low-dimensional representation)** mà vẫn giữ lại được nhiều thông tin quan trọng nhất của dữ liệu gốc không?
4.  **Mục tiêu của PCA:** Tìm một không gian con (subspace) có số chiều thấp hơn (`k < p`) và chiếu (project) dữ liệu gốc lên không gian con này sao cho **mất mát thông tin là ít nhất**.

**II. Ý Tưởng Cốt Lõi: Phép Chiếu (Projections)**

1.  **Analogy (Phép loại suy):** Giống như việc chụp ảnh một vật thể 3D (cái cốc). Các góc chụp khác nhau (các phép chiếu khác nhau lên mặt phẳng 2D) sẽ cho ra những hình ảnh khác nhau.
    *   Một số góc chụp (phép chiếu) sẽ giữ lại được nhiều đặc điểm nhận dạng của vật thể hơn các góc khác.
    *   Hình ảnh 2D luôn có sự **mất mát thông tin** so với vật thể 3D gốc.
2.  **Tìm phép chiếu tốt nhất:** Chúng ta muốn tìm một phép chiếu (một không gian con) sao cho "hình ảnh" (dữ liệu sau khi chiếu) **giống nhất có thể** với "vật thể gốc" (dữ liệu gốc). Điều này đồng nghĩa với việc **giảm thiểu sự mất mát thông tin**.
3.  **Không gian con đơn giản nhất:** Bắt đầu với việc tìm một không gian con **1 chiều (1D)**, tức là một **đường thẳng (trục - axis)** đi qua "tâm" của đám mây dữ liệu.
    *   Ký hiệu trục này là `dim_v`. Hướng của trục được xác định bởi một **vector chỉ phương `v`**.
    *   Mục tiêu là **chiếu vuông góc** tất cả các điểm dữ liệu (cá thể) lên trục `dim_v` này.

**III. Toán Học Hóa Phép Chiếu**

1.  **Chuẩn bị:** Để đơn giản, ta **tâm hóa dữ liệu (mean-center)**, tức là dịch chuyển đám mây điểm sao cho tâm (centroid `g`) của nó trùng với gốc tọa độ (origin). Khi đó, trục `dim_v` sẽ đi qua gốc tọa độ.
2.  **Phép chiếu điểm `x_i` lên trục `v`:** (Giả sử `x_i` là vector từ gốc tọa độ đến điểm thứ i sau khi đã tâm hóa)
    *   **Vector chiếu (Vector projection):** Là hình chiếu vuông góc của vector `x_i` lên đường thẳng chứa `v`. Kết quả là một vector cùng phương với `v`. Công thức: `v̂ = (v^T x_i / v^T v) * v`.
    *   **Giá trị chiếu (Scalar projection - `z_ik`):** Là **tọa độ** của điểm chiếu trên trục `dim_v` (nếu coi `v` là vector đơn vị xác định trục). Nó cho biết "độ dài" của hình chiếu theo hướng `v`. Công thức: `z_ik = v^T x_i / ||v||`. Đây chính là giá trị của "thành phần chính" tương ứng với điểm `x_i`.

**IV. Tiêu Chuẩn Chọn Phép Chiếu Tốt Nhất: Maximizing Variance (Tối Đa Hóa Phương Sai)**

1.  **Liên hệ giữa khoảng cách và phương sai:** Việc giữ cho khoảng cách giữa các điểm sau khi chiếu (`d_H²(i, l)`) "tương tự" nhất với khoảng cách gốc (`d²(i, l)`) tương đương với việc **tối đa hóa Quán tính (Inertia)** của các điểm đã chiếu.
    *   `Inertia = (1/n) * Σ d²(i, g)` (Trung bình bình phương khoảng cách từ điểm đến tâm).
    *   `Overall Dispersion = 2n² * Inertia`.
    *   Do đó, mục tiêu là: `max_H { (1/n) Σ d_H²(i, g) }`, trong đó `H` là không gian con để chiếu.
2.  **Trường hợp chiếu lên trục 1D (vector `v`):**
    *   Quán tính của các điểm chiếu lên trục `v` (với dữ liệu đã tâm hóa `g=0` và `v` là vector đơn vị `||v||=1`): `(1/n) Σ d_H²(i, 0) = (1/n) Σ (v^T x_i)² = (1/n) Σ z_i²`.
    *   Biểu thức này chính là **phương sai (Variance)** của các giá trị chiếu `z_i`.
3.  **Bài toán tối ưu:** Tìm vector đơn vị `v` sao cho phương sai của dữ liệu khi chiếu lên `v` là lớn nhất:
    *   `max_v { (1/n) Σ (x_i^T v)² }` với điều kiện `v^T v = 1`.
    *   Viết dưới dạng ma trận (với `X` là ma trận dữ liệu đã tâm hóa): `max_v { (1/n) v^T X^T X v }` với điều kiện `v^T v = 1`.

**V. Giải Pháp: Phân Tích Trị Riêng (Eigenvalue Decomposition)**

1.  **Sử dụng Nhân tử Lagrange:** Giải bài toán tối ưu có điều kiện ở trên.
2.  **Kết quả:** Vector `v` tối ưu phải là **vector riêng (eigenvector)** của ma trận `S = (1/n) X^T X`.
    *   `Sv = λv`.
    *   `S` chính là **ma trận hiệp phương sai (covariance matrix)** của dữ liệu `X` (nếu `X` đã tâm hóa). Nếu `X` được chuẩn hóa (mean=0, var=1), `S` là **ma trận tương quan (correlation matrix)**.
3.  **Ý nghĩa:**
    *   **Vector riêng `v` (Eigenvector):** Xác định **hướng (direction)** của trục mà khi chiếu dữ liệu lên đó, phương sai được tối đa hóa. Đây chính là các **trục chính (principal axes)**.
    *   **Giá trị riêng `λ` (Eigenvalue):** Chính là giá trị phương sai tối đa đạt được khi chiếu lên hướng `v` tương ứng. Nó đo lường **lượng thông tin (variance)** mà trục chính đó nắm giữ. `λ = Var(z)` với `z = Xv`.

**VI. Thành Phần Chính (Principal Components - PCs)**

1.  **Ma trận hiệp phương sai `S`:** Là ma trận đối xứng `p x p`, có `p` giá trị riêng thực (`λ_1 ≥ λ_2 ≥ ... ≥ λ_p ≥ 0`) và `p` vector riêng trực chuẩn tương ứng (`v_1, v_2, ..., v_p`).
    *   Các vector riêng tạo thành ma trận `V`.
    *   Các giá trị riêng tạo thành ma trận đường chéo `Λ`.
    *   Phân tích trị riêng của `S`: `S = V Λ V^T`.
2.  **Tổng phương sai:** Tổng các giá trị riêng bằng tổng phương sai của các biến ban đầu (sau khi tâm hóa): `Σ λ_k = trace(S) = Inertia`.
3.  **Định nghĩa PC:**
    *   **Thành phần chính thứ k (k-th PC), ký hiệu `z_k`:** Là một **biến mới** được tạo ra bằng cách chiếu dữ liệu gốc `X` lên **vector riêng thứ k (`v_k`)**: `z_k = X v_k`.
    *   `z_k` là một vector cột `n x 1`, chứa giá trị của PC thứ k cho tất cả `n` cá thể.
    *   `z_k` cũng là một **tổ hợp tuyến tính (linear combination)** của các biến gốc `x_j`: `z_k = v_1k*x_1 + v_2k*x_2 + ... + v_pk*x_p`. Các `v_jk` là thành phần của vector riêng `v_k`.
4.  **Ma trận các thành phần chính `Z`:**
    *   Nếu giữ lại tất cả `p` thành phần chính: `Z = [z_1, z_2, ..., z_p] = XV`.
    *   Nếu chỉ giữ lại `k` thành phần chính đầu tiên (tương ứng `k` giá trị riêng lớn nhất): `Z_k = X V_k`, trong đó `V_k = [v_1, ..., v_k]`.
5.  **Tính chất quan trọng:**
    *   `Var(z_k) = λ_k`: Phương sai của PC thứ `k` bằng giá trị riêng thứ `k`.
    *   Các thành phần chính `z_h` và `z_l` là **không tương quan (uncorrelated)** hay **trực giao (orthogonal)** với nhau nếu `h ≠ l` (`z_h^T z_l = 0`).

**VII. Quy Trình Thực Hiện PCA**

1.  **(Tùy chọn nhưng thường làm) Chuẩn hóa dữ liệu (Standardize):** Trừ đi trung bình (mean-centering) và chia cho độ lệch chuẩn của từng cột (biến). Điều này đảm bảo các biến có cùng thang đo và không bị ảnh hưởng bởi đơn vị đo. Nếu chuẩn hóa, PCA sẽ được thực hiện trên ma trận tương quan. Nếu chỉ tâm hóa, PCA thực hiện trên ma trận hiệp phương sai.
2.  **Tính ma trận `S`:** Tính ma trận hiệp phương sai (hoặc tương quan nếu đã chuẩn hóa) `S = (1/n) X^T X` (với `X` là ma trận dữ liệu đã xử lý ở bước 1).
3.  **Phân tích trị riêng/vector riêng:** Tìm các giá trị riêng `λ_k` và vector riêng `v_k` của ma trận `S`.
4.  **Sắp xếp:** Sắp xếp các giá trị riêng theo thứ tự giảm dần (`λ_1 ≥ λ_2 ≥ ...`) và sắp xếp các vector riêng tương ứng.
5.  **Chọn số chiều `k`:** Quyết định giữ lại bao nhiêu thành phần chính (`k`). Có thể dựa trên:
    *   Tỷ lệ phương sai giải thích được (ví dụ: giữ đủ PC để giải thích 90% tổng phương sai: `Σ(λ_i) / Σ(λ_j) ≥ 0.9`).
    *   Biểu đồ Scree plot (tìm điểm "khuỷu tay" - elbow point).
    *   Số chiều mong muốn cho mục đích cụ thể (ví dụ: `k=2` hoặc `k=3` để trực quan hóa).
6.  **Tạo ma trận `V_k`:** Chọn `k` vector riêng đầu tiên tương ứng với `k` giá trị riêng lớn nhất.
7.  **Chiếu dữ liệu:** Tính toán ma trận các thành phần chính mới: `Z = X V_k`. `Z` là ma trận dữ liệu `n x k` đã được giảm chiều.

**VIII. Mục Đích và Ứng Dụng**

*   **Giảm chiều dữ liệu:** Biến đổi dữ liệu từ `p` chiều xuống `k` chiều (`k < p`).
*   **Loại bỏ nhiễu:** Các PC cuối (ứng với `λ` nhỏ) thường chứa nhiều nhiễu hơn là thông tin cấu trúc.
*   **Trực quan hóa dữ liệu:** Chiếu dữ liệu xuống 2D hoặc 3D (chọn `k=2` hoặc `k=3`).
*   **Tiền xử lý dữ liệu:** Sử dụng các PC làm đầu vào cho các thuật toán học máy khác (giảm tương quan, giảm số chiều).
*   **Nén dữ liệu.**
*   **Nhận dạng khuôn mặt (Eigenfaces)** là một ứng dụng nổi tiếng.