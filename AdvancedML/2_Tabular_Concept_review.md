**Ôn Tập - Góc Nhìn Hình Học và Thống Kê về Dữ Liệu**

**I. Giới Thiệu và Mục Tiêu**

*   Bài giảng này nhằm mục đích ôn tập và cung cấp một cách nhìn sâu hơn về cấu trúc dữ liệu dạng bảng (tabular data) thông qua các khái niệm toán học và hình học.
*   Nội dung chủ yếu dựa trên tài liệu "All Models Are Wrong: Concepts of Statistical Learning" của Gaston Sanchez và Ethan Marzban, đặc biệt là phần về "tính đối ngẫu" (duality) của ma trận dữ liệu.

**II. Biểu Diễn Dữ Liệu: Ma Trận Dữ Liệu**

1.  **Giả định:** Dữ liệu của chúng ta thường được tổ chức dưới dạng bảng.
2.  **Đối tượng toán học:** Dữ liệu này có thể được biểu diễn bằng một **ma trận (matrix)**, ký hiệu là `X`.
3.  **Kích thước:** Ma trận `X` có kích thước `n x p`, trong đó:
    *   `n`: Số lượng **hàng (rows)**, tương ứng với số lượng **mẫu (data items), cá thể (individuals), hoặc đối tượng (objects)**.
    *   `p`: Số lượng **cột (columns)**, tương ứng với số lượng **biến số (variables) hoặc đặc trưng (features)** được quan sát/đo lường trên mỗi cá thể.
4.  **Phần tử:** `x_ij` là giá trị của **biến thứ j** được quan sát trên **cá thể thứ i**.

**III. Tính Đối Ngẫu (Duality) của Ma Trận Dữ Liệu**

Ma trận dữ liệu `X` có thể được nhìn nhận theo hai cách cơ bản:

1.  **Nhìn theo hàng (Rows Perspective):** Tập trung vào các cá thể/mẫu.
2.  **Nhìn theo cột (Columns Perspective):** Tập trung vào các biến/đặc trưng.

Hai cách nhìn này dẫn đến hai không gian hình học khác nhau nhưng bổ sung cho nhau.

**IV. Góc Nhìn 1: Đám Mây Cá Thể (Cloud of Individuals) - Không Gian Hàng (Row Space)**

1.  **Không gian:** Không gian `p` chiều (p-dimensional space), với mỗi trục tương ứng với một biến/đặc trưng.
2.  **Biểu diễn:** Mỗi **hàng** của ma trận `X` (đại diện cho một cá thể) được xem là một **điểm (point)** trong không gian `p` chiều này.
3.  **Tổng thể:** Toàn bộ `n` cá thể tạo thành một **đám mây điểm (cloud of points)** trong không gian `p` chiều.
4.  **Các Phép Toán và Thống Kê trên Cá Thể:**
    *   **Cá Thể Trung Bình (Average Individual / Centroid - g):**
        *   Là điểm "trung tâm" của đám mây điểm.
        *   Tọa độ của điểm `g` là trung bình cộng của tất cả các giá trị trên từng cột (từng biến): `g = (mean(col1), mean(col2), ..., mean(colp))`.
        *   Trong không gian 1 chiều (chỉ có 1 biến), nó là điểm cân bằng (balancing point).
        *   Còn gọi là tâm (centroid), tâm tỷ cự (barycenter), hoặc trọng tâm (center of gravity).
    *   **Dữ Liệu Được Tâm Hóa (Centered Data):**
        *   Là phép biến đổi dữ liệu bằng cách trừ đi tọa độ của centroid `g` khỏi mỗi điểm (mỗi hàng). `x_i_centered = x_i - g`.
        *   Về mặt hình học, đây là phép **dịch chuyển trục tọa độ** sao cho gốc tọa độ mới trùng với tâm `g` của đám mây điểm.
    *   **Khoảng Cách Giữa Các Cá Thể (Distance between Individuals):**
        *   Thường dùng **khoảng cách Euclidean (bình phương)** giữa hai điểm `i` và `l`: `d²(i, l) = Σ (x_ij - x_lj)²` (tổng trên tất cả các biến `j`). Có thể viết dưới dạng vector: `d²(i, l) = (x_i - x_l)^T (x_i - x_l)`.
    *   **Độ Phân Tán Tổng Thể (Overall Dispersion):**
        *   Đo mức độ "lan rộng" hay "phân tán" (spread/scatter) của đám mây điểm.
        *   Tính bằng tổng bình phương khoảng cách giữa tất cả các cặp điểm (kể cả điểm với chính nó): `Σ Σ d²(i, l)`.
    *   **Quán Tính (Inertia):**
        *   Một thước đo khác của độ phân tán, tính bằng **trung bình bình phương khoảng cách** từ mỗi điểm đến tâm `g`: `(1/n) * Σ d²(i, g) = (1/n) * Σ (x_i - g)^T (x_i - g)`.
        *   Trong trường hợp 1 chiều, quán tính chính là **phương sai** của biến đó.

**V. Góc Nhìn 2: Đám Mây Biến Số (Cloud of Variables) - Không Gian Cột (Column Space)**

1.  **Không gian:** Không gian `n` chiều (n-dimensional space), với mỗi trục tương ứng với một cá thể/mẫu.
2.  **Biểu diễn:** Mỗi **cột** của ma trận `X` (đại diện cho một biến/đặc trưng) được xem là một **vector (vector)** trong không gian `n` chiều này, xuất phát từ gốc tọa độ.
3.  **Biến đổi:** Trong tiền xử lý, chúng ta có thể thay đổi độ lớn (scale - co giãn) của các vector biến nhưng thường cố gắng không thay đổi hướng (direction) của chúng.
4.  **Các Phép Toán và Thống Kê trên Biến:**
    *   **Trung Bình của Biến (Mean of a Variable - x̄):**
        *   Là trung bình cộng của tất cả các giá trị trong một vector cột (giá trị của biến đó trên tất cả các cá thể): `x̄ = (1/n) * Σ x_i`.
        *   Có thể tổng quát hóa thành **trung bình có trọng số (weighted average)** nếu các cá thể có trọng số `w_i` khác nhau: `x̄_weighted = Σ w_i * x_i`.
    *   **Phương Sai của Biến (Variance - Var(X)):**
        *   Đo lường độ phân tán của các giá trị của biến đó quanh giá trị trung bình: `Var(X) = (1/n) * Σ (x_i - x̄)²`.
        *   Có thể viết dưới dạng vector (nếu `x` là vector cột): `Var(x) = (1/n) * (x - x̄*1)^T (x - x̄*1)`. Nếu `x` đã được tâm hóa (mean-centered), `Var(x) = (1/n) * x^T x = (1/n) * ||x||²`.
    *   **Độ Lệch Chuẩn (Standard Deviation - sd(X)):**
        *   Là căn bậc hai của phương sai, `sd(X) = sqrt(Var(X))`. Nó có cùng đơn vị với biến gốc.
    *   **Hiệp Phương Sai (Covariance - cov(x, y)):**
        *   Tổng quát hóa khái niệm phương sai cho hai biến `x` và `y`. Đo lường mức độ hai biến cùng biến thiên (đồng biến hoặc nghịch biến).
        *   `cov(x, y) = (1/n) * Σ (x_i - x̄)(y_i - ȳ)`.
        *   Nếu `x`, `y` đã được tâm hóa: `cov(x, y) = (1/n) * x^T y`.
    *   **Tương Quan (Correlation - cor(x, y)):**
        *   Là hiệp phương sai được **chuẩn hóa** để loại bỏ ảnh hưởng của đơn vị đo, cho biết **hướng** và **độ mạnh của mối quan hệ tuyến tính** giữa hai biến.
        *   `cor(x, y) = cov(x, y) / (sd(x) * sd(y))`.
        *   Giá trị nằm trong khoảng [-1, 1].
        *   Nếu `x`, `y` đã được tâm hóa: `cor(x, y) = (x^T y) / (||x|| * ||y||)`.
        *   Nếu `x`, `y` đã được **chuẩn hóa** (tâm hóa và chia cho độ lệch chuẩn, tức là có mean=0, sd=1): `cor(x, y) = (1/n) * x^T y`. (Lưu ý: có thể thiếu 1/n tùy định nghĩa chuẩn hóa). *Slide gốc ghi `cor(x, y) = x^T y` khi chuẩn hóa, điều này đúng nếu chuẩn hóa không bao gồm chia cho `sqrt(n)` trong độ lệch chuẩn mẫu.*
    *   **Ý Nghĩa Hình Học của Tương Quan:** Đối với các vector biến đã được tâm hóa `x` và `y`, **hệ số tương quan chính là cosin của góc (θ_xy) giữa hai vector đó**: `cor(x, y) = cos(θ_xy)`.
        *   `cor = 1`: Góc 0 độ (cùng hướng).
        *   `cor = -1`: Góc 180 độ (ngược hướng).
        *   `cor = 0`: Góc 90 độ (trực giao - không có quan hệ tuyến tính).
    *   **Phép Chiếu Vuông Góc (Orthogonal Projection):**
        *   Mục tiêu: Xấp xỉ vector biến `y` bằng một bội số của vector biến `x`, tức là tìm `ŷ = bx` sao cho `ŷ` gần `y` nhất (khoảng cách `||y - ŷ||²` nhỏ nhất).
        *   Giải pháp: Chiếu vuông góc vector `y` lên vector `x`. Vector chiếu `ŷ` được tính bằng: `ŷ = x * (x^T y / ||x||²)`. Hệ số `b = x^T y / ||x||²`.
    *   **Trung Bình như một Phép Chiếu Vuông Góc:**
        *   Xét vector biến `x` và vector hằng số `1 = (1, 1, ..., 1)`.
        *   Phép chiếu vuông góc của `x` lên `1` là: `x̂ = 1 * (1^T x / ||1||²) = 1 * (Σ x_i / n) = x̄ * 1`.
        *   Như vậy, **giá trị trung bình (x̄)** của biến `x` chính là **hệ số tỉ lệ** khi chiếu vector `x` lên vector hằng số `1`.

**VI. Kết Luận**

Việc hiểu dữ liệu dưới dạng ma trận và hai góc nhìn hình học (đám mây cá thể trong không gian đặc trưng và đám mây biến số trong không gian cá thể) cung cấp nền tảng quan trọng để hiểu sâu hơn về các phương pháp phân tích dữ liệu và học máy (ví dụ: PCA, hồi quy, phân cụm,...). Các khái niệm như khoảng cách, tâm, phương sai, tương quan, phép chiếu đều có những diễn giải hình học trực quan trong các không gian này.