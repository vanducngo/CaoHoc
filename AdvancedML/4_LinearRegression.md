Chắc chắn rồi, đây là nội dung tổng hợp và giải thích chi tiết bài giảng về Hồi Quy Tuyến Tính (Linear Regression) từ các slide:

**Bài Giảng 4: Hồi Quy Tuyến Tính (Linear Regression)**

**I. Động Lực và Mục Tiêu (Motivation)**

1.  **Bài toán:** Chúng ta muốn dự đoán một biến mục tiêu (biến phụ thuộc, response) `y` dựa trên một hoặc nhiều biến đầu vào (biến độc lập, features, predictors) `X`.
2.  **Ví dụ:** Dự đoán mức lương (salary `y`) của cầu thủ NBA dựa trên các thông tin như: đội (team), chiều cao (height), cân nặng (weight), vị trí (position), số năm kinh nghiệm (years of experience), số điểm 2pts, 3pts, blocks,... (các `x_j`).
3.  **Ký hiệu:**
    *   `x_i`: Vector chứa các thông số (features) của cầu thủ thứ `i`.
    *   `y_i`: Mức lương (response) của cầu thủ thứ `i`.
4.  **Giả định nền tảng:** Tồn tại một hàm **không xác định (unknown function)** `f` mô tả mối quan hệ "lý tưởng" giữa `X` và `y` (`f: X -> y`).
5.  **Mục tiêu của chúng ta:** Tìm một **mô hình (model)**, ký hiệu là `f̂` (`f̂: X -> y`), được chọn từ một tập các hàm ứng viên (ví dụ `h1, h2,...`), sao cho mô hình `f̂` này **xấp xỉ tốt nhất (best approximates)** hàm `f` không xác định kia.

**II. Trực Giác về Hồi Quy (Intuition of Regression)**

Ý tưởng cơ bản của hồi quy là sử dụng thông tin có sẵn (features) để đưa ra dự đoán tốt hơn cho biến mục tiêu.

1.  **Kịch bản 1: Không có thông tin đầu vào.**
    *   Làm thế nào để dự đoán lương `y0` cho một cầu thủ mới mà không biết gì về anh ta?
    *   **Dự đoán tốt nhất:** Sử dụng **giá trị trung bình lịch sử (historical average)** `ȳ` của tất cả các cầu thủ. Giá trị trung bình là một thước đo trung tâm, đại diện cho giá trị "điển hình". (Có thể dùng trung vị - median - để giảm ảnh hưởng của giá trị ngoại lai).
    *   `ŷ0 = ȳ`
2.  **Kịch bản 2: Biết cầu thủ sẽ vào đội Lakers.**
    *   Dự đoán có thể tốt hơn bằng cách chỉ sử dụng thông tin của những người cùng điều kiện.
    *   **Dự đoán tốt hơn:** Sử dụng **giá trị trung bình lương của riêng các cầu thủ Lakers**.
    *   `ŷ0 = avg(Laker's Salaries)`
3.  **Kịch bản 3: Biết cầu thủ có 6 năm kinh nghiệm.**
    *   **Dự đoán tốt hơn nữa:** Sử dụng **giá trị trung bình lương của những cầu thủ có cùng 6 năm kinh nghiệm**.
4.  **Tổng quát hóa:** Dự đoán trong hồi quy là một dạng **trung bình có điều kiện (conditional mean)**:
    *   `ŷ0 = avg(y_i | x_i = x_0)` (Trung bình của `y` với điều kiện `x` bằng giá trị cụ thể `x0`).
    *   **Vấn đề:** Cách này chỉ hoạt động nếu có đủ dữ liệu `x_i` trùng khớp chính xác với `x_0`.
    *   **Ý tưởng cốt lõi của hồi quy:** Formal hóa việc dự đoán này bằng **kỳ vọng có điều kiện (conditional expectation)**: `E(y | X = x0)`. **Hàm hồi quy (regression function)** chính là hàm kỳ vọng có điều kiện này.

**III. Mô Hình Hồi Quy Tuyến Tính (The Linear Regression Model)**

Mô hình hồi quy tuyến tính giả định rằng mối quan hệ giữa các biến đầu vào `X` và biến mục tiêu `Y` là **tuyến tính**.

1.  **Trường hợp đơn biến (Univariate - 1 feature X):**
    *   Phương trình đường thẳng: `Ŷ = b0 + b1*X`
    *   `b0`: Hệ số chặn (intercept) - giá trị dự đoán của Y khi X=0.
    *   `b1`: Hệ số góc (slope) - mức thay đổi của Y dự đoán khi X tăng 1 đơn vị.
    *   Với cá thể `i`: `ŷ_i = b0 + b1*x_i`
2.  **Trường hợp đa biến (Multivariate - p features):**
    *   Phương trình siêu phẳng: `Ŷ = b0 + b1*X1 + b2*X2 + ... + bp*Xp`
3.  **Biểu diễn dưới dạng ma trận:**
    *   Để gom `b0` vào, ta thêm một cột gồm toàn số 1 vào ma trận dữ liệu `X` (gọi là design matrix).
    *   `X` (n x (p+1)), `b` ((p+1) x 1), `ŷ` (n x 1)
    *   Phương trình: `ŷ = Xb`
4.  **Dữ liệu đã tâm hóa (Mean-centered):** Nếu cả predictors `X` và response `Y` đã được trừ đi giá trị trung bình của chúng, thì hệ số chặn `b0` thường có thể được bỏ qua trong một số diễn giải hình học (siêu phẳng đi qua gốc tọa độ).
5.  **Câu hỏi:** Làm thế nào để tìm ra vector hệ số `b` tốt nhất?

**IV. Đo Lường Sai Số (The Error Measure)**

Để tìm `b` tốt nhất, chúng ta cần một cách đo lường mức độ "tốt" của dự đoán.

1.  **Mục tiêu:** Chúng ta muốn dự đoán `ŷ_i` **càng gần** giá trị thực tế `y_i` càng tốt.
2.  **Thước đo phổ biến:** **Bình phương khoảng cách (Squared Distance)** giữa giá trị thực và giá trị dự đoán: `d²(y_i, ŷ_i) = (y_i - ŷ_i)²`.
3.  **Tổng sai số trên toàn bộ dữ liệu:**
    *   **Tổng bình phương sai số (Sum of Squared Errors - SSE):** `SSE = Σ d²(y_i, ŷ_i) = Σ (y_i - ŷ_i)²`.
    *   **Sai số bình phương trung bình (Mean Squared Error - MSE):** `MSE = (1/n) * SSE = (1/n) * Σ (y_i - ŷ_i)²`. MSE thường được ưa chuộng hơn vì nó không phụ thuộc vào số lượng mẫu `n`.
4.  **Sai số dưới dạng vector (Residuals):**
    *   Vector phần dư (residual vector): `e = y - ŷ = y - Xb`.
    *   `MSE = (1/n) * ||e||² = (1/n) * ||y - Xb||² = (1/n) * (y - Xb)^T (y - Xb)`.
    *   **Quan trọng:** MSE tỉ lệ thuận với bình phương độ dài (norm) của vector phần dư `e`.

**V. Thuật Toán Bình Phương Tối Thiểu (The Least Squares Algorithm - OLS)**

Phương pháp phổ biến nhất để tìm `b` trong hồi quy tuyến tính là **Bình phương tối thiểu thông thường (Ordinary Least Squares - OLS)**, nhằm mục đích **tối thiểu hóa MSE**.

1.  **Phương pháp:** Tìm `b` sao cho đạo hàm (gradient) của MSE theo `b` bằng 0.
    *   `∇MSE(b) = ∂MSE/∂b`
    *   Tính toán đạo hàm: `∇MSE(b) = (2/n) X^T X b - (2/n) X^T y`.
2.  **Phương trình chuẩn (Normal Equations):** Đặt đạo hàm bằng 0:
    *   `(2/n) X^T X b - (2/n) X^T y = 0`
    *   `=> X^T X b = X^T y`
3.  **Nghiệm OLS:** Nếu ma trận `X^T X` khả nghịch (invertible), thì nghiệm duy nhất cho `b` là:
    *   `b = (X^T X)^(-1) X^T y`
4.  **Vector dự đoán:** `ŷ = Xb = X (X^T X)^(-1) X^T y`
5.  **Ma trận Hat (H):** `H = X (X^T X)^(-1) X^T`. Ma trận này "đội mũ" cho `y` để tạo ra `ŷ` (`ŷ = Hy`).
    *   `H` là một **ma trận chiếu vuông góc (orthogonal projector)**. Nó chiếu vector `y` lên không gian con được tạo bởi các cột của `X`.
    *   Tính chất: Đối xứng (`H^T = H`), Lũy đẳng (`H^2 = H`), Trị riêng bằng 0 hoặc 1.

**VI. Góc Nhìn Hình Học của OLS (Geometries of OLS)**

1.  **Góc Nhìn Hàng (Rows Perspective - Không gian Đặc trưng):**
    *   Mỗi cặp `(x_i, y_i)` là một điểm trong không gian (p+1) chiều.
    *   Mô hình hồi quy `ŷ = Xb` định nghĩa một đường thẳng (p=1), mặt phẳng (p=2), hoặc siêu phẳng (p>2).
    *   Mục tiêu OLS: Tìm siêu phẳng **tối thiểu hóa tổng bình phương các khoảng cách theo chiều dọc (phần dư `e_i`)** từ các điểm dữ liệu thực tế `y_i` đến siêu phẳng đó (`ŷ_i`).
2.  **Góc Nhìn Cột (Columns Perspective - Không gian Cá thể):**
    *   Mỗi biến (cột của `X` và vector `y`) là một vector trong không gian `n` chiều.
    *   Các cột của `X` tạo thành một không gian con `S_X`.
    *   Vector dự đoán `ŷ = Xb` là một tổ hợp tuyến tính của các cột `X`, do đó `ŷ` **nằm trong** không gian con `S_X`.
    *   Vector thực tế `y` thường nằm ngoài `S_X`.
    *   Mục tiêu OLS: Tìm vector `ŷ` **trong `S_X`** sao cho nó **gần nhất** với vector `y`. Điểm gần nhất chính là **hình chiếu vuông góc** của `y` lên `S_X`.
    *   Vector phần dư `e = y - ŷ` là vector nối từ điểm chiếu `ŷ` đến điểm thực `y`. Vector `e` **vuông góc** với không gian con `S_X`. OLS tối thiểu hóa độ dài bình phương của vector `e`.
3.  **Góc Nhìn Tham Số (Parameters Perspective):**
    *   Xem MSE là một hàm của các hệ số `b`.
    *   Hàm `MSE(b)` có dạng một **mặt paraboloid (hình cái bát)** trong không gian của `b` và `MSE`.
    *   Mục tiêu OLS: Tìm điểm **thấp nhất (minimum)** trên bề mặt paraboloid này.
    *   Các đường đồng mức (contours) của MSE trên mặt phẳng tham số `b` là các hình elip đồng tâm. Tâm của các elip này chính là nghiệm OLS `b* = (X^T X)^(-1) X^T y`.

**VII. Gradient Descent (GD) - Một Phương Pháp Tối Ưu Khác**

Khi việc giải trực tiếp Normal Equations (tính ma trận nghịch đảo) gặp khó khăn (ma trận lớn, không khả nghịch), ta có thể dùng phương pháp lặp như Gradient Descent để tìm nghiệm xấp xỉ.

1.  **Ý tưởng:** Bắt đầu từ một điểm `b^(0)` ngẫu nhiên trên bề mặt lỗi MSE, và lặp đi lặp lại việc di chuyển một bước nhỏ theo hướng **ngược lại với hướng dốc nhất (gradient)** tại điểm hiện tại, cho đến khi tiến gần đến điểm cực tiểu.
2.  **Công thức cập nhật:** `b^(s+1) = b^(s) + α * v^(s)`
    *   `b^(s)`: Vector tham số ở bước `s`.
    *   `v^(s)`: Hướng di chuyển (là vector đơn vị ngược hướng với gradient: `v^(s) ≈ -∇E(b^(s))`).
    *   `α`: Kích thước bước đi (step size) hay **tốc độ học (learning rate)**.
3.  **Thuật toán GD cho Hồi quy tuyến tính:**
    *   **Khởi tạo:** Chọn `b^(0)` ngẫu nhiên.
    *   **Lặp (cho s = 0, 1, 2, ...):**
        *   Tính gradient: `∇E(b^(s)) = (2/n) X^T (Xb^(s) - y)`.
        *   Cập nhật `b`: `b^(s+1) = b^(s) - α * ∇E(b^(s))`.
    *   **Dừng:** Khi `b` hội tụ (thay đổi rất ít giữa các bước).
4.  **Cập nhật Pointwise:** Có thể viết lại công thức cập nhật cho từng hệ số `b_j` riêng lẻ, sử dụng đạo hàm riêng `∂E/∂b_j`. Việc cập nhật phải được thực hiện đồng thời cho tất cả `j`.

Tóm lại, Hồi quy tuyến tính là một mô hình cơ bản nhưng mạnh mẽ để mô tả mối quan hệ tuyến tính giữa các biến. Phương pháp Bình phương tối thiểu (OLS) cung cấp một nghiệm chính xác (nếu có thể) bằng cách giải Normal Equations, dựa trên việc tối thiểu hóa tổng bình phương sai số (MSE). Các góc nhìn hình học giúp hiểu bản chất của OLS là tìm đường/mặt phẳng phù hợp nhất (Row view) hoặc tìm phép chiếu vuông góc (Column view). Gradient Descent là một thuật toán lặp hiệu quả để tìm nghiệm OLS khi giải trực tiếp gặp khó khăn.