**Tổng Quan Kiến Thức về Học Máy (Machine Learning)**

**I. Học Máy Là Gì?**

1.  **Định nghĩa Cốt Lõi (Tom Mitchell):** Học máy (ML) là lĩnh vực nghiên cứu giúp máy tính có khả năng "học" từ **Kinh nghiệm (E)** để cải thiện hiệu năng thực hiện một **Tác vụ (T)**, được đo lường bằng một **Thước đo hiệu năng (P)**. Nói đơn giản: Máy tính học cách làm tốt hơn một việc gì đó thông qua dữ liệu và trải nghiệm.
2.  **Góc Nhìn Thực Hành:** Thường thì mục tiêu là học một **mô hình (model)** hoặc **hàm (function) f** với các **tham số (parameters) θ**. Mô hình này nhận **đầu vào X** và tạo ra **đầu ra dự đoán y**. Quá trình "học" chính là tìm ra bộ tham số `θ` tốt nhất để **giảm thiểu sai số (Error E)** giữa dự đoán và kết quả thực tế (`argmin_θ E(f_θ(X))`).

**II. Các Khái Niệm Nền Tảng**

1.  **Inductive Bias (Thiên kiến quy nạp - b):**
    *   **Bản chất:** Là những **giả định (assumptions)** mà chúng ta phải đưa vào mô hình để nó có thể học và tổng quát hóa từ dữ liệu hữu hạn. Không có mô hình nào hoạt động mà không có thiên kiến.
    *   **Các dạng:** Lựa chọn cấu trúc mô hình (mạng nơ-ron, cây quyết định,...), cài đặt siêu tham số (độ sâu cây, tốc độ học,...), giả định về phân phối dữ liệu, hoặc sử dụng kiến thức từ bài toán trước (học chuyển giao - transfer learning).
    *   **Công thức đầy đủ:** `argmin_{θ, b} E(f_{θ, b}(X))` - Việc tối ưu bao gồm cả chọn tham số và chọn thiên kiến phù hợp.
2.  **Học Máy vs. Thống Kê:**
    *   **Điểm chung:** Cùng phân tích dữ liệu và đưa ra dự đoán.
    *   **Khác biệt chính:**
        *   **Thống kê:** Tập trung vào việc **hiểu** dữ liệu và quá trình sinh ra nó, thường giả định một mô hình cơ bản có thể diễn giải được (ví dụ: hồi quy tuyến tính). Mục tiêu là suy luận và giải thích.
        *   **Học máy:** Tập trung vào việc **dự đoán** chính xác nhất có thể, thường xem quá trình sinh dữ liệu là "hộp đen". Mục tiêu là tự động hóa tác vụ và tối ưu hiệu năng dự đoán.

**III. Các Loại Hình Học Máy Chính**

1.  **Supervised Learning (Học có giám sát):**
    *   **Dữ liệu:** Có đầu vào (features) và **nhãn (labels)** hoặc kết quả đúng tương ứng.
    *   **Mục tiêu:** Học một ánh xạ từ đầu vào đến đầu ra đã biết.
    *   **Ví dụ:** Phân loại email (spam/không spam), dự đoán giá nhà.
2.  **Unsupervised Learning (Học không giám sát):**
    *   **Dữ liệu:** Chỉ có đầu vào, **không có nhãn**.
    *   **Mục tiêu:** Khám phá cấu trúc, quy luật, hoặc các mẫu tiềm ẩn trong dữ liệu.
    *   **Ví dụ:** Phân cụm khách hàng, giảm chiều dữ liệu để trực quan hóa.
3.  **Reinforcement Learning (Học tăng cường):**
    *   **Cơ chế:** Một **tác nhân (agent)** tương tác với **môi trường (environment)**, nhận **phần thưởng (reward)** hoặc phạt dựa trên **hành động (action)** của mình.
    *   **Mục tiêu:** Học một **chính sách (policy)** - chiến lược hành động - để tối đa hóa tổng phần thưởng tích lũy.
    *   **Ví dụ:** Huấn luyện robot chơi game, tự lái xe.

*(Góc nhìn Yann LeCun: Học không giám sát/tiên đoán là phần "bánh", học có giám sát là "kem phủ", học tăng cường là "quả cherry" – nhấn mạnh tiềm năng của việc học từ dữ liệu không nhãn).*

**IV. Đi Sâu Vào Supervised Learning**

1.  **Hai Bài Toán Chính:**
    *   **Classification (Phân loại):** Dự đoán một nhãn **rời rạc (discrete)**, không có thứ tự (ví dụ: "mèo", "chó", "chim"; "spam", "không spam"; loại hoa Iris).
        *   **Binary Classification:** 2 lớp.
        *   **Multi-class Classification:** Nhiều hơn 2 lớp.
        *   Mô hình tạo ra **Decision Boundary** (đường/bề mặt phân chia các lớp).
    *   **Regression (Hồi quy):** Dự đoán một giá trị **liên tục (continuous)** (ví dụ: giá nhà, nhiệt độ, giá cổ phiếu).
2.  **Hàm Mất Mát (Loss Function - l(y, ŷ)):** Đo lường sự khác biệt/sai số giữa giá trị dự đoán `ŷ` và giá trị thực `y`.
    *   **Phân loại:** Thường dùng **Zero-one Loss** (`I(y ≠ ŷ)` - lỗi bằng 1 nếu sai, 0 nếu đúng) hoặc các hàm surrogate khác (như Cross-Entropy). Có thể dùng **Asymmetric Loss** nếu các loại lỗi có chi phí khác nhau.
    *   **Hồi quy:** Thường dùng **Quadratic Loss / l2 Loss** (`(y - ŷ)^2`).
3.  **Empirical Risk (Rủi ro thực nghiệm - L(θ)):** Là **trung bình mất mát** trên toàn bộ **tập huấn luyện (training set)**.
    *   `L(θ; D_train) = (1 / |D_train|) * Σ l(y_n, f(x_n; θ))`
    *   **Mục tiêu huấn luyện (ERM - Empirical Risk Minimization):** Tìm tham số `θ` để tối thiểu hóa rủi ro thực nghiệm này.
4.  **Ví dụ Mô Hình:**
    *   **Decision Tree (Cây quyết định):** Dùng các quy tắc "if-then" dựa trên đặc trưng để phân loại/hồi quy.
    *   **Linear Regression (Hồi quy tuyến tính):** Mô hình hóa mối quan hệ tuyến tính `y = w^T*x + b`.
    *   **Polynomial Regression (Hồi quy đa thức):** Mở rộng hồi quy tuyến tính bằng cách tạo đặc trưng bậc cao (`x^2`, `x^3`,...), cho phép mô hình hóa quan hệ phi tuyến.

**V. Đi Sâu Vào Unsupervised Learning**

1.  **Clustering (Phân cụm):**
    *   **Mục tiêu:** Tự động nhóm các điểm dữ liệu tương tự nhau thành các cụm (clusters).
    *   **Nguyên tắc:** Tối đa hóa sự tương đồng trong cụm và tối thiểu hóa sự tương đồng giữa các cụm.
    *   **Ứng dụng:** Phân khúc thị trường, phân loại tài liệu.
2.  **Dimensionality Reduction (Giảm chiều dữ liệu):**
    *   **Mục tiêu:** Giảm số lượng đặc trưng (chiều) của dữ liệu trong khi giữ lại nhiều thông tin quan trọng nhất.
    *   **Lợi ích:** Giảm nhiễu, tăng tốc độ huấn luyện, dễ trực quan hóa.
    *   **Ví dụ:** PCA (Principal Component Analysis).

**VI. Ba Thành Phần Cốt Lõi Của Quá Trình Học (Domingos)**

1.  **Representation (Biểu diễn):** Cách mô hình được cấu trúc và không gian các hàm mà nó có thể học (hypothesis space). Ví dụ: cây quyết định, mạng nơ-ron.
2.  **Evaluation (Đánh giá):** Hàm mục tiêu hoặc hàm mất mát dùng để đo lường mức độ "tốt" của một mô hình trên dữ liệu. Ví dụ: MSE, tỷ lệ lỗi.
3.  **Optimization (Tối ưu hóa):** Thuật toán dùng để tìm kiếm các tham số tốt nhất cho mô hình dựa trên hàm đánh giá. Ví dụ: Gradient Descent.

**VII. Tầm Quan Trọng Của Dữ Liệu và Biểu Diễn**

1.  **Feature Engineering (Kỹ thuật đặc trưng):**
    *   Quá trình sử dụng kiến thức chuyên môn để tạo ra các **đặc trưng (features)** tốt hơn từ dữ liệu thô, giúp mô hình học hiệu quả hơn.
    *   Bao gồm: lựa chọn đặc trưng, tạo đặc trưng mới (ví dụ: đa thức hóa), chuẩn hóa dữ liệu (scaling).
    *   Rất quan trọng, đặc biệt với dữ liệu có cấu trúc (dạng bảng).
2.  **Representation Matters:** Hiệu năng của mô hình phụ thuộc rất nhiều vào cách dữ liệu được biểu diễn. Một biểu diễn tốt có thể làm bài toán trở nên dễ dàng hơn đáng kể.

**VIII. Deep Learning (Học sâu) - Hướng Tới Học Biểu Diễn Tự Động**

1.  **Động lực:** Việc feature engineering thủ công rất khó khăn cho dữ liệu phức tạp (ảnh, âm thanh, văn bản).
2.  **Ý tưởng:** Xây dựng các mô hình (thường là Mạng Nơ-ron Sâu - DNN) có khả năng **tự động học (learn)** các biểu diễn đặc trưng hữu ích từ dữ liệu thô qua nhiều lớp (layers) xử lý.
    *   `f(x; w, V) = w^T * φ(x; V)`: Mô hình học cả bộ trích xuất đặc trưng `φ(x; V)` và bộ phân loại/hồi quy cuối cùng `w`.
3.  **Kiến trúc phổ biến:**
    *   **CNN (Convolutional Neural Network):** Cho dữ liệu dạng lưới (ảnh).
    *   **RNN (Recurrent Neural Network):** Cho dữ liệu tuần tự (văn bản, chuỗi thời gian).

**IX. Thách Thức Cốt Lõi: Generalization (Tổng quát hóa)**

1.  **Mục tiêu thực sự:** Mô hình phải hoạt động tốt trên **dữ liệu mới, chưa từng thấy (unseen data)**, chứ không chỉ trên dữ liệu đã dùng để huấn luyện.
2.  **Overfitting (Quá khớp):** Mô hình học quá tốt dữ liệu huấn luyện (kể cả nhiễu), trở nên quá phức tạp và hoạt động kém trên dữ liệu mới. Lỗi huấn luyện thấp, lỗi kiểm tra cao.
3.  **Underfitting (Dưới khớp):** Mô hình quá đơn giản, không nắm bắt được quy luật trong dữ liệu. Lỗi huấn luyện và lỗi kiểm tra đều cao.
4.  **Đánh giá khả năng tổng quát hóa:**
    *   **Population Risk:** Mất mát kỳ vọng trên toàn bộ phân phối dữ liệu (lý thuyết, không tính được).
    *   **Test Risk:** Ước lượng Population Risk bằng cách tính mất mát trên một **tập kiểm tra (test set)** độc lập.
    *   **Generalization Gap:** Sự khác biệt giữa Test Risk và Training Risk. Gap lớn là dấu hiệu overfitting.
    *   **Đường cong chữ U:** Lỗi kiểm tra thường giảm khi tăng độ phức tạp mô hình đến một điểm tối ưu, sau đó lại tăng lên do overfitting.
5.  **Quy trình thực hành:** Chia dữ liệu thành 3 tập:
    *   **Training Set:** Huấn luyện tham số mô hình (`θ`).
    *   **Validation Set:** Lựa chọn mô hình, điều chỉnh siêu tham số (ví dụ: chọn bậc đa thức `D`, kiến trúc mạng).
    *   **Test Set:** Đánh giá cuối cùng, khách quan về hiệu năng của mô hình đã chọn. **Không được** dùng tập test để điều chỉnh mô hình.

**X. Giới Hạn Cơ Bản: No Free Lunch Theorem**

*   **Nội dung:** Không có thuật toán học máy nào là tốt nhất cho mọi bài toán.
*   **Ý nghĩa:** Một mô hình/thuật toán hoạt động tốt trên loại dữ liệu/bài toán này có thể thất bại trên loại khác. Việc lựa chọn mô hình phù hợp (inductive bias đúng) dựa trên kiến thức lĩnh vực và thực nghiệm là rất quan trọng. Cần có nhiều công cụ (mô hình, thuật toán) khác nhau.

Tóm lại, học máy là một lĩnh vực mạnh mẽ giúp máy tính học từ dữ liệu để thực hiện các tác vụ. Quá trình này bao gồm việc lựa chọn cách biểu diễn dữ liệu, định nghĩa hàm đánh giá (mất mát), và sử dụng thuật toán tối ưu để tìm mô hình tốt nhất. Thách thức lớn nhất là đảm bảo mô hình có khả năng tổng quát hóa tốt trên dữ liệu mới, tránh overfitting và underfitting, và không có giải pháp nào là hoàn hảo cho mọi vấn đề.