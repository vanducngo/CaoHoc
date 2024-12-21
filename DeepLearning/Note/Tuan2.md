<h2> Mô hình máy học tổng quát</h2>

- Đạo hàm:
    - Đạo hàm dương: Hướng đi lên (bên phải so với cực tiểu)
    - Đạo hàm âm: Hướng đi xuống (bên trái so với cực tiểu)
    - => Dấu đạo hàm ngược hướng điểm cực tiểu.

- Khởi tạo theta random
- Di chuyển về điểm cực tiểu: Theta = theta - f'(theta)

- Problem 1: Giá trị đạo hàm lớn => Khi cập nhật, theta đi "quá đà"
    -  => Solution: nhân với hệ số learning rate alpha: Theta = theta - alpha * f'(theta)
- Problem 2: Cập nhật theta đến khi nào thì dừng.
    - => Solutions 1: Độ dốc không còn hay đạo hàm đủ nhỏ. |dL(theta)|/dtheta < epsilon
    - => Solutions 2: Lặp sau số vòng lặp nhất định (đủ lớn)
![Hình minh họa](./images/img1.jpg)

- Problem 3: Có nhiều điểm cực tiểu.
    - => Solutions 1: Chạy nhiều lần, lấy giá trị nhỏ nhất => Tốn tài nguyên
    - => Solutions 2: Dùng momentum (quán tính) khi cập nhật
        -  Thuật toán Adam được cài sẵn trong các tensorflow, pytorch.


- 3 công việc cần làm khi thiết kế mô hình: 
    - Thiết kế mô hình dự đoán f(x)
    - Thiết kế hàm lỗi dự đoán L(theta, x,y)
    - Tìm tham số theta để hàm lộ lỗi L nhỏ nhất => Các mô hình DL