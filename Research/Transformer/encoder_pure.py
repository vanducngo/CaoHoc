import math
import copy # Để tạo bản sao sâu của các đối tượng (ví dụ: trọng số cho các layer)

# -------------------------------------
# Hàm tiện ích cho Ma trận và Vector (Python thuần)
# Lưu ý: Các hàm này rất cơ bản và không hiệu quả bằng NumPy.
# Chúng được viết để minh họa logic bằng Python thuần.
# -------------------------------------

def shape(matrix):
    """Trả về kích thước của ma trận (list of lists)"""
    if not isinstance(matrix, list):
        return () # Scalar hoặc vector 1D (không phải list of lists)
    if not matrix:
        return (0,) # Ma trận rỗng
    if not isinstance(matrix[0], list):
        return (len(matrix),) # Vector (list of numbers)
    # Giả định tất cả các hàng con có cùng độ dài
    return (len(matrix), len(matrix[0]) if matrix else 0)

def dot_product(vec1, vec2):
    """Tính tích vô hướng của hai vector"""
    if len(vec1) != len(vec2):
        raise ValueError("Vectors phải có cùng độ dài để tính tích vô hướng")
    return sum(x * y for x, y in zip(vec1, vec2))

def transpose(matrix):
    """Chuyển vị ma trận (list of lists)"""
    rows, cols = shape(matrix)
    transposed_matrix = [[0 for _ in range(rows)] for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            transposed_matrix[j][i] = matrix[i][j]
    return transposed_matrix

def matrix_multiply(matrix1, matrix2):
    """Nhân hai ma trận (list of lists)"""
    rows1, cols1 = shape(matrix1)
    rows2, cols2 = shape(matrix2)

    if cols1 != rows2:
        raise ValueError(f"Kích thước không khớp để nhân ma trận: {shape(matrix1)} vs {shape(matrix2)}")

    result_matrix = [[0 for _ in range(cols2)] for _ in range(rows1)]

    # Chuyển vị matrix2 để tính toán hiệu quả hơn (truy cập cột dễ dàng hơn)
    # Mặc dù không thực sự hiệu quả trong Python thuần, nó mô phỏng cách hoạt động
    matrix2_T = transpose(matrix2)

    for i in range(rows1):
        for j in range(cols2):
            # Tích vô hướng của hàng i từ matrix1 và cột j từ matrix2 (hàng j từ matrix2_T)
            result_matrix[i][j] = dot_product(matrix1[i], matrix2_T[j])

    return result_matrix

def matrix_add(matrix1, matrix2):
    """Cộng hai ma trận cùng kích thước"""
    rows1, cols1 = shape(matrix1)
    rows2, cols2 = shape(matrix2)
    if rows1 != rows2 or cols1 != cols2:
        raise ValueError(f"Kích thước ma trận phải giống nhau để cộng: {shape(matrix1)} vs {shape(matrix2)}")

    result_matrix = [[0 for _ in range(cols1)] for _ in range(rows1)]
    for i in range(rows1):
        for j in range(cols1):
            result_matrix[i][j] = matrix1[i][j] + matrix2[i][j]
    return result_matrix

def matrix_vector_add(matrix, vector):
    """Cộng một vector (bias) vào mỗi hàng của ma trận (broadcasting)"""
    rows, cols = shape(matrix)
    if len(vector) != cols:
         raise ValueError(f"Độ dài vector ({len(vector)}) phải khớp với số cột ma trận ({cols})")

    result_matrix = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            result_matrix[i][j] = matrix[i][j] + vector[j] # Cộng phần tử bias tương ứng
    return result_matrix

# -------------------------------------
# Positional Encoding
# -------------------------------------

def positional_encoding(seq_len, d_model):
    """
    Tạo ma trận Positional Encoding.

    Args:
        seq_len (int): Độ dài của chuỗi (số lượng token).
        d_model (int): Kích thước của vector embedding (chiều sâu model).

    Returns:
        list[list[float]]: Ma trận positional encoding với shape (seq_len, d_model).
    """
    pe = [[0.0 for _ in range(d_model)] for _ in range(seq_len)]
    position = [[pos] for pos in range(seq_len)] # shape (seq_len, 1)
    # Tính toán tần số góc: 1 / (10000^(2i / d_model))
    div_term = [math.pow(10000.0, (2 * i) / d_model) for i in range(0, d_model, 2)]

    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            # Chỉ số trong div_term
            div_idx = i // 2
            # Áp dụng sin cho các chiều chẵn
            pe[pos][i] = math.sin(position[pos][0] / div_term[div_idx])
            # Áp dụng cos cho các chiều lẻ (nếu có)
            if i + 1 < d_model:
                pe[pos][i + 1] = math.cos(position[pos][0] / div_term[div_idx])

    return pe

# -------------------------------------
# Scaled Dot-Product Attention
# -------------------------------------

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Tính toán Scaled Dot-Product Attention.

    Args:
        Q (list[list[float]]): Ma trận Query, shape (seq_len_q, d_k).
        K (list[list[float]]): Ma trận Key, shape (seq_len_k, d_k).
        V (list[list[float]]): Ma trận Value, shape (seq_len_v, d_v). (seq_len_k == seq_len_v)
        mask (list[list[bool]], optional): Ma trận mask, shape (seq_len_q, seq_len_k).
                                           True ở vị trí cần mask (bỏ qua). Defaults to None.

    Returns:
        tuple(list[list[float]], list[list[float]]): Output của attention và ma trận attention weights.
                                                    Output shape (seq_len_q, d_v), weights shape (seq_len_q, seq_len_k).
    """
    seq_len_q, d_k = shape(Q)
    seq_len_k, _ = shape(K) # d_k của K phải bằng d_k của Q
    seq_len_v, d_v = shape(V) # d_v có thể khác d_k, seq_len_v phải bằng seq_len_k

    if shape(K)[1] != d_k:
        raise ValueError("Kích thước chiều cuối của Q và K phải giống nhau.")
    if seq_len_k != seq_len_v:
         raise ValueError("Độ dài chuỗi của K và V phải giống nhau.")

    # 1. Tính điểm attention: MatMul(Q, K^T)
    K_T = transpose(K) # shape (d_k, seq_len_k)
    scores = matrix_multiply(Q, K_T) # shape (seq_len_q, seq_len_k)

    # 2. Scale điểm attention: scores / sqrt(d_k)
    scale_factor = math.sqrt(d_k)
    if scale_factor == 0:
        raise ValueError("d_k không thể bằng 0")

    scaled_scores = [[score / scale_factor for score in row] for row in scores]

    # 3. Áp dụng mask (nếu có) trước softmax
    # Mask thường dùng để ẩn đi padding hoặc các vị trí tương lai (trong decoder)
    if mask is not None:
        if shape(mask) != (seq_len_q, seq_len_k):
             raise ValueError(f"Kích thước mask {shape(mask)} không khớp với kích thước scores {(seq_len_q, seq_len_k)}")
        for i in range(seq_len_q):
            for j in range(seq_len_k):
                if mask[i][j]: # Nếu vị trí (i, j) cần được mask
                    scaled_scores[i][j] = -1e9 # Thay thế bằng một giá trị rất nhỏ (âm vô cùng)

    # 4. Tính attention weights bằng Softmax trên từng hàng (theo chiều key)
    attention_weights = []
    for row in scaled_scores:
        # Tính exp cho mỗi phần tử trong hàng
        exp_row = [math.exp(score) for score in row]
        # Tính tổng các giá trị exp trong hàng
        sum_exp_row = sum(exp_row)
        # Chuẩn hóa để có softmax (nếu tổng khác 0)
        if sum_exp_row == 0:
             # Trường hợp đặc biệt: tất cả scores đều là -inf -> softmax = 0
             softmax_row = [0.0] * len(row)
        else:
             softmax_row = [exp_val / sum_exp_row for exp_val in exp_row]
        attention_weights.append(softmax_row)
        # attention_weights shape: (seq_len_q, seq_len_k)

    # 5. Tính output: MatMul(attention_weights, V)
    output = matrix_multiply(attention_weights, V) # shape (seq_len_q, d_v)

    return output, attention_weights

# -------------------------------------
# Multi-Head Attention
# -------------------------------------

class MultiHeadAttention:
    """
    Lớp Multi-Head Attention.

    Thực hiện chiếu tuyến tính đầu vào, chia thành nhiều 'đầu' (heads),
    áp dụng scaled dot-product attention cho từng đầu song song,
    ghép kết quả lại và chiếu tuyến tính lần cuối.
    """
    def __init__(self, d_model, num_heads):
        """
        Khởi tạo lớp MultiHeadAttention.

        Args:
            d_model (int): Kích thước embedding đầu vào và đầu ra.
            num_heads (int): Số lượng attention heads. d_model phải chia hết cho num_heads.
        """
        if d_model % num_heads != 0:
            raise ValueError("d_model phải chia hết cho num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # Kích thước của mỗi head

        # Khởi tạo các ma trận trọng số (Weight matrices) và biases
        # Trong thực tế, chúng được học trong quá trình huấn luyện.
        # Ở đây, chúng ta chỉ khởi tạo giả lập (ví dụ: ma trận đơn vị hoặc giá trị nhỏ)
        # Wq, Wk, Wv: Các ma trận chiếu cho Query, Key, Value cho tất cả các head gộp lại
        # Wo: Ma trận chiếu đầu ra sau khi ghép các head
        # Kích thước: (d_model, d_model)
        # Bias: kích thước d_model
        scale = 0.1 # Giá trị nhỏ để khởi tạo
        self.Wq = [[(scale * (i * d_model + j + 1)) % 1 for j in range(d_model)] for i in range(d_model)]
        self.Wk = [[(scale * (i * d_model + j + 2)) % 1 for j in range(d_model)] for i in range(d_model)]
        self.Wv = [[(scale * (i * d_model + j + 3)) % 1 for j in range(d_model)] for i in range(d_model)]
        self.Wo = [[(scale * (i * d_model + j + 4)) % 1 for j in range(d_model)] for i in range(d_model)]

        # Biases (giả lập là vector 0)
        self.bq = [0.0] * d_model
        self.bk = [0.0] * d_model
        self.bv = [0.0] * d_model
        self.bo = [0.0] * d_model


    def split_heads(self, matrix):
        """
        Chia ma trận (seq_len, d_model) thành (num_heads, seq_len, d_k).
        Trong Python thuần, chúng ta trả về list[list[list[float]]] với shape (num_heads, seq_len, d_k).
        """
        seq_len, model_dim = shape(matrix)
        if model_dim != self.d_model:
             raise ValueError(f"Kích thước ma trận ({model_dim}) không khớp d_model ({self.d_model})")

        # Khởi tạo cấu trúc dữ liệu cho các head
        heads = [[[0.0 for _ in range(self.d_k)] for _ in range(seq_len)] for _ in range(self.num_heads)]

        for h in range(self.num_heads):
            start_col = h * self.d_k
            end_col = start_col + self.d_k
            for i in range(seq_len):
                # Trích xuất phần tương ứng với head 'h' cho vị trí 'i'
                heads[h][i] = matrix[i][start_col:end_col]
        return heads # Shape: (num_heads, seq_len, d_k)

    def combine_heads(self, heads):
        """
        Ghép các head lại từ (num_heads, seq_len, d_k) thành (seq_len, d_model).
        """
        num_heads_in, seq_len, d_k_in = shape(heads)
        if num_heads_in != self.num_heads or d_k_in != self.d_k:
             raise ValueError("Kích thước đầu vào không khớp với cấu hình head")

        # Khởi tạo ma trận kết quả
        combined = [[0.0 for _ in range(self.d_model)] for _ in range(seq_len)]

        for i in range(seq_len):
            row_combined = []
            for h in range(self.num_heads):
                # Nối các vector d_k từ mỗi head lại với nhau
                row_combined.extend(heads[h][i])
            combined[i] = row_combined

        return combined # Shape: (seq_len, d_model)

    def forward(self, Q_in, K_in, V_in, mask=None):
        """
        Tính toán forward pass cho Multi-Head Attention.

        Args:
            Q_in (list[list[float]]): Input Query, shape (seq_len_q, d_model).
            K_in (list[list[float]]): Input Key, shape (seq_len_k, d_model).
            V_in (list[list[float]]): Input Value, shape (seq_len_v, d_model).
                                      Trong self-attention, Q_in=K_in=V_in.
            mask (list[list[bool]], optional): Mask cho scaled_dot_product_attention.
                                               Shape (seq_len_q, seq_len_k).

        Returns:
            list[list[float]]: Output của Multi-Head Attention, shape (seq_len_q, d_model).
        """
        seq_len_q, _ = shape(Q_in)
        seq_len_k, _ = shape(K_in)
        seq_len_v, _ = shape(V_in)

        # 1. Chiếu tuyến tính Q, K, V bằng các ma trận trọng số Wq, Wk, Wv
        # Q = Q_in * Wq + bq
        # K = K_in * Wk + bk
        # V = V_in * Wv + bv
        Q = matrix_vector_add(matrix_multiply(Q_in, self.Wq), self.bq) # shape (seq_len_q, d_model)
        K = matrix_vector_add(matrix_multiply(K_in, self.Wk), self.bk) # shape (seq_len_k, d_model)
        V = matrix_vector_add(matrix_multiply(V_in, self.Wv), self.bv) # shape (seq_len_v, d_model)

        # 2. Chia Q, K, V thành các head
        Q_heads = self.split_heads(Q) # shape (num_heads, seq_len_q, d_k)
        K_heads = self.split_heads(K) # shape (num_heads, seq_len_k, d_k)
        V_heads = self.split_heads(V) # shape (num_heads, seq_len_v, d_k)

        # 3. Áp dụng Scaled Dot-Product Attention cho từng head
        attention_outputs_heads = []
        # attention_weights_heads = [] # Có thể lưu lại nếu cần phân tích

        for h in range(self.num_heads):
            # Lấy Q, K, V của head thứ h
            Q_h = Q_heads[h] # shape (seq_len_q, d_k)
            K_h = K_heads[h] # shape (seq_len_k, d_k)
            V_h = V_heads[h] # shape (seq_len_v, d_k)

            # Tính attention cho head này
            # Lưu ý: mask (nếu có) cần được áp dụng giống nhau cho tất cả các head
            output_h, _ = scaled_dot_product_attention(Q_h, K_h, V_h, mask) # output_h shape (seq_len_q, d_k)
            attention_outputs_heads.append(output_h)
            # attention_weights_heads.append(weights_h)

        # attention_outputs_heads shape: (num_heads, seq_len_q, d_k)

        # 4. Ghép kết quả từ các head lại
        concatenated_output = self.combine_heads(attention_outputs_heads) # shape (seq_len_q, d_model)

        # 5. Áp dụng chiếu tuyến tính cuối cùng Wo và bias bo
        # output = concatenated_output * Wo + bo
        output = matrix_vector_add(matrix_multiply(concatenated_output, self.Wo), self.bo) # shape (seq_len_q, d_model)

        return output

# -------------------------------------
# Position-wise Feed-Forward Network
# -------------------------------------

class PositionwiseFeedForward:
    """
    Mạng Feed-Forward áp dụng độc lập cho từng vị trí.
    Bao gồm hai lớp tuyến tính với hàm kích hoạt ReLU ở giữa.
    FFN(x) = max(0, x * W1 + b1) * W2 + b2
    """
    def __init__(self, d_model, d_ff):
        """
        Khởi tạo lớp PositionwiseFeedForward.

        Args:
            d_model (int): Kích thước đầu vào và đầu ra.
            d_ff (int): Kích thước lớp ẩn bên trong (thường lớn hơn d_model, ví dụ: 2048).
        """
        self.d_model = d_model
        self.d_ff = d_ff

        # Khởi tạo trọng số và bias giả lập
        # W1: (d_model, d_ff), b1: (d_ff)
        # W2: (d_ff, d_model), b2: (d_model)
        scale = 0.1
        self.W1 = [[(scale * (i * d_ff + j + 5)) % 1 for j in range(d_ff)] for i in range(d_model)]
        self.b1 = [(scale * (j + 6)) % 1 for j in range(d_ff)]
        self.W2 = [[(scale * (i * d_model + j + 7)) % 1 for j in range(d_model)] for i in range(d_ff)]
        self.b2 = [(scale * (j + 8)) % 1 for j in range(d_model)]

    def forward(self, x):
        """
        Tính toán forward pass cho PositionwiseFeedForward.

        Args:
            x (list[list[float]]): Input, shape (seq_len, d_model).

        Returns:
            list[list[float]]: Output, shape (seq_len, d_model).
        """
        seq_len, model_dim = shape(x)
        if model_dim != self.d_model:
             raise ValueError(f"Kích thước input ({model_dim}) không khớp d_model ({self.d_model})")

        # 1. Lớp tuyến tính thứ nhất: x * W1 + b1
        hidden = matrix_vector_add(matrix_multiply(x, self.W1), self.b1) # shape (seq_len, d_ff)

        # 2. Hàm kích hoạt ReLU: max(0, hidden)
        # Áp dụng ReLU cho từng phần tử
        activated = [[max(0, val) for val in row] for row in hidden] # shape (seq_len, d_ff)

        # 3. Lớp tuyến tính thứ hai: activated * W2 + b2
        output = matrix_vector_add(matrix_multiply(activated, self.W2), self.b2) # shape (seq_len, d_model)

        return output

# -------------------------------------
# Layer Normalization
# -------------------------------------

class LayerNorm:
    """
    Chuẩn hóa lớp (Layer Normalization).
    Chuẩn hóa các features (d_model) độc lập tại mỗi vị trí (seq_len).
    LN(x) = gamma * (x - mean) / sqrt(variance + epsilon) + beta
    """
    def __init__(self, d_model, epsilon=1e-5):
        """
        Khởi tạo lớp LayerNorm.

        Args:
            d_model (int): Kích thước của chiều cần chuẩn hóa (feature dimension).
            epsilon (float): Giá trị nhỏ để tránh chia cho 0. Defaults to 1e-5.
        """
        self.d_model = d_model
        self.epsilon = epsilon
        # Tham số học được: gamma (scale) và beta (shift)
        # Khởi tạo giả lập: gamma = 1, beta = 0
        self.gamma = [1.0] * d_model # Vector shape (d_model,)
        self.beta = [0.0] * d_model  # Vector shape (d_model,)

    def forward(self, x):
        """
        Tính toán forward pass cho Layer Normalization.

        Args:
            x (list[list[float]]): Input, shape (seq_len, d_model).

        Returns:
            list[list[float]]: Output đã được chuẩn hóa, shape (seq_len, d_model).
        """
        seq_len, model_dim = shape(x)
        if model_dim != self.d_model:
             raise ValueError(f"Kích thước input ({model_dim}) không khớp d_model ({self.d_model})")

        output = [[0.0 for _ in range(self.d_model)] for _ in range(seq_len)]

        # Lặp qua từng vị trí (token) trong chuỗi
        for i in range(seq_len):
            # Lấy vector feature tại vị trí i
            vector = x[i] # shape (d_model,)

            # Tính mean của vector feature này
            mean = sum(vector) / self.d_model

            # Tính variance của vector feature này
            # Tổng bình phương độ lệch so với mean
            variance_sum = sum(math.pow(val - mean, 2) for val in vector)
            variance = variance_sum / self.d_model

            # Chuẩn hóa vector tại vị trí i
            denominator = math.sqrt(variance + self.epsilon)
            if denominator == 0: # Xử lý trường hợp mẫu số cực nhỏ
                normalized_vector = [0.0] * self.d_model
            else:
                normalized_vector = [(val - mean) / denominator for val in vector]

            # Áp dụng scale (gamma) và shift (beta)
            for j in range(self.d_model):
                output[i][j] = self.gamma[j] * normalized_vector[j] + self.beta[j]

        return output

# -------------------------------------
# Lớp Encoder (Encoder Layer)
# -------------------------------------

class EncoderLayer:
    """
    Một lớp Encoder đơn lẻ, bao gồm Multi-Head Attention và Feed-Forward Network,
    với các kết nối residual (Add) và chuẩn hóa lớp (Norm).
    """
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        """
        Khởi tạo một lớp Encoder.

        Args:
            d_model (int): Kích thước embedding.
            num_heads (int): Số lượng attention heads.
            d_ff (int): Kích thước lớp ẩn trong Feed-Forward network.
            dropout_rate (float): Tỷ lệ dropout (chỉ dùng trong huấn luyện, ở đây bỏ qua).
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        # self.dropout_rate = dropout_rate # Bỏ qua dropout trong bản demo này

        # 1. Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(d_model, num_heads)

        # 2. Layer Normalization sau Attention
        self.norm1 = LayerNorm(d_model)

        # 3. Position-wise Feed-Forward Network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

        # 4. Layer Normalization sau Feed-Forward
        self.norm2 = LayerNorm(d_model)

        # Dropout thường được áp dụng sau mỗi sub-layer (attention, feed-forward)
        # trước khi cộng residual và sau khi cộng residual + norm.
        # Chúng ta sẽ bỏ qua dropout ở đây cho đơn giản.

    def forward(self, x, mask=None):
        """
        Tính toán forward pass cho một lớp Encoder.

        Args:
            x (list[list[float]]): Input của lớp Encoder, shape (seq_len, d_model).
            mask (list[list[bool]], optional): Mask cho self-attention. Shape (seq_len, seq_len).

        Returns:
            list[list[float]]: Output của lớp Encoder, shape (seq_len, d_model).
        """
        seq_len, model_dim = shape(x)
        if model_dim != self.d_model:
             raise ValueError(f"Kích thước input ({model_dim}) không khớp d_model ({self.d_model})")

        # --- Sub-layer 1: Multi-Head Self-Attention ---
        # Tính output của attention. Vì là self-attention, Q, K, V đều là x.
        attn_output = self.self_attention.forward(x, x, x, mask)
        # attn_output shape: (seq_len, d_model)

        # --- Add & Norm 1 ---
        # Kết nối residual: cộng input ban đầu (x) với output của sub-layer (attn_output)
        residual_sum1 = matrix_add(x, attn_output)
        # Áp dụng Layer Normalization
        norm1_output = self.norm1.forward(residual_sum1)
        # norm1_output shape: (seq_len, d_model)

        # --- Sub-layer 2: Position-wise Feed-Forward ---
        # Tính output của feed-forward network
        ff_output = self.feed_forward.forward(norm1_output)
        # ff_output shape: (seq_len, d_model)

        # --- Add & Norm 2 ---
        # Kết nối residual: cộng input của sub-layer 2 (norm1_output) với output của nó (ff_output)
        residual_sum2 = matrix_add(norm1_output, ff_output)
        # Áp dụng Layer Normalization
        norm2_output = self.norm2.forward(residual_sum2)
        # norm2_output shape: (seq_len, d_model)

        # Output cuối cùng của lớp Encoder này
        return norm2_output

# -------------------------------------
# Encoder tổng thể
# -------------------------------------

class Encoder:
    """
    Encoder của mô hình Transformer, bao gồm nhiều lớp Encoder xếp chồng lên nhau.
    """
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, max_seq_len, dropout_rate=0.1):
        """
        Khởi tạo Encoder.

        Args:
            num_layers (int): Số lượng lớp Encoder (ví dụ: 6).
            d_model (int): Kích thước embedding.
            num_heads (int): Số lượng attention heads.
            d_ff (int): Kích thước lớp ẩn trong Feed-Forward network.
            input_vocab_size (int): Kích thước từ vựng đầu vào (dùng cho embedding).
            max_seq_len (int): Độ dài tối đa của chuỗi (dùng cho positional encoding).
            dropout_rate (float): Tỷ lệ dropout (bỏ qua trong demo).
        """
        self.d_model = d_model
        self.num_layers = num_layers
        # self.dropout_rate = dropout_rate # Bỏ qua dropout

        # 1. Input Embedding (Giả lập)
        # Trong thực tế, đây là một lớp học được (nn.Embedding trong PyTorch/TensorFlow)
        # Ở đây, ta tạo một ma trận giả lập.
        # Kích thước (input_vocab_size, d_model)
        self.embedding_matrix = [[(0.01 * (i * d_model + j)) % 1 for j in range(d_model)]
                                 for i in range(input_vocab_size)]

        # 2. Positional Encoding
        self.pos_encoding_matrix = positional_encoding(max_seq_len, d_model)

        # 3. Tạo danh sách các lớp Encoder
        # Sử dụng copy.deepcopy để đảm bảo mỗi lớp có trọng số riêng (mặc dù khởi tạo giống nhau ở đây)
        self.encoder_layers = [
            copy.deepcopy(EncoderLayer(d_model, num_heads, d_ff, dropout_rate))
            for _ in range(num_layers)
        ]

    def forward(self, input_sequence, mask=None):
        """
        Tính toán forward pass cho toàn bộ Encoder.

        Args:
            input_sequence (list[int]): Chuỗi ID token đầu vào, shape (seq_len,).
                                        Mỗi phần tử là một index trong từ vựng.
            mask (list[list[bool]], optional): Mask cho self-attention trong tất cả các lớp.
                                               Thường dùng để ẩn padding tokens. Shape (seq_len, seq_len).

        Returns:
            list[list[float]]: Output của Encoder, shape (seq_len, d_model).
        """
        seq_len = len(input_sequence)
        if seq_len == 0:
            return [] # Trả về rỗng nếu input rỗng

        # 1. Lấy Input Embeddings từ ma trận embedding
        # input_sequence chứa các index, ta cần lấy các vector tương ứng
        try:
            embedded_input = [self.embedding_matrix[token_id] for token_id in input_sequence]
        except IndexError:
             raise ValueError("Token ID trong input_sequence nằm ngoài phạm vi từ vựng")
        # embedded_input shape: (seq_len, d_model)

        # 2. Cộng Positional Encoding
        # Lấy phần tương ứng của positional encoding matrix
        # Giả định seq_len <= max_seq_len
        if seq_len > len(self.pos_encoding_matrix):
             raise ValueError(f"Độ dài chuỗi ({seq_len}) dài hơn max_seq_len ({len(self.pos_encoding_matrix)})")
        pos_enc_part = self.pos_encoding_matrix[:seq_len] # Lấy các hàng đầu tiên

        # Cộng embedding và positional encoding
        x = matrix_add(embedded_input, pos_enc_part)
        # x shape: (seq_len, d_model)
        # Lưu ý: Dropout thường được áp dụng ở đây trong mô hình gốc

        # 3. Đưa qua các lớp Encoder tuần tự
        for i in range(self.num_layers):
            x = self.encoder_layers[i].forward(x, mask)
            # Output của mỗi lớp trở thành input cho lớp tiếp theo
            # x shape vẫn là (seq_len, d_model)

        # Output cuối cùng của Encoder
        return x

# -------------------------------------
# Ví dụ sử dụng
# -------------------------------------

if __name__ == "__main__":
    # --- Định nghĩa các tham số cho mô hình ---
    NUM_LAYERS = 2       # Số lớp Encoder (ít hơn 6 để chạy nhanh hơn)
    D_MODEL = 8         # Kích thước embedding (nhỏ để dễ nhìn)
    NUM_HEADS = 2        # Số attention heads (phải chia hết D_MODEL)
    D_FF = 16           # Kích thước lớp ẩn Feed-Forward (thường lớn hơn D_MODEL)
    INPUT_VOCAB_SIZE = 100 # Kích thước từ vựng giả lập
    MAX_SEQ_LEN = 20     # Độ dài chuỗi tối đa cho positional encoding
    DROPOUT_RATE = 0.1   # (Không dùng trong demo này)

    # --- Tạo một chuỗi đầu vào giả lập ---
    # Giả sử chuỗi có độ dài 5 tokens
    input_seq_ids = [10, 25, 5, 42, 3] # IDs của các token trong từ vựng
    seq_len = len(input_seq_ids)

    # --- Tạo mask (tùy chọn) ---
    # Ví dụ: không có padding, không cần mask
    # Nếu có padding, ví dụ token cuối là padding:
    # input_seq_ids = [10, 25, 5, 42, 0] # Giả sử 0 là padding ID
    # seq_len = 5
    # padding_mask = [[False, False, False, False, True]] # Shape (1, seq_len)
    # Transformer thường cần mask shape (batch_size, seq_len, seq_len) hoặc (seq_len, seq_len)
    # Ở đây ta làm đơn giản với self-attention mask không che gì cả (False hết)
    # Hoặc có thể tạo mask che padding nếu cần.
    # Ví dụ đơn giản: không mask gì cả
    encoder_mask = [[False for _ in range(seq_len)] for _ in range(seq_len)]
    print(f"Kích thước mask: {shape(encoder_mask)}")

    # --- Khởi tạo Encoder ---
    print("Khởi tạo Encoder...")
    encoder = Encoder(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        input_vocab_size=INPUT_VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        dropout_rate=DROPOUT_RATE
    )
    print("Encoder đã được khởi tạo.")

    # --- Thực hiện forward pass qua Encoder ---
    print(f"\nInput sequence IDs: {input_seq_ids} (Độ dài: {seq_len})")
    print("Thực hiện forward pass qua Encoder...")
    try:
        encoder_output = encoder.forward(input_seq_ids, encoder_mask)
        print("Forward pass hoàn thành.")

        # --- In kết quả ---
        print(f"\nOutput của Encoder có shape: {shape(encoder_output)}")
        print("Output (mỗi hàng là vector biểu diễn cho một token đầu vào):")
        for i, row in enumerate(encoder_output):
            # Làm tròn để dễ đọc hơn
            rounded_row = [round(val, 3) for val in row]
            print(f"  Token {i}: {rounded_row}")

    except ValueError as e:
        print(f"\nLỗi xảy ra: {e}")
    except Exception as e:
         print(f"\nLỗi không xác định: {e}")