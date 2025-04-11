import torch
import torch.nn as nn
import numpy as np
import math

# --- 1. Positional Encoding ---
# Thành phần này thêm thông tin về vị trí của token trong chuỗi.
# Nó không phải là một lớp học được (non-trainable).
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Args:
            d_model (int): Kích thước của vector embedding (số chiều).
            dropout (float): Tỷ lệ dropout được áp dụng sau khi thêm positional encoding.
            max_len (int): Độ dài tối đa của chuỗi mà mô hình có thể xử lý.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Tạo ma trận positional encoding tĩnh (không học)
        pe = torch.zeros(max_len, d_model) # Ma trận shape (max_len, d_model)

        # Tạo vector vị trí: [0, 1, 2, ..., max_len-1] -> shape (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Tính toán phần mẫu số của công thức positional encoding
        # div_term = 1 / (10000^(2i / d_model))
        # i là chỉ số chiều (từ 0 đến d_model/2 - 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # div_term shape: (d_model / 2)

        # Áp dụng hàm sin cho các chiều chẵn (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        # position * div_term có shape (max_len, d_model/2) do broadcasting

        # Áp dụng hàm cos cho các chiều lẻ (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)
        # position * div_term có shape (max_len, d_model/2)

        # Thêm một chiều batch vào đầu: (1, max_len, d_model)
        # Chúng ta lưu pe như một buffer (tham số không cần cập nhật gradient)
        # thay vì một tham số (parameter) của mô hình.
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input embedding, shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor sau khi thêm positional encoding,
                          shape (batch_size, seq_len, d_model).
        """
        # Lấy phần positional encoding tương ứng với độ dài chuỗi (seq_len) của input
        # self.pe[:, :x.size(1), :] có shape (1, seq_len, d_model)
        # Cộng positional encoding vào input embedding x
        # PyTorch sẽ tự động broadcast chiều batch (1) của pe lên batch_size của x
        x = x + self.pe[:, :x.size(1), :]
        # Áp dụng dropout
        return self.dropout(x)

# --- 2. Multi-Head Self-Attention ---
# Cơ chế cốt lõi cho phép mô hình cân nhắc các từ khác nhau trong chuỗi khi biểu diễn một từ cụ thể.
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Args:
            d_model (int): Kích thước embedding (phải chia hết cho num_heads).
            num_heads (int): Số lượng attention heads.
            dropout (float): Tỷ lệ dropout áp dụng cho attention weights.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model phải chia hết cho num_heads"

        self.d_model = d_model       # Tổng số chiều embedding
        self.num_heads = num_heads   # Số lượng attention heads
        self.d_k = d_model // num_heads # Số chiều của mỗi head (key/query/value)

        # Tạo các lớp Linear để chiếu (project) input vào Query, Key, Value và Output
        # Bias thường được đặt là False trong các triển khai Transformer chuẩn
        self.W_q = nn.Linear(d_model, d_model, bias=False) # Linear layer cho Query
        self.W_k = nn.Linear(d_model, d_model, bias=False) # Linear layer cho Key
        self.W_v = nn.Linear(d_model, d_model, bias=False) # Linear layer cho Value
        self.W_o = nn.Linear(d_model, d_model, bias=False) # Linear layer cho Output

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1) # Softmax trên chiều cuối cùng (key dimension)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Tính toán Scaled Dot-Product Attention.
        Args:
            Q (torch.Tensor): Query tensor, shape (batch_size, num_heads, seq_len_q, d_k)
            K (torch.Tensor): Key tensor, shape (batch_size, num_heads, seq_len_k, d_k)
            V (torch.Tensor): Value tensor, shape (batch_size, num_heads, seq_len_v, d_k)
                              (seq_len_k và seq_len_v thường bằng nhau)
            mask (torch.Tensor, optional): Mask để ẩn đi các vị trí không mong muốn (ví dụ: padding).
                                           Shape (batch_size, 1, 1, seq_len_k) hoặc (batch_size, 1, seq_len_q, seq_len_k)

        Returns:
            torch.Tensor: Output sau attention, shape (batch_size, num_heads, seq_len_q, d_k)
            torch.Tensor: Attention weights, shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        # 1. Tính Attention Scores: Q * K^T / sqrt(d_k)
        # Matmul giữa Q và K.T (K chuyển vị 2 chiều cuối)
        # Q:   (batch_size, num_heads, seq_len_q, d_k)
        # K.transpose(-2, -1): (batch_size, num_heads, d_k, seq_len_k)
        # scores: (batch_size, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 2. Áp dụng Mask (nếu có)
        # Mask thường dùng để che đi các token padding hoặc các token tương lai (trong decoder).
        if mask is not None:
            # Gán giá trị rất nhỏ (-inf) vào các vị trí được mask
            # để sau khi qua softmax, các vị trí này sẽ có giá trị gần bằng 0.
            scores = scores.masked_fill(mask == 0, -1e9) # Hoặc float('-inf')

        # 3. Tính Attention Weights bằng Softmax
        # Softmax được áp dụng trên chiều cuối cùng (seq_len_k)
        attn_weights = self.softmax(scores)
        attn_weights = self.dropout(attn_weights) # Áp dụng dropout cho attention weights

        # 4. Tính Output: Attention Weights * V
        # attn_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        # V:            (batch_size, num_heads, seq_len_v, d_k) (seq_len_v == seq_len_k)
        # output:       (batch_size, num_heads, seq_len_q, d_k)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

    def forward(self, query, key, value, mask=None):
        """
        Forward pass của Multi-Head Attention.
        Trong self-attention của Encoder, query, key, value đều là input x.
        Args:
            query (torch.Tensor): Input Query, shape (batch_size, seq_len_q, d_model)
            key (torch.Tensor): Input Key, shape (batch_size, seq_len_k, d_model)
            value (torch.Tensor): Input Value, shape (batch_size, seq_len_v, d_model)
            mask (torch.Tensor, optional): Mask, shape broadcast được tới
                                           (batch_size, 1, seq_len_q, seq_len_k)

        Returns:
            torch.Tensor: Output cuối cùng, shape (batch_size, seq_len_q, d_model)
            torch.Tensor: Attention weights, shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)

        # 1. Chiếu tuyến tính Query, Key, Value
        # Input: (batch_size, seq_len, d_model) -> Output: (batch_size, seq_len, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. Chia thành nhiều heads (num_heads)
        # Ban đầu: (batch_size, seq_len, d_model)
        # Dùng .view() để reshape thành (batch_size, seq_len, num_heads, d_k)
        # Dùng .transpose(1, 2) để đổi chiều thành (batch_size, num_heads, seq_len, d_k)
        # Việc này cần thiết để tính toán attention cho từng head song song.
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # seq_len_q, seq_len_k, seq_len_v được suy ra tự động từ -1 trong view

        # 3. Áp dụng Scaled Dot-Product Attention cho từng head
        # attention_output: (batch_size, num_heads, seq_len_q, d_k)
        # attn_weights:     (batch_size, num_heads, seq_len_q, seq_len_k)
        attention_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # 4. Ghép (Concatenate) các heads lại
        # attention_output: (batch_size, num_heads, seq_len_q, d_k)
        # Dùng .transpose(1, 2) để về lại (batch_size, seq_len_q, num_heads, d_k)
        # Dùng .contiguous() để đảm bảo tensor liền mạch trong bộ nhớ trước khi .view()
        # Dùng .view() để reshape về (batch_size, seq_len_q, d_model) (vì d_model = num_heads * d_k)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 5. Chiếu tuyến tính lần cuối (Final Linear Layer)
        # output: (batch_size, seq_len_q, d_model)
        output = self.W_o(attention_output)

        return output, attn_weights

# --- 3. Position-wise Feed-Forward Network ---
# Mạng nơ-ron feed-forward đơn giản được áp dụng độc lập cho từng vị trí (token).
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model (int): Kích thước input và output.
            d_ff (int): Kích thước lớp ẩn bên trong (thường lớn hơn d_model, ví dụ 2048).
            dropout (float): Tỷ lệ dropout.
        """
        super(PositionwiseFeedForward, self).__init__()
        # Lớp Linear thứ nhất: mở rộng chiều từ d_model -> d_ff
        self.fc1 = nn.Linear(512, 2048)
        # Lớp Linear thứ hai: thu hẹp chiều từ d_ff -> d_model
        self.fc2 = nn.Linear(2048, 512)
        # Hàm kích hoạt ReLU (thông dụng)
        self.relu = nn.ReLU()
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor, shape (batch_size, seq_len, d_model).
        """
        # Áp dụng lớp Linear 1 -> ReLU -> Dropout -> Linear 2
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

# --- 4. Encoder Layer ---
# Kết hợp Multi-Head Attention và Feed-Forward Network, cùng với Add & Norm.
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model (int): Kích thước embedding.
            num_heads (int): Số lượng attention heads.
            d_ff (int): Kích thước lớp ẩn trong Feed-Forward Network.
            dropout (float): Tỷ lệ dropout.
        """
        super(EncoderLayer, self).__init__()
        # Khởi tạo lớp Multi-Head Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # Khởi tạo lớp Position-wise Feed-Forward
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        # Khởi tạo 2 lớp Layer Normalization
        self.norm1 = nn.LayerNorm(d_model) # Norm sau attention
        self.norm2 = nn.LayerNorm(d_model) # Norm sau feed-forward
        # Khởi tạo 2 lớp Dropout
        # Dropout được áp dụng *trước* khi cộng vào residual connection và norm
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Forward pass của một lớp Encoder.
        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len, d_model).
            mask (torch.Tensor): Mask cho self-attention.

        Returns:
            torch.Tensor: Output tensor, shape (batch_size, seq_len, d_model).
        """
        # 1. Multi-Head Self-Attention
        # Input x được dùng làm query, key, value
        attn_output, _ = self.self_attn(x, x, x, mask)
        # attn_output shape: (batch_size, seq_len, d_model)

        # 2. Add & Norm 1 (Residual Connection + Layer Normalization)
        # Cộng input gốc (x) với output của attention (sau dropout)
        # Rồi áp dụng Layer Normalization
        x = self.norm1(x + self.dropout1(attn_output))
        # x shape: (batch_size, seq_len, d_model)

        # 3. Position-wise Feed-Forward Network
        ff_output = self.feed_forward(x)
        # ff_output shape: (batch_size, seq_len, d_model)

        # 4. Add & Norm 2 (Residual Connection + Layer Normalization)
        # Cộng input của feed-forward (x) với output của feed-forward (sau dropout)
        # Rồi áp dụng Layer Normalization
        x = self.norm2(x + self.dropout2(ff_output))
        # x shape: (batch_size, seq_len, d_model)

        return x

# --- 5. Encoder ---
# Toàn bộ khối Encoder, bao gồm lớp Embedding, Positional Encoding và nhiều Encoder Layers xếp chồng.
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout=0.1):
        """
        Args:
            vocab_size (int): Kích thước từ điển (số lượng token).
            d_model (int): Kích thước embedding.
            num_layers (int): Số lượng EncoderLayer xếp chồng.
            num_heads (int): Số lượng attention heads trong mỗi EncoderLayer.
            d_ff (int): Kích thước lớp ẩn trong Feed-Forward Network của mỗi EncoderLayer.
            max_len (int): Độ dài chuỗi tối đa cho Positional Encoding.
            dropout (float): Tỷ lệ dropout.
        """
        super(Encoder, self).__init__()
        self.d_model = d_model
        # Lớp Embedding để chuyển đổi ID token thành vector embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Lớp Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        # Tạo một danh sách chứa các lớp EncoderLayer
        # nn.ModuleList rất hữu ích khi chứa nhiều sub-modules giống nhau.
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers) # Tạo num_layers lớp EncoderLayer
        ])
        # (Tùy chọn) Một số kiến trúc có thêm LayerNorm cuối cùng sau khi qua tất cả các lớp
        # self.final_norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask):
        """
        Forward pass của toàn bộ Encoder.
        Args:
            src (torch.Tensor): Input token IDs, shape (batch_size, seq_len).
            src_mask (torch.Tensor): Mask cho input, shape thường là (batch_size, 1, 1, seq_len)
                                     để che padding tokens trong self-attention.

        Returns:
            torch.Tensor: Output của Encoder, shape (batch_size, seq_len, d_model).
        """
        # 1. Embedding + Scaling
        # Chuyển đổi token IDs thành embeddings
        # src_embed shape: (batch_size, seq_len, d_model)
        # Nhân với sqrt(d_model) theo đề xuất trong paper "Attention is All You Need"
        src_embed = self.embedding(src) * math.sqrt(self.d_model)

        # 2. Add Positional Encoding
        # Thêm thông tin vị trí vào embedding
        # output shape: (batch_size, seq_len, d_model)
        output = self.pos_encoding(src_embed)

        # 3. Pass through Encoder Layers
        # Cho dữ liệu đi qua từng lớp EncoderLayer trong danh sách self.layers
        for layer in self.layers:
            output = layer(output, src_mask)

        # (Tùy chọn) Áp dụng LayerNorm cuối cùng
        # output = self.final_norm(output)

        return output

# --- Ví dụ sử dụng ---
if __name__ == '__main__':
    # --- Hyperparameters ---
    vocab_size = 10000   # Kích thước từ điển ví dụ
    d_model = 512        # Kích thước embedding (chuẩn)
    num_layers = 6       # Số lớp Encoder (chuẩn)
    num_heads = 8        # Số attention heads (chuẩn, d_model phải chia hết cho num_heads)
    d_ff = 2048          # Kích thước lớp ẩn feed-forward (chuẩn)
    max_len = 100        # Độ dài chuỗi tối đa ví dụ
    dropout = 0.1

    # --- Dữ liệu đầu vào ví dụ ---
    batch_size = 32
    seq_len = 50         # Độ dài chuỗi thực tế (<= max_len)

    # Tạo dữ liệu input giả (IDs của token)
    # Giá trị từ 0 đến vocab_size-1
    src_tokens = torch.randint(1, vocab_size, (batch_size, seq_len)) # Bắt đầu từ 1 để 0 là padding
    # Giả sử 10 token cuối là padding (ID=0) cho một số câu trong batch
    src_tokens[0, seq_len-10:] = 0
    src_tokens[1, seq_len-5:] = 0

    # --- Tạo Mask ---
    # Mask rất quan trọng để self-attention bỏ qua các token padding.
    # Mask phải có shape phù hợp để broadcast với attention scores
    # Attention scores shape: (batch_size, num_heads, seq_len, seq_len)
    # Mask shape thường là:   (batch_size, 1, 1, seq_len)
    # Giá trị 1 cho vị trí hợp lệ, 0 cho vị trí cần mask (padding)
    src_padding_mask = (src_tokens != 0).unsqueeze(1).unsqueeze(2)
    # Giải thích shape:
    # (src_tokens != 0) -> (batch_size, seq_len) [Boolean Tensor]
    # .unsqueeze(1)     -> (batch_size, 1, seq_len)
    # .unsqueeze(2)     -> (batch_size, 1, 1, seq_len)

    print("Shape của src_tokens:", src_tokens.shape)
    print("Shape của src_padding_mask:", src_padding_mask.shape)
    print("Ví dụ src_padding_mask[0]:\n", src_padding_mask[0]) # Sẽ thấy các giá trị False ở cuối

    # --- Khởi tạo và chạy Encoder ---
    encoder = Encoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)

    # Đưa dữ liệu qua Encoder
    encoder_output = encoder(src_tokens, src_padding_mask)

    # --- Kiểm tra Output ---
    print("\nShape của output từ Encoder:", encoder_output.shape)
    # Expected output shape: (batch_size, seq_len, d_model) -> (32, 50, 512)

    # Kiểm tra số lượng tham số
    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Tổng số tham số có thể huấn luyện trong Encoder: {total_params:,}")