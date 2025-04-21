# model.py
"""
Defines the Vision Transformer model architecture and its components.
"""
import torch
import torch.nn as nn

# --- Patch Embedding ---
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, patch_dim, d_model, dropout_rate):
        super().__init__()
        if image_size % patch_size != 0:
             raise ValueError("Image dimensions must be divisible by the patch size.")
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Linear(patch_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        # Create patches: (B, C, H, W) -> (B, C, N_h, p_h, N_w, p_w)
        x = x.view(batch_size, channels, height // self.patch_size, self.patch_size, width // self.patch_size, self.patch_size)
        # Permute: (B, N_h, N_w, p_h, p_w, C)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        # Flatten patches: (B, N_h * N_w, p_h * p_w * C) = (B, num_patches, patch_dim)
        x = x.view(batch_size, self.num_patches, -1)

        x = self.projection(x)  # (B, num_patches, d_model)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # (B, 1, d_model)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, d_model)

        x = x + self.pos_embedding # Add positional embedding
        x = self.dropout(x)
        return x

# --- Multi-Head Self-Attention ---
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, n_heads, seq_len, head_dim)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale # (B, n_heads, seq_len, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = (attn_weights @ v).transpose(1, 2).reshape(batch_size, seq_len, self.d_model) # (B, seq_len, d_model)
        out = self.out(attn_output)
        # Note: Original notebook applied dropout after self.out, keeping it here
        out = self.dropout(out)
        return out

# --- Feed-Forward Network ---
class FeedForward(nn.Module):
    def __init__(self, d_model, ffn_dim, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.dropout = nn.Dropout(dropout_rate) # Applied twice in original notebook

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x) # First dropout
        x = self.fc2(x)
        x = self.dropout(x) # Second dropout
        return x

# --- Transformer Encoder Layer ---
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, dropout_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout_rate)
        # Note: Original notebook applied dropout within Attention and FFN, and also after adding residual.
        # Standard Transformer usually applies dropout *before* adding residual.
        # Let's keep the original notebook's structure for now (dropout inside submodules).

    def forward(self, x):
        # Pre-Normalization variation is sometimes used, but let's stick to Post-Norm like original
        # Attention block
        residual = x
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm)
        x = residual + attn_out # Add residual connection

        # Feed-Forward block
        residual = x
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = residual + ffn_out # Add residual connection
        return x


# --- Vision Transformer Model ---
class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, patch_dim, d_model, num_heads, num_layers, ffn_dim, num_classes, dropout_rate):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, patch_dim, d_model, dropout_rate)
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, ffn_dim, dropout_rate)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model) # Final normalization
        self.head = nn.Linear(d_model, num_classes) # Classification head

    def forward(self, x):
        x = self.patch_embed(x) # (B, num_patches+1, d_model)
        for layer in self.encoder:
            x = layer(x)
        x = self.norm(x)
        cls_token = x[:, 0] # Extract CLS token embedding (B, d_model)
        out = self.head(cls_token) # (B, num_classes)
        return out