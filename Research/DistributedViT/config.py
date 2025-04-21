# config.py
"""
Stores hyperparameters and configuration settings for the ViT training.
"""

# Data and Patch Parameters
image_size = 32  # CIFAR-10 image size (32x32 pixels)
patch_size = 4   # Kích thước (chiều cao/rộng) của mỗi patch
num_classes = 10 # Số lớp đầu ra (CIFAR-10)
data_root = './data' # Root directory for datasets

# Calculated dimensions (can be derived, but defining helps clarity)
num_patches = (image_size // patch_size) ** 2
patch_dim = 3 * patch_size * patch_size # 3 channels * patch_height * patch_width

# Vision Transformer Architecture Parameters
d_model = 128     # Embedding dimension
num_heads = 8     # Number of attention heads (d_model must be divisible by num_heads)
num_layers = 6    # Number of Transformer Encoder layers
ffn_dim = 512     # Hidden dimension in Feed-Forward Networks
dropout = 0.1     # Dropout rate

# Training Parameters
epochs = 10           # Number of training epochs
batch_size = 64       # Batch size PER PROCESS (effective global batch size = batch_size * world_size)
learning_rate = 0.001 # Base learning rate (might need scaling in distributed setup)
num_workers = 2       # Number of data loading workers per process

# Distributed Training Parameters (Defaults, can be overridden by launcher)
master_addr = 'localhost'
master_port = '12355'

# Logging and Checkpointing
log_interval = 100 # Log training status every N batches
checkpoint_dir = './checkpoints' # Directory to save model checkpoints