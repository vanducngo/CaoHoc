# On Computer 1 (Node 0 - Master):
# Set the IP address of THIS machine (Node 0)
export MASTER_ADDR=192.168.1.100
# Set the agreed-upon port
export MASTER_PORT=12355

echo "Starting Node 0 (Master)..."
# nnodes=2: Total number of computers
# nproc_per_node=4: Number of CPU processes on this computer
# node_rank=0: This is the first computer (master)
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
train_distributed.py --epochs=5 --batch-size=32 --lr=0.001 --scale-lr --log-interval=50
# Adjust batch-size, epochs, lr, log-interval etc. as needed for CPU training
# Lower batch-size might be necessary for CPU memory
# --scale-lr is generally still recommended

# On Computer 2 (Node 1):
# Set the IP address of Node 0 (the MASTER node)
export MASTER_ADDR=192.168.1.100
# Set the SAME port used on Node 0
export MASTER_PORT=12355

echo "Starting Node 1..."
# nnodes=2: Total number of computers
# nproc_per_node=4: Number of CPU processes on this computer
# node_rank=1: This is the second computer
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
train_distributed.py --epochs=5 --batch-size=32 --lr=0.001 --scale-lr --log-interval=50
# Use the EXACT SAME arguments for the script as on Node 0



