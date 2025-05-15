# train_distributed.py
"""
Main script for distributed training of the Vision Transformer on CIFAR-10.
Uses torchrun for launching.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Import components from other files
import config
from model import VisionTransformer
from dataset import get_dataloaders

def setup(rank, world_size):
    """Initializes the distributed environment."""
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', config.master_addr)
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', config.master_port)

    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    try:
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        print(f"Rank {rank}/{world_size}: Initialized process group using {backend} backend.")
    except Exception as e:
        print(f"Rank {rank}: Failed to initialize process group: {e}")
        raise

def cleanup():
    """Cleans up the distributed environment."""
    dist.destroy_process_group()
    print("Cleaned up process group.")

def train_epoch(model, loader, sampler, optimizer, criterion, device, epoch, rank, world_size, log_interval):
    """Runs one epoch of training."""
    model.train()
    sampler.set_epoch(epoch) # Ensure shuffling varies across epochs
    total_loss = torch.tensor(0.0).to(device) # Use tensor for potential reduction
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # --- Logging (aggregate loss across ranks for more accurate reporting) ---
        batch_loss = loss.detach() # Detach from graph
        dist.all_reduce(batch_loss, op=dist.ReduceOp.AVG) # Average loss across all processes
        total_loss += batch_loss # Add averaged loss

        # --- Accuracy calculation (can be done per rank or aggregated) ---
        # Let's calculate per rank for simplicity during training logging
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0) # This is per-rank total
        correct += (predicted == labels).sum().item() # This is per-rank correct

        if (batch_idx + 1) % log_interval == 0 and rank == 0:
            avg_loss = total_loss.item() / (batch_idx + 1)
            # Note: This accuracy is based on rank 0's subset unless aggregated
            # For a quick progress check, rank 0's accuracy is often sufficient.
            # To get global accuracy, you'd need another all_reduce for 'correct' and 'total'.
            current_acc = 100.0 * correct / total
            print(f'Train Epoch: {epoch} [{batch_idx * len(images) * world_size}/{len(loader.dataset)} '\
                  f'({100. * batch_idx / len(loader):.0f}%)]\tAvg Loss: {avg_loss:.6f} Rank 0 Acc: {current_acc:.2f}%')

    # Return average loss across all batches for the epoch (already averaged across ranks)
    epoch_avg_loss = total_loss.item() / len(loader)
    # Return Rank 0's accuracy - for full accuracy, aggregation is needed.
    epoch_acc = 100.0 * correct / total
    return epoch_avg_loss, epoch_acc


def evaluate(model, loader, criterion, device, rank):
    """Evaluates the model on the test set (only runs on Rank 0)."""
    if rank != 0:
        return 0.0 # Only rank 0 evaluates

    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            # Access underlying model using model.module when using DDP
            outputs = model.module(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    print(f'\nRank 0 Test set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    return accuracy

def save_checkpoint(model, optimizer, epoch, path):
    """Saves model checkpoint (only on Rank 0)."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Save the underlying model's state dict
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved to {path}")

def main():
    parser = argparse.ArgumentParser(description='Distributed ViT Training on CIFAR-10')
    # Add arguments to potentially override config.py defaults
    parser.add_argument('--batch-size', type=int, default=config.batch_size, metavar='N',
                        help=f'input batch size for training (default: {config.batch_size}) PER PROCESS')
    parser.add_argument('--epochs', type=int, default=config.epochs, metavar='N',
                        help=f'number of epochs to train (default: {config.epochs})')
    parser.add_argument('--lr', type=float, default=config.learning_rate, metavar='LR',
                        help=f'base learning rate (default: {config.learning_rate})')
    parser.add_argument('--scale-lr', action='store_true', default=False,
                        help='scale learning rate by world size (recommended)')
    parser.add_argument('--data-root', type=str, default=config.data_root,
                        help=f'path to dataset (default: {config.data_root})')
    parser.add_argument('--num-workers', type=int, default=config.num_workers,
                        help=f'number of data loading workers (default: {config.num_workers})')
    parser.add_argument('--log-interval', type=int, default=config.log_interval,
                        help=f'how many batches to wait before logging training status (default: {config.log_interval})')
    parser.add_argument('--checkpoint-dir', type=str, default=config.checkpoint_dir,
                        help=f'directory to save checkpoints (default: {config.checkpoint_dir})')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    args = parser.parse_args()

    # --- Distributed Setup ---
    # torchrun sets these environment variables
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK']) # Rank of process on the current node

    setup(rank, world_size)

    # --- Reproducibility ---
    torch.manual_seed(args.seed + rank) # Add rank for different initializations if desired
    np.random.seed(args.seed + rank)
    # Consider torch.backends.cudnn.deterministic = True for more determinism (can slow down)

    # --- Device Setup ---
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"Rank {rank}/{world_size}: Using GPU {local_rank}.")
    else:
        device = torch.device("cpu")
        print(f"Rank {rank}/{world_size}: Using CPU.")

    # --- Data Loading ---
    train_loader, test_loader, train_sampler = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        world_size=world_size,
        rank=rank
    )

    # --- Model Initialization ---
    model = VisionTransformer(
        image_size=config.image_size,
        patch_size=config.patch_size,
        patch_dim=config.patch_dim,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ffn_dim=config.ffn_dim,
        num_classes=config.num_classes,
        dropout_rate=config.dropout
    ).to(device)

    # --- Wrap model with DDP ---
    # find_unused_parameters=True might be needed if some model outputs aren't used in loss
    # (shouldn't be necessary for this standard ViT)
    model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None,
                output_device=local_rank if torch.cuda.is_available() else None)
    if rank == 0:
      print("Model wrapped with DistributedDataParallel.")
      # You can print model summary here if needed (e.g., using torchinfo)
      # from torchinfo import summary
      # summary(model.module, input_size=(args.batch_size, 3, config.image_size, config.image_size))


    # --- Loss and Optimizer ---
    criterion = nn.CrossEntropyLoss().to(device)

    learning_rate = args.lr
    if args.scale_lr:
        learning_rate = args.lr * world_size
        if rank == 0:
            print(f"Scaling learning rate: {args.lr} * {world_size} = {learning_rate}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Training Loop ---
    if rank == 0:
        print("\nStarting Training...")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, train_sampler, optimizer, criterion, device, epoch, rank, world_size, args.log_interval
        )

        # Evaluate on rank 0 after each epoch
        test_acc = evaluate(model, test_loader, criterion, device, rank)

        # Log results (only rank 0)
        if rank == 0:
            print(f'--- Epoch {epoch}/{args.epochs} Summary ---')
            print(f'Train Loss: {train_loss:.4f}, Train Acc (Rank 0): {train_acc:.2f}%')
            print(f'Test Accuracy (Rank 0): {test_acc:.2f}%')
            print('------------------------------------\n')

            # Save checkpoint (only rank 0)
            checkpoint_path = os.path.join(args.checkpoint_dir, f"vit_cifar10_epoch_{epoch}.pt")
            save_checkpoint(model, optimizer, epoch, checkpoint_path)

        # Barrier to ensure all processes finish the epoch before starting the next one
        # or before rank 0 potentially exits if training finishes early.
        dist.barrier()


    # --- Cleanup ---
    if rank == 0:
      print("Training finished.")
    cleanup()

if __name__ == '__main__':
    main()