# dataset.py
"""
Handles dataset loading, transformations, and distributed data loading.
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def get_dataloaders(data_root, batch_size, num_workers, world_size, rank):
    """
    Creates training and test dataloaders with distributed sampling for training.

    Args:
        data_root (str): Path to the dataset directory.
        batch_size (int): Batch size per process.
        num_workers (int): Number of subprocesses for data loading.
        world_size (int): Total number of processes in the distributed group.
        rank (int): Rank of the current process.

    Returns:
        tuple: (train_loader, test_loader, train_sampler)
               test_loader will be None for ranks != 0.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1]
    ])

    # --- Training Dataset and Loader ---
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform
    )
    # DistributedSampler ensures each process gets unique data slices
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Shuffle is handled by DistributedSampler
        num_workers=num_workers,
        pin_memory=True, # Improves data transfer speed to GPU
        sampler=train_sampler
    )

    # --- Test Dataset and Loader (only on Rank 0 for simplicity) ---
    test_loader = None
    if rank == 0:
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=transform
        )
        # No sampler needed for rank 0 eval, can use larger batch size
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size * 2, # Example: double batch size for eval
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        print(f"Rank 0: Test dataset loaded with {len(test_dataset)} samples.")


    print(f"Rank {rank}: Train dataset loaded with {len(train_dataset)} samples (sampler assigns subset).")

    return train_loader, test_loader, train_sampler