import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
from tqdm.auto import tqdm # Use auto version for notebook/script compatibility
import math

# Function to get CIFAR-100 class names (needed for plotting)
def get_cifar100_class_names(data_dir='./data_cifar100'):
    """Loads CIFAR-100 dataset instance just to get class names."""
    try:
        # Temporarily load dataset to access class names
        from torchvision.datasets import CIFAR100
        # Ensure download=False if you've already downloaded it,
        # or True if you want it to download if missing.
        # We don't need transforms here.
        temp_dataset = CIFAR100(root=data_dir, train=False, download=True)
        return temp_dataset.classes
    except Exception as e:
        print(f"Could not load CIFAR-100 to get class names: {e}")
        # Fallback if loading fails (less informative plots)
        return [str(i) for i in range(100)]

# Function to unnormalize and display an image tensor
def imshow(inp, title=None, mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(mean)
    std = np.array(std)
    inp = std * inp + mean # Unnormalize
    inp = np.clip(inp, 0, 1) # Clip values to [0, 1]
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

def evaluate_model(model, test_loader, device, num_classes, class_names=None):
    """Performs detailed evaluation on the test set."""
    model.eval()
    all_preds = []
    all_labels = []
    all_outputs = [] # Store raw outputs for Top-5

    print("Evaluating on test set...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader): # Add progress bar
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_outputs.extend(outputs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_outputs = np.array(all_outputs)

    # --- Calculate Metrics ---
    # Top-1 Accuracy
    top1_accuracy = accuracy_score(all_labels, all_preds)

    # Top-5 Accuracy
    top5_preds_indices = np.argsort(all_outputs, axis=1)[:, -5:]
    correct_top5 = np.array([all_labels[i] in top5_preds_indices[i] for i in range(len(all_labels))])
    top5_accuracy = correct_top5.mean()

    # Precision, Recall, F1-Score (Weighted and Macro)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    # Classification Report (provides per-class metrics)
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print("\nClassification Report:")
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    metrics = {
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm
    }

    print(f"\nEvaluation Summary:")
    print(f"  Top-1 Accuracy: {top1_accuracy:.4f}")
    print(f"  Top-5 Accuracy: {top5_accuracy:.4f}")
    print(f"  Macro Avg - Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1-Score: {f1_macro:.4f}")
    print(f"  Weighted Avg - Precision: {precision_weighted:.4f}, Recall: {recall_weighted:.4f}, F1-Score: {f1_weighted:.4f}")

    return metrics


def plot_confusion_matrix(cm, class_names, figsize=(20, 20), filename='confusion_matrix.png', normalize=False):
    """Plots the confusion matrix as a heatmap."""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        print("Normalized confusion matrix")
    else:
        fmt = 'd'
        print('Confusion matrix, without normalization')

    plt.style.use('seaborn-v0_8-whitegrid') # Use a seaborn style
    plt.figure(figsize=figsize)
    # Use annot=False for large matrices like 100x100 as numbers overlap
    sns.heatmap(cm, annot=False, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title('Confusion Matrix')
    plt.tight_layout() # Adjust layout
    try:
        plt.savefig(filename)
        print(f"Confusion matrix saved to {filename}")
    except Exception as e:
        print(f"Error saving confusion matrix: {e}")
    plt.show() # Optionally display plot immediately


def visualize_misclassified(model, test_loader, device, class_names, num_images=25, filename_prefix='misclassified'):
    """Visualizes some misclassified images."""
    model.eval()
    misclassified_examples = []
    count = 0
    target_count = num_images

    print(f"\nSearching for {target_count} misclassified images...")
    # Need original images for better visualization, requires modifying loader or dataset
    # For simplicity, we'll unnormalize the tensor from the loader here.
    # Get mean/std used by the loader (assuming they are standard)
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs_device, labels_device = inputs.to(device), labels.to(device)
            outputs = model(inputs_device)
            _, predicted = torch.max(outputs.data, 1)

            misclassified_mask = (predicted != labels_device)
            misclassified_indices = torch.where(misclassified_mask)[0]

            for idx in misclassified_indices:
                if count < target_count:
                    img = inputs[idx].cpu() # Get original tensor from batch
                    true_label = labels[idx].item()
                    pred_label = predicted[idx].item()
                    misclassified_examples.append((img, true_label, pred_label))
                    count += 1
                else:
                    break # Stop searching once enough examples are found
            if count >= target_count:
                break

    if not misclassified_examples:
        print("No misclassified images found.")
        return

    print(f"Visualizing {len(misclassified_examples)} misclassified images...")

    # Determine grid size
    num_rows = int(math.sqrt(len(misclassified_examples)))
    num_cols = math.ceil(len(misclassified_examples) / num_rows)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    fig.suptitle('Misclassified Images (True vs. Predicted)', fontsize=16)
    axes = axes.flatten() # Flatten to 1D array for easy iteration

    for i, (img, true_idx, pred_idx) in enumerate(misclassified_examples):
        ax = axes[i]
        # Unnormalize for display
        img_display = img.numpy().transpose((1, 2, 0))
        img_display = cifar100_std * img_display + cifar100_mean
        img_display = np.clip(img_display, 0, 1)

        ax.imshow(img_display)
        true_name = class_names[true_idx] if class_names else f"Class {true_idx}"
        pred_name = class_names[pred_idx] if class_names else f"Class {pred_idx}"
        ax.set_title(f"True: {true_name}\nPred: {pred_name}", fontsize=9)
        ax.axis('off') # Hide axes ticks

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    filename = f"{filename_prefix}_{len(misclassified_examples)}.png"
    try:
        plt.savefig(filename)
        print(f"Misclassified images visualization saved to {filename}")
    except Exception as e:
        print(f"Error saving misclassified images plot: {e}")
    plt.show()