"""
Deep Learning with Differential Privacy - DP-SGD on MNIST
==========================================================
Implementation based on:
  M. Abadi et al., "Deep learning with differential privacy,"
  Proceedings of the 2016 ACM SIGSAC CCS, 2016.

This script implements Differentially Private Stochastic Gradient Descent (DP-SGD)
using PyTorch and the Opacus library. It trains a CNN on MNIST under various
privacy budgets (epsilon) and compares accuracy vs. privacy trade-offs.

Key DP-SGD concepts:
  1. Per-sample gradient clipping: Each sample's gradient is clipped to a max norm C,
     bounding any single sample's influence on the model update.
  2. Gaussian noise addition: Calibrated noise N(0, sigma^2 * C^2 * I) is added to
     the aggregated gradient, providing (epsilon, delta)-differential privacy.
  3. Privacy accounting: The Renyi Differential Privacy (RDP) accountant tracks the
     cumulative privacy cost across training steps.

Usage:
  python dp_sgd_mnist.py
"""

import os
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

warnings.filterwarnings("ignore")

# ============================================================
# 1. Hyperparameters and Configuration
# ============================================================

# List of target epsilon values to experiment with.
# Smaller epsilon = stronger privacy but lower accuracy.
# Larger epsilon = weaker privacy but higher accuracy.
EPSILON_VALUES = [1.0, 2.0, 4.0, 8.0]

# Delta: probability of privacy guarantee failing.
# Set to less than 1/N where N is the training set size (60000 for MNIST).
DELTA = 1e-5

# Training hyperparameters
EPOCHS = 5               # Number of training epochs per experiment
BATCH_SIZE = 256          # Lot size (L in the paper)
LEARNING_RATE = 0.1       # SGD learning rate (eta in the paper)
MAX_GRAD_NORM = 1.0       # Gradient clipping bound (C in the paper)

# Data directory for MNIST
DATA_DIR = "./data"

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. CNN Model Definition
# ============================================================

class MNIST_CNN(nn.Module):
    """
    A simple CNN for MNIST digit classification.
    Architecture follows the paper's setup:
      - Conv(16, 8x8, stride=2, padding=2) + ReLU + MaxPool(2x2)
      - Conv(32, 4x4, stride=2, padding=0) + ReLU + MaxPool(2x2)
      - Fully connected 32 -> 10 (output logits)

    Note: Opacus requires replacing BatchNorm with GroupNorm
    and ensuring all layers are compatible with per-sample gradients.
    """

    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.features = nn.Sequential(
            # First convolution layer: 1 input channel, 16 filters, 8x8 kernel
            nn.Conv2d(1, 16, kernel_size=8, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),

            # Second convolution layer: 16 input channels, 32 filters, 4x4 kernel
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        # Flatten and classify
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 10),  # 10 classes for digits 0-9
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# 3. Data Loading
# ============================================================

def get_data_loaders(batch_size):
    """
    Load and preprocess the MNIST dataset.
    - Training set: 60,000 images
    - Test set: 10,000 images
    - Normalization: mean=0.1307, std=0.3081 (MNIST statistics)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        DATA_DIR, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        DATA_DIR, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,  # Required by Opacus for uniform batch sizes
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader, len(train_dataset)


# ============================================================
# 4. Training Functions
# ============================================================

def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """
    Train the model for one epoch.

    In DP-SGD mode (when Opacus PrivacyEngine is attached), each step:
      1. Computes per-sample gradients
      2. Clips each gradient to MAX_GRAD_NORM
      3. Aggregates clipped gradients
      4. Adds calibrated Gaussian noise
      5. Updates model parameters
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass and optimization
        # (Opacus hooks handle gradient clipping and noise addition automatically)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the model on the test set.
    Returns average loss and accuracy.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ============================================================
# 5. Training with Differential Privacy (DP-SGD)
# ============================================================

def train_with_dp(target_epsilon, delta, epochs, batch_size, lr, max_grad_norm, device):
    """
    Train the CNN on MNIST using DP-SGD with a target epsilon.

    The Opacus PrivacyEngine:
      - Wraps the model to compute per-sample gradients
      - Wraps the optimizer to clip gradients and add noise
      - Uses the RDP accountant to compute the actual (epsilon, delta) spent

    Args:
        target_epsilon: Target privacy budget (epsilon)
        delta: Privacy parameter delta
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        max_grad_norm: Maximum gradient norm for clipping (C)
        device: Torch device (CPU/GPU)

    Returns:
        Dictionary with training history and final metrics
    """
    print(f"\n{'='*60}")
    print(f"  Training with target epsilon = {target_epsilon}")
    print(f"{'='*60}")

    # Load data
    train_loader, test_loader, n_train = get_data_loaders(batch_size)

    # Initialize model (Opacus requires GN instead of BN)
    model = MNIST_CNN().to(device)
    model = ModuleValidator.fix(model)  # Fix any incompatible layers

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Attach Opacus PrivacyEngine
    # This modifies the model and optimizer to perform DP-SGD
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=delta,
        max_grad_norm=max_grad_norm,
    )

    # Print the noise multiplier (sigma) computed by Opacus
    print(f"  Noise multiplier (sigma): {optimizer.noise_multiplier:.4f}")
    print(f"  Delta: {delta}")
    print(f"  Max gradient norm (C): {max_grad_norm}")
    print(f"  Batch size (L): {batch_size}")
    print(f"  Training samples: {n_train}")
    print(f"  Sampling rate (q = L/N): {batch_size/n_train:.4f}")
    print()

    # Training loop
    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": [],
        "epsilon": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Get current epsilon spent (privacy budget consumed so far)
        current_epsilon = privacy_engine.get_epsilon(delta=delta)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["epsilon"].append(current_epsilon)

        print(f"  Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}% | epsilon: {current_epsilon:.2f}")

    final_result = {
        "target_epsilon": target_epsilon,
        "actual_epsilon": history["epsilon"][-1],
        "delta": delta,
        "noise_multiplier": optimizer.noise_multiplier,
        "final_test_accuracy": history["test_acc"][-1],
        "final_train_accuracy": history["train_acc"][-1],
        "history": history,
    }

    print(f"\n  Final Test Accuracy: {final_result['final_test_accuracy']:.2f}%")
    print(f"  Actual epsilon spent: {final_result['actual_epsilon']:.2f}")

    return final_result


# ============================================================
# 6. Baseline Training (No Privacy)
# ============================================================

def train_baseline(epochs, batch_size, lr, device):
    """
    Train the same CNN on MNIST WITHOUT differential privacy (baseline).
    This serves as the upper bound for accuracy comparison.
    """
    print(f"\n{'='*60}")
    print(f"  Training BASELINE (no differential privacy)")
    print(f"{'='*60}")

    train_loader, test_loader, n_train = get_data_loaders(batch_size)

    model = MNIST_CNN().to(device)
    model = ModuleValidator.fix(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"  Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}%")

    return {
        "target_epsilon": float("inf"),
        "final_test_accuracy": history["test_acc"][-1],
        "final_train_accuracy": history["train_acc"][-1],
        "history": history,
    }


# ============================================================
# 7. Visualization
# ============================================================

def plot_results(all_results, baseline_result, save_path="results"):
    """
    Generate plots comparing accuracy across different epsilon values.
    Produces:
      1. Bar chart: Test accuracy vs. epsilon
      2. Line chart: Test accuracy over epochs for each epsilon
    """
    os.makedirs(save_path, exist_ok=True)

    # --- Plot 1: Test Accuracy vs. Epsilon (bar chart) ---
    fig, ax = plt.subplots(figsize=(8, 5))

    epsilons = [r["target_epsilon"] for r in all_results]
    accuracies = [r["final_test_accuracy"] for r in all_results]
    labels = [f"eps={e}" for e in epsilons]

    # Add baseline
    labels.append("No DP\n(Baseline)")
    accuracies.append(baseline_result["final_test_accuracy"])

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(labels)))
    bars = ax.bar(labels, accuracies, color=colors, edgecolor="black", linewidth=0.8)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Privacy Budget (epsilon)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Impact of Differential Privacy on Model Accuracy (MNIST)", fontsize=13)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "accuracy_vs_epsilon.png"), dpi=150)
    plt.close()
    print(f"\n  Saved: {save_path}/accuracy_vs_epsilon.png")

    # --- Plot 2: Accuracy curves over epochs ---
    fig, ax = plt.subplots(figsize=(8, 5))

    for result in all_results:
        eps = result["target_epsilon"]
        acc_history = result["history"]["test_acc"]
        ax.plot(range(1, len(acc_history) + 1), acc_history,
                marker="o", label=f"epsilon={eps}", linewidth=2)

    # Baseline curve
    baseline_acc = baseline_result["history"]["test_acc"]
    ax.plot(range(1, len(baseline_acc) + 1), baseline_acc,
            marker="s", label="No DP (Baseline)", linewidth=2, linestyle="--", color="black")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Test Accuracy Over Epochs at Different Privacy Levels", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "accuracy_over_epochs.png"), dpi=150)
    plt.close()
    print(f"  Saved: {save_path}/accuracy_over_epochs.png")

    # --- Plot 3: Epsilon consumed over epochs ---
    fig, ax = plt.subplots(figsize=(8, 5))

    for result in all_results:
        eps_target = result["target_epsilon"]
        eps_history = result["history"]["epsilon"]
        ax.plot(range(1, len(eps_history) + 1), eps_history,
                marker="o", label=f"target eps={eps_target}", linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Epsilon Spent", fontsize=12)
    ax.set_title("Privacy Budget Consumption Over Training", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "epsilon_over_epochs.png"), dpi=150)
    plt.close()
    print(f"  Saved: {save_path}/epsilon_over_epochs.png")


# ============================================================
# 8. Main Experiment
# ============================================================

def main():
    """
    Main function: runs the full experiment.
    1. Train baseline model (no DP)
    2. Train DP-SGD models at different epsilon values
    3. Compare and visualize results
    """
    print("=" * 60)
    print("  Deep Learning with Differential Privacy - MNIST Experiment")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Epsilon values to test: {EPSILON_VALUES}")
    print(f"  Delta: {DELTA}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max gradient norm: {MAX_GRAD_NORM}")

    # Step 1: Train baseline (no DP)
    baseline_result = train_baseline(EPOCHS, BATCH_SIZE, LEARNING_RATE, DEVICE)

    # Step 2: Train with DP-SGD at various epsilon levels
    dp_results = []
    for eps in EPSILON_VALUES:
        result = train_with_dp(
            target_epsilon=eps,
            delta=DELTA,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            max_grad_norm=MAX_GRAD_NORM,
            device=DEVICE,
        )
        dp_results.append(result)

    # Step 3: Print summary table
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Setting':<20} {'Test Accuracy':>15} {'Epsilon':>10}")
    print(f"  {'-'*45}")
    print(f"  {'Baseline (No DP)':<20} {baseline_result['final_test_accuracy']:>14.2f}% {'inf':>10}")
    for r in dp_results:
        label = f"DP (eps={r['target_epsilon']})"
        print(f"  {label:<20} {r['final_test_accuracy']:>14.2f}% {r['actual_epsilon']:>10.2f}")

    # Step 4: Generate plots
    plot_results(dp_results, baseline_result, save_path="results")

    # Step 5: Save results to JSON for report generation
    summary = {
        "baseline": {
            "test_accuracy": baseline_result["final_test_accuracy"],
            "train_accuracy": baseline_result["final_train_accuracy"],
        },
        "dp_results": [
            {
                "target_epsilon": r["target_epsilon"],
                "actual_epsilon": r["actual_epsilon"],
                "noise_multiplier": r["noise_multiplier"],
                "test_accuracy": r["final_test_accuracy"],
                "train_accuracy": r["final_train_accuracy"],
                "delta": r["delta"],
            }
            for r in dp_results
        ],
        "config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_grad_norm": MAX_GRAD_NORM,
            "delta": DELTA,
        },
    }
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: results/experiment_results.json")

    print(f"\n{'='*60}")
    print("  Experiment complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
