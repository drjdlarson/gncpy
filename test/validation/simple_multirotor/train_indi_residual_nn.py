"""Train MLP for NN-augmented INDI residual prediction."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# Configuration
# ============================================================================
HIDDEN_DIMS = [64, 64, 64]  # 3 hidden layers, 64 neurons each
EPOCHS = 200
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 30
MODEL_NAME = "hifi_residual_mlp"

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(16)


# ============================================================================
# Model
# ============================================================================
class ResidualMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================================
# Load and prepare data
# ============================================================================
data_dir = Path(__file__).parent / "training_data"
print(f"Loading data from {data_dir}")

X = pd.read_csv(data_dir / "nn_inputs.csv").iloc[:, 2:].values.astype(np.float32)
y = pd.read_csv(data_dir / "nn_targets.csv").iloc[:, 2:].values.astype(np.float32)
print(f"  Samples: {len(X)}, Input dim: {X.shape[1]}, Output dim: {y.shape[1]}")

# Train/val split
n_val = int(len(X) * VAL_SPLIT)
idx = np.random.permutation(len(X))
X_train, X_val = X[idx[n_val:]], X[idx[:n_val]]
y_train, y_val = y[idx[n_val:]], y[idx[:n_val]]
print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

# Normalize (compute stats from training data only)
X_mean, X_std = X_train.mean(0), X_train.std(0)
y_mean, y_std = y_train.mean(0), y_train.std(0)
X_std[X_std < 1e-8] = 1.0
y_std[y_std < 1e-8] = 1.0

X_train_n = (X_train - X_mean) / X_std
X_val_n = (X_val - X_mean) / X_std
y_train_n = (y_train - y_mean) / y_std
y_val_n = (y_val - y_mean) / y_std

# Data loaders
train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train_n), torch.tensor(y_train_n)),
    batch_size=BATCH_SIZE,
    shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val_n), torch.tensor(y_val_n)), batch_size=BATCH_SIZE
)

# ============================================================================
# Create and train model
# ============================================================================
model = ResidualMLP(X.shape[1], y.shape[1], HIDDEN_DIMS)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: {X.shape[1]} -> {HIDDEN_DIMS} -> {y.shape[1]}  ({n_params:,} params)")

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.5
)
criterion = nn.MSELoss()

train_losses, val_losses = [], []
best_val_loss = float("inf")
best_state = None
patience = 0

print(
    f"\nTraining for up to {EPOCHS} epochs (early stopping patience: {EARLY_STOPPING_PATIENCE})"
)
print("-" * 60)

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= len(X_train_n)
    train_losses.append(train_loss)

    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            val_loss += criterion(model(xb), yb).item() * len(xb)
    val_loss /= len(X_val_n)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience = 0
    else:
        patience += 1

    if (epoch + 1) % 20 == 0 or epoch == 0:
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:3d}: train={train_loss:.6f}, val={val_loss:.6f}, lr={lr:.1e}"
        )

    if patience >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(best_state)
print(f"\nBest validation loss: {best_val_loss:.6f}")

# ============================================================================
# Evaluate
# ============================================================================
model.eval()
with torch.no_grad():
    y_pred_n = model(torch.tensor(X_val_n)).numpy()

# Denormalize
y_pred = y_pred_n * y_std + y_mean
y_true = y_val

# Metrics
mse = np.mean((y_pred - y_true) ** 2)
rmse = np.sqrt(mse)
r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean(0)) ** 2)

print(f"\nValidation metrics (original scale):")
print(f"  RMSE: {rmse:.6f}")
print(f"  R²:   {r2:.4f}")

# ============================================================================
# Save model
# ============================================================================
models_dir = data_dir / "models"
models_dir.mkdir(exist_ok=True)

torch.save(
    {
        "state_dict": model.state_dict(),
        "hidden_dims": HIDDEN_DIMS,
        "X_mean": X_mean,
        "X_std": X_std,
        "y_mean": y_mean,
        "y_std": y_std,
    },
    models_dir / f"{MODEL_NAME}.pt",
)
print(f"\nModel saved: {models_dir / MODEL_NAME}.pt")

# ============================================================================
# Plot results
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle(f"NN-INDI Training Results (R²={r2:.4f})", fontweight="bold")

# Training curves
ax = axes[0]
ax.semilogy(train_losses, "b-", alpha=0.7, label="Train")
ax.semilogy(val_losses, "r-", alpha=0.7, label="Val")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("Training Progress")
ax.legend()
ax.grid(True, alpha=0.3)

# Prediction scatter
ax = axes[1]
true_mag = np.linalg.norm(y_true, axis=1)
pred_mag = np.linalg.norm(y_pred, axis=1)
ax.scatter(true_mag, pred_mag, alpha=0.1, s=1)
ax.plot([0, true_mag.max()], [0, true_mag.max()], "r--", lw=2)
ax.set_xlabel("True Residual Magnitude")
ax.set_ylabel("Predicted Residual Magnitude")
ax.set_title("Prediction vs Truth")
ax.grid(True, alpha=0.3)

# Error distribution
ax = axes[2]
error_mag = np.linalg.norm(y_pred - y_true, axis=1)
ax.hist(error_mag, bins=50, density=True, alpha=0.7)
ax.axvline(error_mag.mean(), color="r", ls="--", label=f"Mean: {error_mag.mean():.4f}")
ax.set_xlabel("Prediction Error Magnitude")
ax.set_ylabel("Density")
ax.set_title("Error Distribution")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(data_dir / "training_results.png", dpi=150)
print(f"Plot saved: {data_dir / 'training_results.png'}")

# Quick inference benchmark
import time

x_test = torch.randn(1, X.shape[1])
for _ in range(100):
    model(x_test)  # warmup
t0 = time.perf_counter()
for _ in range(10000):
    model(x_test)
dt = (time.perf_counter() - t0) / 10000 * 1e6
print(f"\nInference: {dt:.1f} µs/sample ({1e6/dt:.0f} Hz max)")
