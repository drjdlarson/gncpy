"""Train NN to learn state-dependent B matrix corrections (V3).

This implements B-learning for INDI augmentation:

    Δẋ ≈ B(x,u) Δu

The NN predicts ΔB_θ(z) such that B_θ(z) = B0 + ΔB_θ(z).

Training minimizes:
    ||Δẋ - (B0 + ΔB_θ(z)) @ Δu||²

This is INDI-consistent because it directly learns what local effectiveness
best explains how ẋ changes when inputs change.

Two training modes:
1. Learn full ΔB: NN outputs 48 values (6×8 matrix)
2. Learn diagonal scaling: NN outputs 8 per-motor thrust scalers (simpler)
"""

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
HIDDEN_DIMS = [64, 64, 64]
EPOCHS = 300
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 40
MODEL_NAME = "b_matrix_correction"

# Choose mode: "full" for 6x8 ΔB, "diagonal" for per-motor scaling
B_LEARNING_MODE = "full"  # or "diagonal"

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(16)


# ============================================================================
# Model for full ΔB learning
# ============================================================================
class DeltaBMatrixMLP(nn.Module):
    """MLP that predicts ΔB matrix correction (6×8 = 48 outputs)."""

    def __init__(self, input_dim, hidden_dims, num_outputs=48):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, num_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Output: (batch, 48) -> reshape to (batch, 6, 8)
        return self.net(x).view(-1, 6, 8)


# ============================================================================
# Model for diagonal scaling (simpler)
# ============================================================================
class DiagonalScalingMLP(nn.Module):
    """MLP that predicts per-motor effectiveness scaling (8 outputs).

    B_effective = B0 @ diag(1 + scale_θ(z))

    This is simpler and more physically interpretable.
    """

    def __init__(self, input_dim, hidden_dims, num_motors=8):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, num_motors))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Output: (batch, 8) scaling factors (centered around 0)
        return self.net(x)


# ============================================================================
# Load data
# ============================================================================
data_dir = Path(__file__).parent / "training_data_v3"
print(f"Loading V3 data from {data_dir}")

X = pd.read_csv(data_dir / "nn_inputs.csv").iloc[:, 2:].values.astype(np.float32)
delta_u = pd.read_csv(data_dir / "delta_u.csv").iloc[:, 2:].values.astype(np.float32)
y = pd.read_csv(data_dir / "nn_targets.csv").iloc[:, 2:].values.astype(np.float32)
B0 = np.load(data_dir / "B0.npy").astype(np.float32)

print(f"  Samples: {len(X)}")
print(f"  Input (z) dim: {X.shape[1]}")
print(f"  Δu dim: {delta_u.shape[1]}")
print(f"  Target (Δẋ) dim: {y.shape[1]}")
print(f"  B0 shape: {B0.shape}")

# Train/val split
n_val = int(len(X) * VAL_SPLIT)
idx = np.random.permutation(len(X))
X_train, X_val = X[idx[n_val:]], X[idx[:n_val]]
delta_u_train, delta_u_val = delta_u[idx[n_val:]], delta_u[idx[:n_val]]
y_train, y_val = y[idx[n_val:]], y[idx[:n_val]]
print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

# Normalize inputs only (targets are used directly in loss)
X_mean, X_std = X_train.mean(0), X_train.std(0)
X_std[X_std < 1e-8] = 1.0

X_train_n = (X_train - X_mean) / X_std
X_val_n = (X_val - X_mean) / X_std

# Convert to tensors
X_train_t = torch.tensor(X_train_n)
X_val_t = torch.tensor(X_val_n)
delta_u_train_t = torch.tensor(delta_u_train)
delta_u_val_t = torch.tensor(delta_u_val)
y_train_t = torch.tensor(y_train)
y_val_t = torch.tensor(y_val)
B0_t = torch.tensor(B0)

# Data loaders
train_loader = DataLoader(
    TensorDataset(X_train_t, delta_u_train_t, y_train_t),
    batch_size=BATCH_SIZE,
    shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(X_val_t, delta_u_val_t, y_val_t),
    batch_size=BATCH_SIZE,
)

# ============================================================================
# Create model
# ============================================================================
if B_LEARNING_MODE == "full":
    model = DeltaBMatrixMLP(X.shape[1], HIDDEN_DIMS, num_outputs=48)
    print(f"\nMode: Full ΔB learning (6×8 = 48 outputs)")
else:
    model = DiagonalScalingMLP(X.shape[1], HIDDEN_DIMS, num_motors=8)
    print(f"\nMode: Diagonal scaling (8 outputs)")

n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {X.shape[1]} -> {HIDDEN_DIMS} -> outputs  ({n_params:,} params)")

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=15, factor=0.5
)


# ============================================================================
# Custom loss function
# ============================================================================
def b_learning_loss(model_output, delta_u, target, B0):
    """Compute ||Δẋ - B_θ(z) @ Δu||²

    For full mode: B_θ = B0 + ΔB_θ
    For diagonal mode: B_θ = B0 @ diag(1 + scale_θ)
    """
    if B_LEARNING_MODE == "full":
        # model_output: (batch, 6, 8) = ΔB
        # B_effective = B0 + ΔB
        B_effective = B0.unsqueeze(0) + model_output  # (batch, 6, 8)
        # Compute B @ Δu: (batch, 6, 8) @ (batch, 8, 1) -> (batch, 6, 1)
        pred = torch.bmm(B_effective, delta_u.unsqueeze(-1)).squeeze(-1)  # (batch, 6)
    else:
        # model_output: (batch, 8) = scaling factors
        # B_effective = B0 @ diag(1 + scale)
        scale = 1.0 + model_output  # (batch, 8)
        scaled_delta_u = delta_u * scale  # (batch, 8)
        pred = torch.mm(scaled_delta_u, B0.T)  # (batch, 6)

    return torch.mean((pred - target) ** 2)


# ============================================================================
# Training loop
# ============================================================================
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
    for xb, dub, yb in train_loader:
        optimizer.zero_grad()
        output = model(xb)
        loss = b_learning_loss(output, dub, yb, B0_t)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= len(X_train_n)
    train_losses.append(train_loss)

    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, dub, yb in val_loader:
            output = model(xb)
            val_loss += b_learning_loss(output, dub, yb, B0_t).item() * len(xb)
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
    if B_LEARNING_MODE == "full":
        delta_B_pred = model(X_val_t)  # (N, 6, 8)
        B_effective = B0_t.unsqueeze(0) + delta_B_pred
        y_pred = torch.bmm(B_effective, delta_u_val_t.unsqueeze(-1)).squeeze(-1).numpy()
    else:
        scale = 1.0 + model(X_val_t)
        scaled_du = delta_u_val_t * scale
        y_pred = torch.mm(scaled_du, B0_t.T).numpy()

y_true = y_val

# B0 baseline prediction
y_pred_b0 = (B0 @ delta_u_val.T).T

# Metrics
mse_nn = np.mean((y_pred - y_true) ** 2)
mse_b0 = np.mean((y_pred_b0 - y_true) ** 2)
rmse_nn = np.sqrt(mse_nn)
rmse_b0 = np.sqrt(mse_b0)

ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - y_true.mean(0)) ** 2)
r2 = 1 - ss_res / ss_tot

print(f"\nValidation metrics:")
print(f"  B0 baseline RMSE: {rmse_b0:.6f}")
print(f"  NN-corrected RMSE: {rmse_nn:.6f}")
print(f"  Improvement: {(rmse_b0 - rmse_nn) / rmse_b0 * 100:.1f}%")
print(f"  R²: {r2:.4f}")

# ============================================================================
# Save model
# ============================================================================
models_dir = data_dir / "models"
models_dir.mkdir(exist_ok=True)

torch.save(
    {
        "state_dict": model.state_dict(),
        "hidden_dims": HIDDEN_DIMS,
        "mode": B_LEARNING_MODE,
        "X_mean": X_mean,
        "X_std": X_std,
        "B0": B0,
    },
    models_dir / f"{MODEL_NAME}.pt",
)
print(f"\nModel saved: {models_dir / MODEL_NAME}.pt")

# ============================================================================
# Plot results
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f"V3 B-Matrix Learning Results (R²={r2:.4f})", fontweight="bold")

# Training curves
ax = axes[0, 0]
ax.semilogy(train_losses, "b-", alpha=0.7, label="Train")
ax.semilogy(val_losses, "r-", alpha=0.7, label="Val")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("Training Progress")
ax.legend()
ax.grid(True, alpha=0.3)

# B0 vs NN prediction scatter (velocity)
ax = axes[0, 1]
ax.scatter(y_true[:, 0], y_pred_b0[:, 0], alpha=0.1, s=1, c="b", label="B0")
ax.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.1, s=1, c="r", label="NN")
lim = max(np.abs(y_true[:, 0]).max(), np.abs(y_pred[:, 0]).max())
ax.plot([-lim, lim], [-lim, lim], "k--", lw=2)
ax.set_xlabel("True Δv̇_x")
ax.set_ylabel("Predicted Δv̇_x")
ax.set_title("Δv̇_x: B0 vs NN")
ax.legend(markerscale=10)
ax.grid(True, alpha=0.3)

# B0 vs NN prediction scatter (angular)
ax = axes[0, 2]
ax.scatter(y_true[:, 3], y_pred_b0[:, 3], alpha=0.1, s=1, c="b", label="B0")
ax.scatter(y_true[:, 3], y_pred[:, 3], alpha=0.1, s=1, c="r", label="NN")
lim = max(np.abs(y_true[:, 3]).max(), np.abs(y_pred[:, 3]).max())
ax.plot([-lim, lim], [-lim, lim], "k--", lw=2)
ax.set_xlabel("True Δω̇_x")
ax.set_ylabel("Predicted Δω̇_x")
ax.set_title("Δω̇_x: B0 vs NN")
ax.legend(markerscale=10)
ax.grid(True, alpha=0.3)

# Error comparison
ax = axes[1, 0]
error_b0 = np.linalg.norm(y_pred_b0 - y_true, axis=1)
error_nn = np.linalg.norm(y_pred - y_true, axis=1)
ax.hist(
    error_b0, bins=50, alpha=0.5, density=True, label=f"B0 (mean={error_b0.mean():.4f})"
)
ax.hist(
    error_nn, bins=50, alpha=0.5, density=True, label=f"NN (mean={error_nn.mean():.4f})"
)
ax.set_xlabel("Prediction Error Magnitude")
ax.set_ylabel("Density")
ax.set_title("Error Distribution")
ax.legend()
ax.grid(True, alpha=0.3)

# Per-axis RMSE comparison
ax = axes[1, 1]
axis_names = ["Δv̇_x", "Δv̇_y", "Δv̇_z", "Δω̇_x", "Δω̇_y", "Δω̇_z"]
rmse_b0_per = np.sqrt(np.mean((y_pred_b0 - y_true) ** 2, axis=0))
rmse_nn_per = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))
x_pos = np.arange(6)
width = 0.35
ax.bar(x_pos - width / 2, rmse_b0_per, width, label="B0", alpha=0.7)
ax.bar(x_pos + width / 2, rmse_nn_per, width, label="NN", alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(axis_names)
ax.set_ylabel("RMSE")
ax.set_title("Per-Axis RMSE Comparison")
ax.legend()
ax.grid(True, alpha=0.3)

# If full mode, visualize learned ΔB statistics
if B_LEARNING_MODE == "full":
    ax = axes[1, 2]
    with torch.no_grad():
        delta_B_all = model(X_val_t).numpy()  # (N, 6, 8)
    delta_B_mean = delta_B_all.mean(axis=0)
    im = ax.imshow(delta_B_mean, cmap="RdBu", aspect="auto")
    ax.set_xlabel("Motor index")
    ax.set_ylabel("Output axis (v̇/ω̇)")
    ax.set_title("Mean ΔB correction")
    plt.colorbar(im, ax=ax)
else:
    ax = axes[1, 2]
    with torch.no_grad():
        scales = model(X_val_t).numpy()  # (N, 8)
    ax.boxplot(scales)
    ax.axhline(0, color="r", ls="--")
    ax.set_xlabel("Motor index")
    ax.set_ylabel("Effectiveness scaling")
    ax.set_title("Learned per-motor scaling")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(data_dir / "training_results.png", dpi=150)
print(f"Plot saved: {data_dir / 'training_results.png'}")

plt.show()
