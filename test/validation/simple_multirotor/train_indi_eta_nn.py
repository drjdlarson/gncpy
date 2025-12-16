"""Train NN for structured η-learning (V4).

The NN predicts per-motor effectiveness scalings:
- η_F ∈ R^8: force column scalings
- η_M ∈ R^8: moment column scalings

Then B_hat = [B0_F @ diag(η_F); B0_M @ diag(η_M)]

Key features:
1. Bounded outputs: η = 1 + 0.5*tanh(ρ) → [0.5, 1.5]
2. Regularizer: ||η - 1||² to keep near nominal
3. Input: z = [v_B, ω_B, g_B, u] ∈ R^17
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
MODEL_NAME = "eta_scalings_mlp"

# Regularization weight for ||η - 1||²
REG_WEIGHT = 0.01

# Output bounds: η ∈ [1 - BOUND, 1 + BOUND]
ETA_BOUND = 0.5  # So η ∈ [0.5, 1.5]

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(16)


# ============================================================================
# Model with bounded outputs
# ============================================================================
class EtaScalingMLP(nn.Module):
    """MLP that predicts bounded per-motor effectiveness scalings.

    Output: η_F (8) + η_M (8) = 16 values, bounded to [0.5, 1.5]
    via η = 1 + bound * tanh(raw_output)
    """

    def __init__(self, input_dim, hidden_dims, num_motors=8, eta_bound=0.5):
        super().__init__()
        self.num_motors = num_motors
        self.eta_bound = eta_bound

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        # Output: η_F (8) + η_M (8) = 16 raw values
        layers.append(nn.Linear(prev, 2 * num_motors))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        raw = self.net(x)
        # Bound outputs: η = 1 + bound * tanh(raw)
        eta = 1.0 + self.eta_bound * torch.tanh(raw)
        return eta

    def forward_with_raw(self, x):
        """Return both bounded η and raw pre-tanh values."""
        raw = self.net(x)
        eta = 1.0 + self.eta_bound * torch.tanh(raw)
        return eta, raw


# ============================================================================
# Load data
# ============================================================================
data_dir = Path(__file__).parent / "training_data_v4"
print(f"Loading V4 data from {data_dir}")

X = pd.read_csv(data_dir / "nn_inputs.csv").iloc[:, 2:].values.astype(np.float32)
y = pd.read_csv(data_dir / "nn_targets.csv").iloc[:, 2:].values.astype(np.float32)
B0_F = np.load(data_dir / "B0_F.npy").astype(np.float32)
B0_M = np.load(data_dir / "B0_M.npy").astype(np.float32)

num_motors = B0_F.shape[1]
print(f"  Samples: {len(X)}")
print(f"  Input (z) dim: {X.shape[1]}")
print(f"  Target (η_F + η_M) dim: {y.shape[1]}")
print(f"  Number of motors: {num_motors}")

# Analyze target distribution
eta_F = y[:, :num_motors]
eta_M = y[:, num_motors:]
print(f"\n  η_F range: [{eta_F.min():.3f}, {eta_F.max():.3f}], mean={eta_F.mean():.3f}")
print(f"  η_M range: [{eta_M.min():.3f}, {eta_M.max():.3f}], mean={eta_M.mean():.3f}")

# Check if any η values are outside bounding range
out_of_bounds_F = np.sum((eta_F < 0.5) | (eta_F > 1.5)) / eta_F.size * 100
out_of_bounds_M = np.sum((eta_M < 0.5) | (eta_M > 1.5)) / eta_M.size * 100
print(f"  η_F out of [0.5, 1.5]: {out_of_bounds_F:.2f}%")
print(f"  η_M out of [0.5, 1.5]: {out_of_bounds_M:.2f}%")

# Train/val split
n_val = int(len(X) * VAL_SPLIT)
idx = np.random.permutation(len(X))
X_train, X_val = X[idx[n_val:]], X[idx[:n_val]]
y_train, y_val = y[idx[n_val:]], y[idx[:n_val]]
print(f"\n  Train: {len(X_train)}, Val: {len(X_val)}")

# Normalize inputs only
X_mean, X_std = X_train.mean(0), X_train.std(0)
X_std[X_std < 1e-8] = 1.0

X_train_n = (X_train - X_mean) / X_std
X_val_n = (X_val - X_mean) / X_std

# Convert to tensors
X_train_t = torch.tensor(X_train_n)
X_val_t = torch.tensor(X_val_n)
y_train_t = torch.tensor(y_train)
y_val_t = torch.tensor(y_val)

# Data loaders
train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=BATCH_SIZE,
    shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(X_val_t, y_val_t),
    batch_size=BATCH_SIZE,
)


# ============================================================================
# Create model
# ============================================================================
model = EtaScalingMLP(
    input_dim=X.shape[1],
    hidden_dims=HIDDEN_DIMS,
    num_motors=num_motors,
    eta_bound=ETA_BOUND,
)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: {X.shape[1]} -> {HIDDEN_DIMS} -> 16  ({n_params:,} params)")
print(f"Output bounds: η ∈ [{1-ETA_BOUND}, {1+ETA_BOUND}]")

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=15, factor=0.5
)


# ============================================================================
# Loss function with regularization
# ============================================================================
def eta_loss(eta_pred, eta_true, reg_weight=REG_WEIGHT):
    """MSE loss + regularizer to keep η near 1.

    Loss = ||η_pred - η_true||² + λ * ||η_pred - 1||²
    """
    mse = torch.mean((eta_pred - eta_true) ** 2)
    reg = torch.mean((eta_pred - 1.0) ** 2)
    return mse + reg_weight * reg, mse, reg


# ============================================================================
# Training loop
# ============================================================================
train_losses, val_losses = [], []
train_mses, val_mses = [], []
best_val_loss = float("inf")
best_state = None
patience = 0

print(
    f"\nTraining for up to {EPOCHS} epochs (early stopping patience: {EARLY_STOPPING_PATIENCE})"
)
print(f"Regularization weight: {REG_WEIGHT}")
print("-" * 60)

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss, train_mse_total, train_reg_total = 0, 0, 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        eta_pred = model(xb)
        loss, mse, reg = eta_loss(eta_pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)
        train_mse_total += mse.item() * len(xb)
    train_loss /= len(X_train_n)
    train_mse_total /= len(X_train_n)
    train_losses.append(train_loss)
    train_mses.append(train_mse_total)

    # Validate
    model.eval()
    val_loss, val_mse_total = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            eta_pred = model(xb)
            loss, mse, reg = eta_loss(eta_pred, yb)
            val_loss += loss.item() * len(xb)
            val_mse_total += mse.item() * len(xb)
    val_loss /= len(X_val_n)
    val_mse_total /= len(X_val_n)
    val_losses.append(val_loss)
    val_mses.append(val_mse_total)
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
            f"Epoch {epoch+1:3d}: loss={train_loss:.6f}, mse={train_mse_total:.6f}, "
            f"val_loss={val_loss:.6f}, val_mse={val_mse_total:.6f}, lr={lr:.1e}"
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
    eta_pred = model(X_val_t).numpy()

eta_true = y_val

# Per-output metrics
eta_F_pred = eta_pred[:, :num_motors]
eta_M_pred = eta_pred[:, num_motors:]
eta_F_true = eta_true[:, :num_motors]
eta_M_true = eta_true[:, num_motors:]

rmse_F = np.sqrt(np.mean((eta_F_pred - eta_F_true) ** 2))
rmse_M = np.sqrt(np.mean((eta_M_pred - eta_M_true) ** 2))
rmse_total = np.sqrt(np.mean((eta_pred - eta_true) ** 2))

# R² score
ss_res = np.sum((eta_true - eta_pred) ** 2)
ss_tot = np.sum((eta_true - eta_true.mean()) ** 2)
r2 = 1 - ss_res / ss_tot

print(f"\nValidation metrics:")
print(f"  RMSE η_F: {rmse_F:.6f}")
print(f"  RMSE η_M: {rmse_M:.6f}")
print(f"  RMSE total: {rmse_total:.6f}")
print(f"  R²: {r2:.4f}")

# Per-motor analysis
print(f"\nPer-motor η_F accuracy:")
for i in range(num_motors):
    rmse_i = np.sqrt(np.mean((eta_F_pred[:, i] - eta_F_true[:, i]) ** 2))
    mean_true = eta_F_true[:, i].mean()
    print(f"  Motor {i}: RMSE={rmse_i:.4f}, mean_true={mean_true:.4f}")

print(f"\nPer-motor η_M accuracy:")
for i in range(num_motors):
    rmse_i = np.sqrt(np.mean((eta_M_pred[:, i] - eta_M_true[:, i]) ** 2))
    mean_true = eta_M_true[:, i].mean()
    print(f"  Motor {i}: RMSE={rmse_i:.4f}, mean_true={mean_true:.4f}")


# ============================================================================
# Save model
# ============================================================================
models_dir = data_dir / "models"
models_dir.mkdir(exist_ok=True)

torch.save(
    {
        "state_dict": model.state_dict(),
        "hidden_dims": HIDDEN_DIMS,
        "num_motors": num_motors,
        "eta_bound": ETA_BOUND,
        "X_mean": X_mean,
        "X_std": X_std,
        "B0_F": B0_F,
        "B0_M": B0_M,
    },
    models_dir / f"{MODEL_NAME}.pt",
)
print(f"\nModel saved: {models_dir / MODEL_NAME}.pt")


# ============================================================================
# Plot results
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f"V4 η-Learning Results (R²={r2:.4f})", fontweight="bold")

# Training curves
ax = axes[0, 0]
ax.semilogy(train_losses, "b-", alpha=0.7, label="Train")
ax.semilogy(val_losses, "r-", alpha=0.7, label="Val")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (MSE + reg)")
ax.set_title("Training Progress")
ax.legend()
ax.grid(True, alpha=0.3)

# η_F prediction scatter (average across motors)
ax = axes[0, 1]
eta_F_mean_pred = eta_F_pred.mean(axis=1)
eta_F_mean_true = eta_F_true.mean(axis=1)
ax.scatter(eta_F_mean_true, eta_F_mean_pred, alpha=0.1, s=1)
ax.plot([0.5, 1.5], [0.5, 1.5], "r--", lw=2)
ax.set_xlabel("True η_F (mean)")
ax.set_ylabel("Predicted η_F (mean)")
ax.set_title("η_F Prediction")
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 1.5)
ax.set_ylim(0.5, 1.5)

# η_M prediction scatter (average across motors)
ax = axes[0, 2]
eta_M_mean_pred = eta_M_pred.mean(axis=1)
eta_M_mean_true = eta_M_true.mean(axis=1)
ax.scatter(eta_M_mean_true, eta_M_mean_pred, alpha=0.1, s=1)
ax.plot([0.5, 1.5], [0.5, 1.5], "r--", lw=2)
ax.set_xlabel("True η_M (mean)")
ax.set_ylabel("Predicted η_M (mean)")
ax.set_title("η_M Prediction")
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 1.5)
ax.set_ylim(0.5, 1.5)

# Per-motor η_F distribution
ax = axes[1, 0]
for i in range(num_motors):
    ax.hist(eta_F_pred[:, i], bins=30, alpha=0.3, label=f"M{i}")
ax.axvline(1.0, color="k", ls="--", label="Nominal")
ax.set_xlabel("η_F")
ax.set_ylabel("Count")
ax.set_title("Predicted η_F Distribution")
ax.legend(fontsize=6, ncol=2)
ax.grid(True, alpha=0.3)

# Per-motor η_M distribution
ax = axes[1, 1]
for i in range(num_motors):
    ax.hist(eta_M_pred[:, i], bins=30, alpha=0.3, label=f"M{i}")
ax.axvline(1.0, color="k", ls="--", label="Nominal")
ax.set_xlabel("η_M")
ax.set_ylabel("Count")
ax.set_title("Predicted η_M Distribution")
ax.legend(fontsize=6, ncol=2)
ax.grid(True, alpha=0.3)

# Error distribution
ax = axes[1, 2]
error_F = (eta_F_pred - eta_F_true).flatten()
error_M = (eta_M_pred - eta_M_true).flatten()
ax.hist(
    error_F,
    bins=50,
    alpha=0.5,
    label=f"η_F err (std={error_F.std():.4f})",
    density=True,
)
ax.hist(
    error_M,
    bins=50,
    alpha=0.5,
    label=f"η_M err (std={error_M.std():.4f})",
    density=True,
)
ax.set_xlabel("Prediction Error")
ax.set_ylabel("Density")
ax.set_title("Error Distribution")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(data_dir / "training_results.png", dpi=150)
print(f"Plot saved: {data_dir / 'training_results.png'}")

plt.show()
