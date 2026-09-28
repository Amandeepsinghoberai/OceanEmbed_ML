import numpy as np
import torch
import torch.nn as nn

data = np.load("data/dataset_4yr.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]

depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

N_MODELS = 7
print(f"Training {N_MODELS} independent models for the ensemble...")

X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
X_train_norm = (X_train - X_mean) / X_std
X_val_norm = (X_val - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std

y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0)
y_train_norm = (y_train - y_mean) / y_std
y_val_norm = (y_val - y_mean) / y_std

X_train_t = torch.tensor(X_train_norm, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train_norm, dtype=torch.float32).to(device)
X_val_t = torch.tensor(X_val_norm, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val_norm, dtype=torch.float32).to(device)
X_test_t = torch.tensor(X_test_norm, dtype=torch.float32).to(device)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 48), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(48, 48), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(48, 15)
        )

    def forward(self, x):
        return self.net(x)


def train_one_model(seed):
    """Trains one model with a different random starting point.
    Same architecture and settings as our proven 4-year model - only
    the random seed differs, so any disagreement between models
    reflects genuine uncertainty, not different approaches."""
    torch.manual_seed(seed)
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    batch_size = 4096
    n_train = X_train_t.shape[0]
    best_val_loss = float("inf")
    patience, patience_counter = 15, 0
    max_epochs = 300
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val_loss


all_predictions = []

for i in range(N_MODELS):
    print(f"\nTraining model {i+1}/{N_MODELS} (seed={i})...")
    model, val_loss = train_one_model(seed=i)
    print(f"  Model {i+1} finished, best val loss: {val_loss:.4f}")

    torch.save(model.state_dict(), f"data/ensemble_model_{i}.pt")

    model.eval()
    with torch.no_grad():
        pred_norm = model(X_test_t).cpu().numpy()
    pred_temp = pred_norm * y_std + y_mean
    all_predictions.append(pred_temp)

all_predictions = np.stack(all_predictions, axis=0)  # (N_MODELS, n_samples, 15)

# The ensemble's final answer is the AVERAGE of all models - typically
# more accurate and stable than any single model alone
ensemble_mean = all_predictions.mean(axis=0)
# The ensemble's UNCERTAINTY is how much the models disagree with each
# other - high disagreement = low confidence, low disagreement = high confidence
ensemble_std = all_predictions.std(axis=0)

print("\n" + "=" * 70)
print("ENSEMBLE RESULTS (average of all models, on the held-out test set)")
print("=" * 70)
print(f"\n{'Depth (m)':>10} {'RMSE (C)':>10} {'Correlation':>12} {'Avg Uncertainty (C)':>20}")
for i, d in enumerate(depths):
    err = ensemble_mean[:, i] - y_test[:, i]
    rmse = np.sqrt(np.mean(err ** 2))
    corr = np.corrcoef(y_test[:, i], ensemble_mean[:, i])[0, 1]
    avg_uncertainty = ensemble_std[:, i].mean()
    print(f"{d:>10} {rmse:>10.3f} {corr:>12.3f} {avg_uncertainty:>20.3f}")

overall_rmse = np.sqrt(np.mean((ensemble_mean - y_test) ** 2))
print(f"\nEnsemble overall RMSE: {overall_rmse:.3f}  (single model was 0.794)")

# Sanity check: does higher uncertainty actually correspond to higher real
# error? If yes, the uncertainty is meaningful, not just decorative noise.
mean_uncertainty_per_depth = ensemble_std.mean(axis=0)
mean_error_per_depth = np.abs(ensemble_mean - y_test).mean(axis=0)
sanity_corr = np.corrcoef(mean_uncertainty_per_depth, mean_error_per_depth)[0, 1]
print(f"\nSanity check - correlation between our uncertainty estimate and real error, across depths: {sanity_corr:.3f}")
print("(Close to 1.0 means the uncertainty is trustworthy and tracks real accuracy)")

np.savez(
    "data/ensemble_results.npz",
    ensemble_mean=ensemble_mean,
    ensemble_std=ensemble_std,
    y_test=y_test,
)
print("\nSaved ensemble predictions and uncertainty to data/ensemble_results.npz")