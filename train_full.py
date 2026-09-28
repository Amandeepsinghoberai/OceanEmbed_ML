import numpy as np
import torch
import torch.nn as nn

data = np.load("data/dataset_full.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]

depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Training samples: {X_train.shape[0]:,}")
print(f"Features: {X_train.shape[1]} (SST, SSH, + 5 cluster flags)")


def compute_metrics(y_true, y_pred, depths):
    print(f"\n{'Depth (m)':>10} {'RMSE (C)':>10} {'Bias (C)':>10} {'Correlation':>12}")
    for i, d in enumerate(depths):
        err = y_pred[:, i] - y_true[:, i]
        rmse = np.sqrt(np.mean(err ** 2))
        bias = np.mean(err)
        corr = np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1]
        print(f"{d:>10} {rmse:>10.3f} {bias:>10.3f} {corr:>12.3f}")
    overall_rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    overall_bias = np.mean(y_pred - y_true)
    overall_corr = np.mean([np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1] for i in range(len(depths))])
    print(f"{'OVERALL':>10} {overall_rmse:>10.3f} {overall_bias:>10.3f} {overall_corr:>12.3f}")
    return overall_rmse, overall_corr


X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
X_std[X_std == 0] = 1
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
    def __init__(self, n_inputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 48), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(48, 48), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(48, 15)
        )

    def forward(self, x):
        return self.net(x)


model = MLP(n_inputs=X_train.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=7e-4, weight_decay=3e-4)
loss_fn = nn.MSELoss()

batch_size = 8192  # larger batch since we have ~3x more data now
n_train = X_train_t.shape[0]
best_val_loss = float("inf")
best_epoch = -1
patience, patience_counter = 20, 0
max_epochs = 300

for epoch in range(max_epochs):
    model.train()
    perm = torch.randperm(n_train)
    total_loss = 0.0
    for i in range(0, n_train, batch_size):
        idx = perm[i:i + batch_size]
        xb, yb = X_train_t[idx], y_train_t[idx]
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.shape[0]

    train_loss = total_loss / n_train

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = loss_fn(val_pred, y_val_t).item()

    if epoch % 5 == 0 or epoch == max_epochs - 1:
        print(f"Epoch {epoch:3d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0
        torch.save(model.state_dict(), "data/best_mlp_full.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
            break

print(f"\nBest checkpoint was from epoch {best_epoch} (val MSE {best_val_loss:.4f})")

model.load_state_dict(torch.load("data/best_mlp_full.pt"))
model.eval()
with torch.no_grad():
    y_pred_norm = model(X_test_t).cpu().numpy()
y_pred = y_pred_norm * y_std + y_mean

print("\n" + "=" * 60)
print("FULL REGION MODEL (Bay of Bengal + Arabian Sea, SST+SSH+cluster)")
print("=" * 60)
rmse_full, corr_full = compute_metrics(y_test, y_pred, depths)

print("\n" + "=" * 60)
print(f"COMPARISON: Bay-of-Bengal-only model was RMSE 0.782, Correlation 0.662")
print(f"Full region model:                        RMSE {rmse_full:.3f}, Correlation {corr_full:.3f}")
print("=" * 60)

np.savez("data/normalization_stats_full.npz", X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)
print("\nSaved model to data/best_mlp_full.pt")