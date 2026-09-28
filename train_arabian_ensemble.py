import numpy as np
import torch
import torch.nn as nn

data = np.load("data/dataset_arabian_eddy.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]

depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Training samples: {X_train.shape[0]:,}")
print(f"Features: {X_train.shape[1]} (real Arabian Sea champion architecture)\n")

# Reuse the real, existing normalization - identical training data means
# identical normalization values, no need to compute or save a new file
norm = np.load("data/normalization_stats_arabian_eddy.npz")
X_mean, X_std = norm["X_mean"], norm["X_std"]
y_mean, y_std = norm["y_mean"], norm["y_std"]

X_train_norm = (X_train - X_mean) / X_std
X_val_norm = (X_val - X_mean) / X_std
y_train_norm = (y_train - y_mean) / y_std
y_val_norm = (y_val - y_mean) / y_std

X_train_t = torch.tensor(X_train_norm, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train_norm, dtype=torch.float32).to(device)
X_val_t = torch.tensor(X_val_norm, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val_norm, dtype=torch.float32).to(device)


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


N_ENSEMBLE = 7
n_inputs = X_train.shape[1]
batch_size = 8192

for seed in range(N_ENSEMBLE):
    print(f"{'='*50}")
    print(f"Training ensemble member {seed+1}/{N_ENSEMBLE} (seed={seed})")
    print(f"{'='*50}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MLP(n_inputs=n_inputs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=7e-4, weight_decay=3e-4)
    loss_fn = nn.MSELoss()

    n_train = X_train_t.shape[0]
    best_val_loss = float("inf")
    patience, patience_counter = 20, 0
    max_epochs = 300

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Val MSE: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"data/ensemble_model_arabian_{seed}.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"  Saved: data/ensemble_model_arabian_{seed}.pt (best val MSE: {best_val_loss:.4f})\n")

print("All 7 Arabian Sea ensemble members trained and saved.")
print("Reused normalization_stats_arabian_eddy.npz - no new normalization file needed.")