import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

data = np.load("data/dataset_4yr.npz")
X_test, y_test = data["X_test"], data["y_test"]

norm = np.load("data/normalization_stats_4yr.npz")
X_mean, X_std = norm["X_mean"], norm["X_std"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(2, 48), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(48, 48), nn.ReLU())
        self.output = nn.Linear(48, 15)

    def forward(self, x):
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        out = self.output(h2)
        return out, h2  # return both the final answer AND the embedding


model = MLP().to(device)

# Load your existing trained weights into this restructured (but identical) architecture
old_state = torch.load("data/best_mlp_4yr.pt")
new_state = {
    "layer1.0.weight": old_state["net.0.weight"], "layer1.0.bias": old_state["net.0.bias"],
    "layer2.0.weight": old_state["net.3.weight"], "layer2.0.bias": old_state["net.3.bias"],
    "output.weight": old_state["net.6.weight"], "output.bias": old_state["net.6.bias"],
}
model.load_state_dict(new_state)
model.eval()

# Use a manageable sample - 20,000 random points from the test set - rather
# than all 500,000+, since we just need enough for a meaningful visualization
np.random.seed(42)
sample_idx = np.random.choice(len(X_test), size=min(20000, len(X_test)), replace=False)
X_sample = X_test[sample_idx]

X_norm = (X_sample - X_mean) / X_std
X_t = torch.tensor(X_norm, dtype=torch.float32).to(device)

with torch.no_grad():
    _, embeddings = model(X_t)
embeddings = embeddings.cpu().numpy()  # shape (20000, 48)

print(f"Extracted embeddings: {embeddings.shape}")

# Reduce 48 dimensions down to 2, purely for visualization on a scatter plot
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

print(f"2D projection shape: {embeddings_2d.shape}")
print(f"Variance explained by 2 components: {pca.explained_variance_ratio_.sum():.1%}")

# Save both the raw surface values (for coloring the plot meaningfully -
# e.g. by SST, so we can see if similar temperatures cluster together)
# and the 2D embedding positions
np.savez(
    "data/embedding_visualization.npz",
    embeddings_2d=embeddings_2d,
    sst=X_sample[:, 0],
    ssh=X_sample[:, 1],
)
print("\nSaved to data/embedding_visualization.npz")
print("Sample 2D points (first 5):")
print(embeddings_2d[:5])