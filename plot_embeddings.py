import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/embedding_visualization.npz")
embeddings_2d = data["embeddings_2d"]
sst = data["sst"]
ssh = data["ssh"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Color the same embedding plot by SST on the left, SSH on the right -
# if the embedding is meaningful, points with similar surface conditions
# should form visible color gradients / clusters, not random scatter
sc1 = axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=sst, cmap="coolwarm", s=3, alpha=0.6)
axes[0].set_title("Ocean Embedding Space, colored by SST")
axes[0].set_xlabel("Embedding dimension 1")
axes[0].set_ylabel("Embedding dimension 2")
plt.colorbar(sc1, ax=axes[0], label="SST (°C)")

sc2 = axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=ssh, cmap="viridis", s=3, alpha=0.6)
axes[1].set_title("Ocean Embedding Space, colored by SSH")
axes[1].set_xlabel("Embedding dimension 1")
axes[1].set_ylabel("Embedding dimension 2")
plt.colorbar(sc2, ax=axes[1], label="SSH (m)")

plt.tight_layout()
plt.savefig("data/embedding_plot.png", dpi=150)
print("Saved plot to data/embedding_plot.png")
plt.show()