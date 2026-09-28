import xarray as xr
import numpy as np
from sklearn.cluster import KMeans

sst = xr.open_dataarray("data/processed_sst_4yr.nc")
ssh = xr.open_dataarray("data/processed_ssh_4yr.nc")
glorys = xr.open_dataarray("data/processed_glorys_15depth_4yr_v2.nc")

print("SST shape:", sst.shape)
print("SSH shape:", ssh.shape)

# Build clusters using each grid point's AVERAGE surface behavior over all
# 4 years - this captures "what kind of place is this" (e.g. typically
# warmer coastal water vs typically cooler open ocean), not day-to-day noise
print("\nComputing average SST/SSH per grid point (to define regions)...")
sst_avg = sst.mean(dim="time").values  # shape (lat, lon)
ssh_avg = ssh.mean(dim="time").values

lat_grid, lon_grid = np.meshgrid(sst.longitude.values, sst.latitude.values)
valid_mask = ~np.isnan(sst_avg) & ~np.isnan(ssh_avg)

# Features for clustering: average SST, average SSH, and location -
# so regions are grouped by both their typical behavior AND their geography
cluster_features = np.stack([
    sst_avg[valid_mask],
    ssh_avg[valid_mask],
    lat_grid[valid_mask],
    lon_grid[valid_mask],
], axis=1)

# Normalize before clustering so no single feature (e.g. lat/lon numbers
# being larger) dominates the grouping unfairly
cluster_features_norm = (cluster_features - cluster_features.mean(axis=0)) / cluster_features.std(axis=0)

N_CLUSTERS = 5
print(f"Running k-means to find {N_CLUSTERS} natural regions...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels_flat = kmeans.fit_predict(cluster_features_norm)

# Rebuild a full-grid map of cluster assignments (with -1 for land/invalid points)
cluster_map = np.full(sst_avg.shape, -1)
cluster_map[valid_mask] = cluster_labels_flat

print("\nCluster sizes (number of grid points per region):")
for c in range(N_CLUSTERS):
    print(f"  Cluster {c}: {(cluster_labels_flat == c).sum()} points")

np.save("data/cluster_map.npy", cluster_map)
print("\nSaved cluster map to data/cluster_map.npy")

# ---------- Now build the training dataset using cluster ID as a feature ----------
all_dates = sst.time.values
n_days = len(all_dates)
train_end = int(n_days * 0.75)
val_end = int(n_days * 0.88)
train_dates = all_dates[:train_end]
val_dates = all_dates[train_end:val_end]
test_dates = all_dates[val_end:]

print(f"\nTrain: {len(train_dates)} days")
print(f"Val:   {len(val_dates)} days")
print(f"Test:  {len(test_dates)} days")


def build_samples(dates):
    X_list, y_list = [], []
    cluster_flat_full = cluster_map.flatten()

    for date in dates:
        sst_day = sst.sel(time=date).values
        ssh_day = ssh.sel(time=date).values
        glorys_day = glorys.sel(time=date).values

        sst_flat = sst_day.flatten()
        ssh_flat = ssh_day.flatten()
        glorys_flat = glorys_day.reshape(glorys_day.shape[0], -1).T

        valid = (
            ~np.isnan(sst_flat) & ~np.isnan(ssh_flat) &
            ~np.isnan(glorys_flat).any(axis=1) & (cluster_flat_full != -1)
        )

        # One-hot encode the cluster ID: instead of a single number
        # (which the model could misread as a scale, like "cluster 3 > cluster 1"),
        # each cluster becomes its own 0/1 flag - a standard, safe way to
        # represent categories that have no real order
        cluster_onehot = np.zeros((valid.sum(), N_CLUSTERS))
        cluster_ids_here = cluster_flat_full[valid]
        cluster_onehot[np.arange(len(cluster_ids_here)), cluster_ids_here] = 1

        X_day = np.concatenate([
            np.stack([sst_flat[valid], ssh_flat[valid]], axis=1),
            cluster_onehot,
        ], axis=1)
        y_day = glorys_flat[valid]

        X_list.append(X_day)
        y_list.append(y_day)

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.float32)
    return X, y


print("\nBuilding training samples...")
X_train, y_train = build_samples(train_dates)
print("Building validation samples...")
X_val, y_val = build_samples(val_dates)
print("Building test samples...")
X_test, y_test = build_samples(test_dates)

print(f"\nTrain samples: {X_train.shape[0]:,}  (features: {X_train.shape[1]})")
print(f"Val samples:   {X_val.shape[0]:,}")
print(f"Test samples:  {X_test.shape[0]:,}")

np.savez(
    "data/dataset_cluster.npz",
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
)
print("\nSaved dataset to data/dataset_cluster.npz")