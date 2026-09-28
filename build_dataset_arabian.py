import xarray as xr
import numpy as np
from sklearn.cluster import KMeans

sst_full = xr.open_dataarray("data/processed_sst_full.nc")
ssh_full = xr.open_dataarray("data/processed_ssh_full.nc")
glorys_full = xr.open_dataarray("data/processed_glorys_15depth_full.nc")

# Arabian Sea only - same dividing line used in the sea-by-sea validation breakdown
ARABIAN_LON_MAX = 77.0

sst = sst_full.sel(longitude=slice(None, ARABIAN_LON_MAX))
ssh = ssh_full.sel(longitude=slice(None, ARABIAN_LON_MAX))
glorys = glorys_full.sel(longitude=slice(None, ARABIAN_LON_MAX))

print("Arabian Sea subset shapes:")
print("SST:", sst.shape)
print("SSH:", ssh.shape)
print("GLORYS:", glorys.shape)

# ---------- Build clusters specific to the Arabian Sea only ----------
print("\nComputing average SST/SSH per grid point (Arabian Sea only)...")
sst_avg = sst.mean(dim="time").values
ssh_avg = ssh.mean(dim="time").values

lat_grid, lon_grid = np.meshgrid(sst.longitude.values, sst.latitude.values)
valid_mask = ~np.isnan(sst_avg) & ~np.isnan(ssh_avg)

cluster_features = np.stack([
    sst_avg[valid_mask],
    ssh_avg[valid_mask],
    lat_grid[valid_mask],
    lon_grid[valid_mask],
], axis=1)

cluster_features_norm = (cluster_features - cluster_features.mean(axis=0)) / cluster_features.std(axis=0)

N_CLUSTERS = 5
print(f"Running k-means to find {N_CLUSTERS} natural regions within the Arabian Sea...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels_flat = kmeans.fit_predict(cluster_features_norm)

cluster_map = np.full(sst_avg.shape, -1)
cluster_map[valid_mask] = cluster_labels_flat

print("\nCluster sizes:")
for c in range(N_CLUSTERS):
    print(f"  Cluster {c}: {(cluster_labels_flat == c).sum()} points")

np.save("data/cluster_map_arabian.npy", cluster_map)
print("\nSaved Arabian Sea cluster map to data/cluster_map_arabian.npy")

# ---------- Build training dataset ----------
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

print(f"\nTrain samples: {X_train.shape[0]:,}")
print(f"Val samples:   {X_val.shape[0]:,}")
print(f"Test samples:  {X_test.shape[0]:,}")

np.savez(
    "data/dataset_arabian.npz",
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
)
print("\nSaved dataset to data/dataset_arabian.npz")