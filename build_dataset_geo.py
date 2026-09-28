import xarray as xr
import numpy as np
import pandas as pd

sst = xr.open_dataarray("data/processed_sst_4yr.nc")
ssh = xr.open_dataarray("data/processed_ssh_4yr.nc")
glorys = xr.open_dataarray("data/processed_glorys_15depth_4yr.nc")

print("SST shape:", sst.shape)
print("SSH shape:", ssh.shape)
print("GLORYS shape:", glorys.shape)

lats = sst.latitude.values
lons = sst.longitude.values

# Build 2D grids of lat/lon matching the spatial shape, so every point
# knows WHERE it is, not just what its surface values are
lon_grid, lat_grid = np.meshgrid(lons, lats)
lat_flat_full = lat_grid.flatten()
lon_flat_full = lon_grid.flatten()

all_dates = sst.time.values
n_days = len(all_dates)
train_end = int(n_days * 0.75)
val_end = int(n_days * 0.88)
train_dates = all_dates[:train_end]
val_dates = all_dates[train_end:val_end]
test_dates = all_dates[val_end:]

print(f"\nTrain: {len(train_dates)} days ({train_dates[0]} to {train_dates[-1]})")
print(f"Val:   {len(val_dates)} days ({val_dates[0]} to {val_dates[-1]})")
print(f"Test:  {len(test_dates)} days ({test_dates[0]} to {test_dates[-1]})")


def build_samples(dates):
    """
    Each sample: input = [sst, ssh, lat, lon, sin(day), cos(day)]  (6 values)
    target = [15 temperatures]

    Day-of-year is encoded as sin/cos so the model understands that
    Dec 31 and Jan 1 are next to each other, not opposite ends of a scale.
    """
    X_list, y_list = [], []

    for date in dates:
        day_of_year = pd.Timestamp(date).dayofyear
        day_sin = np.sin(2 * np.pi * day_of_year / 365.25)
        day_cos = np.cos(2 * np.pi * day_of_year / 365.25)

        sst_day = sst.sel(time=date).values
        ssh_day = ssh.sel(time=date).values
        glorys_day = glorys.sel(time=date).values

        sst_flat = sst_day.flatten()
        ssh_flat = ssh_day.flatten()
        glorys_flat = glorys_day.reshape(glorys_day.shape[0], -1).T

        valid = ~np.isnan(sst_flat) & ~np.isnan(ssh_flat) & ~np.isnan(glorys_flat).any(axis=1)

        n_valid = valid.sum()
        X_day = np.stack([
            sst_flat[valid],
            ssh_flat[valid],
            lat_flat_full[valid],
            lon_flat_full[valid],
            np.full(n_valid, day_sin),
            np.full(n_valid, day_cos),
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

print("\nSample input row (sst, ssh, lat, lon, day_sin, day_cos):")
print(X_train[0])

np.savez(
    "data/dataset_geo.npz",
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
)
print("\nSaved dataset to data/dataset_geo.npz")
