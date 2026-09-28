import xarray as xr
import numpy as np
import pandas as pd

sst = xr.open_dataarray("data/processed_sst_4yr.nc")
ssh = xr.open_dataarray("data/processed_ssh_4yr.nc")
glorys = xr.open_dataarray("data/processed_glorys_15depth_4yr.nc")
clim = xr.open_dataarray("data/glorys_climatology_trainonly.nc")

print("SST shape:", sst.shape)
print("SSH shape:", ssh.shape)
print("GLORYS shape:", glorys.shape)
print("Climatology shape:", clim.shape)

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

    for date in dates:
        doy = pd.Timestamp(date).dayofyear
        clim_day = clim.sel(dayofyear=doy).values

        sst_day = sst.sel(time=date).values
        ssh_day = ssh.sel(time=date).values
        glorys_day = glorys.sel(time=date).values

        sst_flat = sst_day.flatten()
        ssh_flat = ssh_day.flatten()
        glorys_flat = glorys_day.reshape(glorys_day.shape[0], -1).T
        clim_flat = clim_day.reshape(clim_day.shape[0], -1).T

        valid = (
            ~np.isnan(sst_flat) & ~np.isnan(ssh_flat) &
            ~np.isnan(glorys_flat).any(axis=1) & ~np.isnan(clim_flat).any(axis=1)
        )

        X_day = np.concatenate([
            np.stack([sst_flat[valid], ssh_flat[valid]], axis=1),
            clim_flat[valid],
        ], axis=1)

        y_day = glorys_flat[valid] - clim_flat[valid]

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
    "data/dataset_anomaly.npz",
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
)
print("\nSaved leak-free anomaly dataset to data/dataset_anomaly.npz")