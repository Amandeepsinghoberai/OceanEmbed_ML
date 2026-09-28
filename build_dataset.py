import xarray as xr
import numpy as np

sst = xr.open_dataarray("data/processed_sst.nc")
ssh = xr.open_dataarray("data/processed_ssh.nc")
sss = xr.open_dataarray("data/processed_sss.nc")
glorys = xr.open_dataarray("data/processed_glorys_15depth.nc")

print("SST shape:", sst.shape)
print("SSH shape:", ssh.shape)
print("SSS shape:", sss.shape)
print("GLORYS shape:", glorys.shape)

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
    Each sample: input = [sst, ssh, sss] (3 values)
    target = [15 temperatures]
    """
    X_list, y_list = [], []
    for date in dates:
        sst_day = sst.sel(time=date).values
        ssh_day = ssh.sel(time=date).values
        sss_day = sss.sel(time=date).values
        glorys_day = glorys.sel(time=date).values

        sst_flat = sst_day.flatten()
        ssh_flat = ssh_day.flatten()
        sss_flat = sss_day.flatten()
        glorys_flat = glorys_day.reshape(glorys_day.shape[0], -1).T

        valid = (
            ~np.isnan(sst_flat) & ~np.isnan(ssh_flat) & ~np.isnan(sss_flat) &
            ~np.isnan(glorys_flat).any(axis=1)
        )

        X_day = np.stack([sst_flat[valid], ssh_flat[valid], sss_flat[valid]], axis=1)
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

print(f"\nTrain samples: {X_train.shape[0]}")
print(f"Val samples:   {X_val.shape[0]}")
print(f"Test samples:  {X_test.shape[0]}")

np.savez(
    "data/dataset_sss.npz",
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
)
print("\nSaved dataset to data/dataset_sss.npz")