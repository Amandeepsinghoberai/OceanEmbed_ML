import xarray as xr
import numpy as np
import gsw

sst = xr.open_dataarray("data/processed_sst_4yr.nc")
ssh = xr.open_dataarray("data/processed_ssh_4yr.nc")
glorys = xr.open_dataarray("data/processed_glorys_15depth_4yr_v2.nc")

# SSS was only downloaded for 2023 (single year) - reload that and note
# this experiment is scoped to the 1-year SSS window, since we haven't
# pulled multi-year SSS yet
sss = xr.open_dataarray("data/processed_sss.nc")

print("SST shape:", sst.shape)
print("SSH shape:", ssh.shape)
print("SSS shape:", sss.shape)
print("GLORYS shape:", glorys.shape)

# Align everything to the dates where SSS actually exists (2023 only)
common_dates = sst.time.values[np.isin(sst.time.values, sss.time.values)]
print(f"\nOverlapping dates (where SSS is available): {len(common_dates)}")

sst_common = sst.sel(time=common_dates)
ssh_common = ssh.sel(time=common_dates)
sss_common = sss.sel(time=common_dates)
glorys_common = glorys.sel(time=common_dates)

n_days = len(common_dates)
train_end = int(n_days * 0.75)
val_end = int(n_days * 0.88)
train_dates = common_dates[:train_end]
val_dates = common_dates[train_end:val_end]
test_dates = common_dates[val_end:]

print(f"Train: {len(train_dates)} days")
print(f"Val:   {len(val_dates)} days")
print(f"Test:  {len(test_dates)} days")


def build_samples(dates):
    X_list, y_list = [], []
    for date in dates:
        sst_day = sst_common.sel(time=date).values
        ssh_day = ssh_common.sel(time=date).values
        sss_day = sss_common.sel(time=date).values
        glorys_day = glorys_common.sel(time=date).values

        # Compute sea surface density using the real oceanographic equation
        # of state - "practical salinity" and "conservative temperature" are
        # the standard forms gsw expects, at pressure=0 for the surface
        density = gsw.rho(sss_day, sst_day, 0)  # kg/m^3, roughly 1020-1028

        sst_flat = sst_day.flatten()
        ssh_flat = ssh_day.flatten()
        density_flat = density.flatten()
        glorys_flat = glorys_day.reshape(glorys_day.shape[0], -1).T

        valid = (
            ~np.isnan(sst_flat) & ~np.isnan(ssh_flat) & ~np.isnan(density_flat) &
            ~np.isnan(glorys_flat).any(axis=1)
        )

        X_day = np.stack([sst_flat[valid], ssh_flat[valid], density_flat[valid]], axis=1)
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

print("\nSample density value (should be around 1020-1028 kg/m^3):")
print(X_train[0, 2])

np.savez(
    "data/dataset_density.npz",
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
)
print("\nSaved dataset to data/dataset_density.npz")