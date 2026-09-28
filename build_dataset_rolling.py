import xarray as xr
import numpy as np

sst = xr.open_dataarray("data/processed_sst.nc")
ssh = xr.open_dataarray("data/processed_ssh.nc")
current_u = xr.open_dataarray("data/processed_current_u.nc")
current_v = xr.open_dataarray("data/processed_current_v.nc")
wind_u = xr.open_dataarray("data/processed_wind_u.nc")
wind_v = xr.open_dataarray("data/processed_wind_v.nc")
glorys = xr.open_dataarray("data/processed_glorys_15depth.nc")

# Instead of today's single reading, use the trailing 7-day average -
# this represents the accumulated mixing effect, not just a snapshot
print("Computing 7-day rolling averages for currents and winds...")
current_u_roll = current_u.rolling(time=7, min_periods=1).mean()
current_v_roll = current_v.rolling(time=7, min_periods=1).mean()
wind_u_roll = wind_u.rolling(time=7, min_periods=1).mean()
wind_v_roll = wind_v.rolling(time=7, min_periods=1).mean()

all_dates = sst.time.values
n_days = len(all_dates)
train_end = int(n_days * 0.75)
val_end = int(n_days * 0.88)
train_dates = all_dates[:train_end]
val_dates = all_dates[train_end:val_end]
test_dates = all_dates[val_end:]

print(f"Train: {len(train_dates)} days, Val: {len(val_dates)} days, Test: {len(test_dates)} days")


def build_samples(dates):
    X_list, y_list = [], []
    for date in dates:
        sst_day = sst.sel(time=date).values
        ssh_day = ssh.sel(time=date).values
        cu_day = current_u_roll.sel(time=date).values
        cv_day = current_v_roll.sel(time=date).values
        wu_day = wind_u_roll.sel(time=date).values
        wv_day = wind_v_roll.sel(time=date).values
        glorys_day = glorys.sel(time=date).values

        sst_flat = sst_day.flatten()
        ssh_flat = ssh_day.flatten()
        cu_flat = cu_day.flatten()
        cv_flat = cv_day.flatten()
        wu_flat = wu_day.flatten()
        wv_flat = wv_day.flatten()
        glorys_flat = glorys_day.reshape(glorys_day.shape[0], -1).T

        valid = (
            ~np.isnan(sst_flat) & ~np.isnan(ssh_flat) &
            ~np.isnan(cu_flat) & ~np.isnan(cv_flat) &
            ~np.isnan(wu_flat) & ~np.isnan(wv_flat) &
            ~np.isnan(glorys_flat).any(axis=1)
        )

        X_day = np.stack([
            sst_flat[valid], ssh_flat[valid],
            cu_flat[valid], cv_flat[valid],
            wu_flat[valid], wv_flat[valid]
        ], axis=1)
        y_day = glorys_flat[valid]

        X_list.append(X_day)
        y_list.append(y_day)

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.float32)
    return X, y


print("Building training samples...")
X_train, y_train = build_samples(train_dates)
print("Building validation samples...")
X_val, y_val = build_samples(val_dates)
print("Building test samples...")
X_test, y_test = build_samples(test_dates)

print(f"\nTrain samples: {X_train.shape[0]}")
print(f"Val samples:   {X_val.shape[0]}")
print(f"Test samples:  {X_test.shape[0]}")

np.savez(
    "data/dataset_rolling.npz",
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
)
print("\nSaved side-experiment dataset to data/dataset_rolling.npz")