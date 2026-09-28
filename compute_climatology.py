import xarray as xr
import numpy as np
import pandas as pd

print("Loading 4-year GLORYS data...")
glorys = xr.open_dataarray("data/processed_glorys_15depth_4yr.nc")

# Use ONLY the training portion of the data to build climatology - the
# same 75% split used everywhere else in this project. This avoids the
# test period ever influencing the "expected normal" values.
n_time = glorys.shape[0]
train_end = int(n_time * 0.75)
glorys_train_only = glorys.isel(time=slice(0, train_end))

print(f"Full dataset: {n_time} days. Using only first {train_end} days (training period) for climatology.")

n_time_tr, n_depth, n_lat, n_lon = glorys_train_only.shape

dates = pd.DatetimeIndex(glorys_train_only.time.values)
doy = dates.dayofyear.values.astype(np.float64)

omega = 2 * np.pi * doy / 365.25
X = np.stack([
    np.ones_like(omega),
    np.cos(omega), np.sin(omega),
    np.cos(2 * omega), np.sin(2 * omega),
], axis=1)

print("Fitting smooth seasonal climatology from training data only...")
Y = glorys_train_only.values.reshape(n_time_tr, -1)
valid_cols = ~np.isnan(Y).any(axis=0)

coeffs = np.full((5, Y.shape[1]), np.nan)
coeffs[:, valid_cols], _, _, _ = np.linalg.lstsq(X, Y[:, valid_cols], rcond=None)

doy_full = np.arange(1, 367)
omega_full = 2 * np.pi * doy_full / 365.25
X_full = np.stack([
    np.ones_like(omega_full),
    np.cos(omega_full), np.sin(omega_full),
    np.cos(2 * omega_full), np.sin(2 * omega_full),
], axis=1)

clim_flat = X_full @ coeffs
clim = clim_flat.reshape(366, n_depth, n_lat, n_lon)

clim_da = xr.DataArray(
    clim,
    dims=["dayofyear", "depth", "latitude", "longitude"],
    coords={
        "dayofyear": doy_full,
        "depth": glorys.depth.values,
        "latitude": glorys.latitude.values,
        "longitude": glorys.longitude.values,
    },
)

print("\nSanity check - climatology at ~13N 90E, 0m depth, across the year:")
sample = clim_da.sel(latitude=13, longitude=90, depth=0, method="nearest")
print(f"  Day 1 (Jan):  {float(sample.sel(dayofyear=1)):.2f} C")
print(f"  Day 100 (Apr): {float(sample.sel(dayofyear=100)):.2f} C")
print(f"  Day 200 (Jul): {float(sample.sel(dayofyear=200)):.2f} C")
print(f"  Day 300 (Oct): {float(sample.sel(dayofyear=300)):.2f} C")

clim_da.to_netcdf("data/glorys_climatology_trainonly.nc")
print("\nSaved leak-free climatology to data/glorys_climatology_trainonly.nc")