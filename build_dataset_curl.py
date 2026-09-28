import xarray as xr
import numpy as np
import os
from sklearn.cluster import KMeans

ARABIAN_LON_MAX = 77.0

sst_full = xr.open_dataarray("data/processed_sst_full.nc")
ssh_full = xr.open_dataarray("data/processed_ssh_full.nc")
glorys_full = xr.open_dataarray("data/processed_glorys_15depth_full.nc")

sst = sst_full.sel(longitude=slice(None, ARABIAN_LON_MAX))
ssh = ssh_full.sel(longitude=slice(None, ARABIAN_LON_MAX))
glorys = glorys_full.sel(longitude=slice(None, ARABIAN_LON_MAX))

print("Loading CCMP wind files and computing wind stress curl...")
wind_files = sorted([f"data/winds_full/{f}" for f in os.listdir("data/winds_full") if f.endswith(".nc")])
winds_raw = xr.open_mfdataset(wind_files, combine="by_coords")

# Average the 4 daily 6-hourly readings into 1 daily value, same approach
# as our earlier wind experiment
winds_daily = winds_raw.resample(time="1D").mean()

u = winds_daily["uwnd"]
v = winds_daily["vwnd"]

# Wind stress magnitude, using the standard oceanographic formula:
# tau = air_density * drag_coefficient * wind_speed * wind_component
# (a simplified constant drag coefficient is standard practice for this scale of analysis)
rho_air = 1.225  # kg/m^3
Cd = 0.0013      # standard drag coefficient
wind_speed = np.sqrt(u**2 + v**2)
tau_x = rho_air * Cd * wind_speed * u
tau_y = rho_air * Cd * wind_speed * v

# Curl = d(tau_y)/dx - d(tau_x)/dy - the "spin" of the wind stress field
# Computed using real distances (not raw degrees), since degrees of
# longitude represent different real distances at different latitudes
lat_rad = np.deg2rad(tau_y.latitude)
dx = 111320 * np.cos(lat_rad)  # meters per degree longitude at this latitude
dy = 110540                     # meters per degree latitude (roughly constant)

dtau_y_dx = tau_y.differentiate("longitude") / dx
dtau_x_dy = tau_x.differentiate("latitude") / dy

curl = dtau_y_dx - dtau_x_dy
curl.name = "wind_stress_curl"

print("Regridding curl to match our 0.25 degree grid...")
target_lat = sst.latitude.values
target_lon = sst.longitude.values
curl_regrid = curl.interp(latitude=target_lat, longitude=target_lon)

print("Curl shape:", curl_regrid.shape)
print("\nSample curl values (Oman coast, ~17N 58E, should show real variation across seasons):")
for date in ["2020-01-15", "2020-07-15"]:
    val = float(curl_regrid.sel(latitude=17, longitude=58, time=date, method="nearest"))
    print(f"  {date}: {val:.2e}")

curl_regrid.to_netcdf("data/processed_curl_arabian.nc")
print("\nSaved to data/processed_curl_arabian.nc")

# ---------- Build dataset: SST, SSH, curl, + fine clustering ----------
print("\nComputing average SST/SSH per grid point for clustering...")
sst_avg = sst.mean(dim="time").values
ssh_avg = ssh.mean(dim="time").values

lat_grid, lon_grid = np.meshgrid(sst.longitude.values, sst.latitude.values)
valid_mask = ~np.isnan(sst_avg) & ~np.isnan(ssh_avg)

cluster_features = np.stack([
    sst_avg[valid_mask], ssh_avg[valid_mask],
    lat_grid[valid_mask], lon_grid[valid_mask],
], axis=1)
cluster_features_norm = (cluster_features - cluster_features.mean(axis=0)) / cluster_features.std(axis=0)

N_CLUSTERS = 10
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels_flat = kmeans.fit_predict(cluster_features_norm)
cluster_map = np.full(sst_avg.shape, -1)
cluster_map[valid_mask] = cluster_labels_flat
np.save("data/cluster_map_arabian_curl.npy", cluster_map)

all_dates = sst.time.values
n_days = len(all_dates)
train_end = int(n_days * 0.75)
val_end = int(n_days * 0.88)
train_dates = all_dates[:train_end]
val_dates = all_dates[train_end:val_end]
test_dates = all_dates[val_end:]

print(f"\nTrain: {len(train_dates)} days, Val: {len(val_dates)} days, Test: {len(test_dates)} days")


def build_samples(dates):
    X_list, y_list = [], []
    cluster_flat_full = cluster_map.flatten()

    for date in dates:
        sst_day = sst.sel(time=date).values
        ssh_day = ssh.sel(time=date).values
        curl_day = curl_regrid.sel(time=date, method="nearest").values
        glorys_day = glorys.sel(time=date).values

        sst_flat = sst_day.flatten()
        ssh_flat = ssh_day.flatten()
        curl_flat = curl_day.flatten()
        glorys_flat = glorys_day.reshape(glorys_day.shape[0], -1).T

        valid = (
            ~np.isnan(sst_flat) & ~np.isnan(ssh_flat) & ~np.isnan(curl_flat) &
            ~np.isnan(glorys_flat).any(axis=1) & (cluster_flat_full != -1)
        )

        cluster_onehot = np.zeros((valid.sum(), N_CLUSTERS))
        cluster_ids_here = cluster_flat_full[valid]
        cluster_onehot[np.arange(len(cluster_ids_here)), cluster_ids_here] = 1

        X_day = np.concatenate([
            np.stack([sst_flat[valid], ssh_flat[valid], curl_flat[valid]], axis=1),
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

np.savez(
    "data/dataset_arabian_curl.npz",
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
)
print("\nSaved dataset to data/dataset_arabian_curl.npz")