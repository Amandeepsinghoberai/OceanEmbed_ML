import xarray as xr
import numpy as np

ARABIAN_LON_MAX = 77.0

sst = xr.open_dataarray("data/processed_sst_full.nc").sel(longitude=slice(None, ARABIAN_LON_MAX))
ssh = xr.open_dataarray("data/processed_ssh_full.nc").sel(longitude=slice(None, ARABIAN_LON_MAX))
glorys = xr.open_dataarray("data/processed_glorys_15depth_full.nc").sel(longitude=slice(None, ARABIAN_LON_MAX))
curl = xr.open_dataarray("data/processed_curl_arabian.nc")
mld = xr.open_dataarray("data/processed_mld_arabian.nc")
sss = xr.open_dataarray("data/processed_sss_arabian.nc")
cluster_map = np.load("data/cluster_map_arabian_curl.npy")

print("Computing geostrophic vorticity (eddy indicator) from SSH...")

g = 9.81  # gravity, m/s^2
omega = 7.2921e-5  # Earth's rotation rate, rad/s
lat_rad = np.deg2rad(ssh.latitude)
f = 2 * omega * np.sin(lat_rad)
f = f.where(np.abs(f) > 1e-6, 1e-6)  # avoid blow-up very close to the equator

lat_rad_full = np.deg2rad(ssh.latitude)
dx = 111320 * np.cos(lat_rad_full)  # meters per degree longitude at this latitude
dy = 110540                          # meters per degree latitude

# Second derivatives of SSH in each direction
d2ssh_dx2 = ssh.differentiate("longitude").differentiate("longitude") / (dx ** 2)
d2ssh_dy2 = ssh.differentiate("latitude").differentiate("latitude") / (dy ** 2)

# Relative vorticity of the geostrophic current, derived from sea surface height -
# a standard oceanographic relationship: zeta ~ (g/f) * Laplacian(SSH)
# Positive = cyclonic rotation, negative = anticyclonic (like the Great Whirl)
vorticity = (g / f) * (d2ssh_dx2 + d2ssh_dy2)
vorticity.name = "vorticity"

print("Vorticity shape:", vorticity.shape)
print("\nSample vorticity values (Great Whirl region, ~9N 52E, should show real signal in summer monsoon):")
for date in ["2020-01-15", "2020-08-15"]:
    val = float(vorticity.sel(latitude=9, longitude=52, time=date, method="nearest"))
    print(f"  {date}: {val:.2e}")

vorticity.to_netcdf("data/processed_vorticity_arabian.nc")
print("\nSaved to data/processed_vorticity_arabian.nc")

N_CLUSTERS = 10
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
        curl_day = curl.sel(time=date, method="nearest").values
        mld_day = mld.sel(time=date, method="nearest").values
        sss_day = sss.sel(time=date, method="nearest").values
        vort_day = vorticity.sel(time=date).values
        glorys_day = glorys.sel(time=date).values

        sst_flat = sst_day.flatten()
        ssh_flat = ssh_day.flatten()
        curl_flat = curl_day.flatten()
        mld_flat = mld_day.flatten()
        sss_flat = sss_day.flatten()
        vort_flat = vort_day.flatten()
        glorys_flat = glorys_day.reshape(glorys_day.shape[0], -1).T

        valid = (
            ~np.isnan(sst_flat) & ~np.isnan(ssh_flat) & ~np.isnan(curl_flat) &
            ~np.isnan(mld_flat) & ~np.isnan(sss_flat) & ~np.isnan(vort_flat) &
            ~np.isnan(glorys_flat).any(axis=1) & (cluster_flat_full != -1)
        )

        cluster_onehot = np.zeros((valid.sum(), N_CLUSTERS))
        cluster_ids_here = cluster_flat_full[valid]
        cluster_onehot[np.arange(len(cluster_ids_here)), cluster_ids_here] = 1

        X_day = np.concatenate([
            np.stack([sst_flat[valid], ssh_flat[valid], curl_flat[valid],
                      mld_flat[valid], sss_flat[valid], vort_flat[valid]], axis=1),
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
    "data/dataset_arabian_eddy.npz",
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
)
print("\nSaved dataset to data/dataset_arabian_eddy.npz")