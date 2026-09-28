import copernicusmarine
import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime, timedelta, timezone

depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_LAT, TEST_LON = 17.0, 58.0  # Oman coast, our known upwelling test point

today = datetime.now(timezone.utc).date()
target_date = today - timedelta(days=7)

print(f"Fetching all 5 live Arabian Sea inputs for {target_date}...\n")

copernicusmarine.subset(
    dataset_id="METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2",
    variables=["analysed_sst"],
    minimum_longitude=TEST_LON - 0.5, maximum_longitude=TEST_LON + 0.5,
    minimum_latitude=TEST_LAT - 0.5, maximum_latitude=TEST_LAT + 0.5,
    start_datetime=str(target_date), end_datetime=str(target_date),
    output_directory="data/live_test", output_filename="ar_live_sst.nc",
)
copernicusmarine.subset(
    dataset_id="cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D",
    variables=["sla"],
    minimum_longitude=TEST_LON - 0.5, maximum_longitude=TEST_LON + 0.5,
    minimum_latitude=TEST_LAT - 0.5, maximum_latitude=TEST_LAT + 0.5,
    start_datetime=str(target_date), end_datetime=str(target_date),
    output_directory="data/live_test", output_filename="ar_live_ssh.nc",
)
copernicusmarine.subset(
    dataset_id="cmems_obs-mob_glo_phy-sss_nrt_multi_P1D",
    variables=["sos"],
    minimum_longitude=TEST_LON - 0.5, maximum_longitude=TEST_LON + 0.5,
    minimum_latitude=TEST_LAT - 0.5, maximum_latitude=TEST_LAT + 0.5,
    start_datetime=str(target_date), end_datetime=str(target_date),
    output_directory="data/live_test", output_filename="ar_live_sss.nc",
)
copernicusmarine.subset(
    dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
    variables=["mlotst"],
    minimum_longitude=TEST_LON - 0.5, maximum_longitude=TEST_LON + 0.5,
    minimum_latitude=TEST_LAT - 0.5, maximum_latitude=TEST_LAT + 0.5,
    start_datetime=str(target_date), end_datetime=str(target_date),
    output_directory="data/live_test", output_filename="ar_live_mld.nc",
)
copernicusmarine.subset(
    dataset_id="cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H",
    variables=["stress_curl"],
    minimum_longitude=TEST_LON - 0.5, maximum_longitude=TEST_LON + 0.5,
    minimum_latitude=TEST_LAT - 0.5, maximum_latitude=TEST_LAT + 0.5,
    start_datetime=f"{target_date}T12:00:00", end_datetime=f"{target_date}T12:00:00",
    output_directory="data/live_test", output_filename="ar_live_curl.nc",
)

print("\nAll 5 downloaded. Extracting values...\n")

sst_val = float(xr.open_dataset("data/live_test/ar_live_sst.nc")["analysed_sst"].sel(latitude=TEST_LAT, longitude=TEST_LON, method="nearest").isel(time=0))
if sst_val > 100:
    sst_val -= 273.15
ssh_val = float(xr.open_dataset("data/live_test/ar_live_ssh.nc")["sla"].sel(latitude=TEST_LAT, longitude=TEST_LON, method="nearest").isel(time=0))
sss_val = float(xr.open_dataset("data/live_test/ar_live_sss.nc")["sos"].squeeze(drop=True).sel(latitude=TEST_LAT, longitude=TEST_LON, method="nearest"))
mld_val = float(xr.open_dataset("data/live_test/ar_live_mld.nc")["mlotst"].sel(latitude=TEST_LAT, longitude=TEST_LON, method="nearest").isel(time=0))
curl_val = float(xr.open_dataset("data/live_test/ar_live_curl.nc")["stress_curl"].sel(latitude=TEST_LAT, longitude=TEST_LON, method="nearest").isel(time=0))

print(f"Live SST:  {sst_val:.2f} C")
print(f"Live SSH:  {ssh_val:.3f} m")
print(f"Live SSS:  {sss_val:.2f} PSU")
print(f"Live MLD:  {mld_val:.1f} m")
print(f"Live curl: {curl_val:.2e}  (sanity check: should be roughly 1e-6 to 1e-5 in magnitude)")

# Eddy vorticity: same geostrophic calculation as training, computed from a small
# patch of live SSH around the point (needs neighboring points for the derivative)
ssh_patch = xr.open_dataset("data/live_test/ar_live_ssh.nc")["sla"].isel(time=0)
g, omega = 9.81, 7.2921e-5
f = 2 * omega * np.sin(np.deg2rad(TEST_LAT))
lat_rad = np.deg2rad(ssh_patch.latitude)
dx = 111320 * np.cos(lat_rad)
dy = 110540
d2_dx2 = ssh_patch.differentiate("longitude").differentiate("longitude") / (dx ** 2)
d2_dy2 = ssh_patch.differentiate("latitude").differentiate("latitude") / (dy ** 2)
vort_field = (g / f) * (d2_dx2 + d2_dy2)
vort_val = float(vort_field.sel(latitude=TEST_LAT, longitude=TEST_LON, method="nearest"))
print(f"Live vorticity: {vort_val:.2e}")

class MLP(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 48), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(48, 48), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(48, 15)
        )
    def forward(self, x):
        return self.net(x)

model = MLP(n_inputs=16).to(device)
model.load_state_dict(torch.load("data/best_mlp_arabian_eddy.pt"))
model.eval()

norm = np.load("data/normalization_stats_arabian_eddy.npz")
correction_df = pd.read_csv("data/depth_bias_correction_arabian_eddy.csv")
correction = dict(zip(correction_df["depth"], correction_df["correction_c"]))

cluster_map = np.load("data/cluster_map_arabian_curl.npy")
ar_sst_grid = xr.open_dataarray("data/processed_sst_full.nc").sel(longitude=slice(None, 77.0))
grid_lats = ar_sst_grid.latitude.values
grid_lons = ar_sst_grid.longitude.values
lat_idx = np.argmin(np.abs(grid_lats - TEST_LAT))
lon_idx = np.argmin(np.abs(grid_lons - TEST_LON))
cluster_id = int(cluster_map[lat_idx, lon_idx])

cluster_onehot = np.zeros(10)
cluster_onehot[cluster_id] = 1
x = np.concatenate([[sst_val, ssh_val, curl_val, mld_val, sss_val, vort_val], cluster_onehot]).reshape(1, -1).astype(np.float32)
x_norm = (x - norm["X_mean"]) / norm["X_std"]
x_t = torch.tensor(x_norm, dtype=torch.float32).to(device)

with torch.no_grad():
    pred_norm = model(x_t).cpu().numpy()[0]
pred_temp = pred_norm * norm["y_std"] + norm["y_mean"]
pred_corrected = [pred_temp[i] - correction[d] for i, d in enumerate(depths)]

print(f"\n{'='*50}")
print(f"LIVE PREDICTION — {target_date} — Arabian Sea ({TEST_LAT}, {TEST_LON})")
print(f"{'='*50}")
for d, t in zip(depths, pred_corrected):
    print(f"  {d:>5}m: {t:.2f} C")