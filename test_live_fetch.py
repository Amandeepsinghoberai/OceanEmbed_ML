import copernicusmarine
import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime, timedelta, timezone

depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_LAT, TEST_LON = 15.0, 88.0

today = datetime.now(timezone.utc).date()
yesterday = today - timedelta(days=2)

print(f"Fetching live NRT SST/SSH for {yesterday}...")

copernicusmarine.subset(
    dataset_id="METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2",
    variables=["analysed_sst"],
    minimum_longitude=TEST_LON - 0.5, maximum_longitude=TEST_LON + 0.5,
    minimum_latitude=TEST_LAT - 0.5, maximum_latitude=TEST_LAT + 0.5,
    start_datetime=str(yesterday), end_datetime=str(yesterday),
    output_directory="data/live_test", output_filename="live_sst.nc",
)

copernicusmarine.subset(
    dataset_id="cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D",
    variables=["sla"],
    minimum_longitude=TEST_LON - 0.5, maximum_longitude=TEST_LON + 0.5,
    minimum_latitude=TEST_LAT - 0.5, maximum_latitude=TEST_LAT + 0.5,
    start_datetime=str(yesterday), end_datetime=str(yesterday),
    output_directory="data/live_test", output_filename="live_ssh.nc",
)

print("\nBoth downloaded. Now running through your actual trained model...\n")

sst_ds = xr.open_dataset("data/live_test/live_sst.nc")
ssh_ds = xr.open_dataset("data/live_test/live_ssh.nc")

sst_val = float(sst_ds["analysed_sst"].sel(latitude=TEST_LAT, longitude=TEST_LON, method="nearest").isel(time=0))
if sst_val > 100:
    sst_val -= 273.15
ssh_val = float(ssh_ds["sla"].sel(latitude=TEST_LAT, longitude=TEST_LON, method="nearest").isel(time=0))

print(f"Live SST at ({TEST_LAT}, {TEST_LON}): {sst_val:.2f} C")
print(f"Live SSH at ({TEST_LAT}, {TEST_LON}): {ssh_val:.3f} m")

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

model = MLP(n_inputs=7).to(device)
model.load_state_dict(torch.load("data/best_mlp_cluster.pt"))
model.eval()

norm = np.load("data/normalization_stats_cluster.npz")
correction_df = pd.read_csv("data/depth_bias_correction_cluster.csv")
correction = dict(zip(correction_df["depth"], correction_df["correction_c"]))

cluster_map = np.load("data/cluster_map.npy")
bay_sst_grid = xr.open_dataarray("data/processed_sst_4yr.nc")
grid_lats = bay_sst_grid.latitude.values
grid_lons = bay_sst_grid.longitude.values
lat_idx = np.argmin(np.abs(grid_lats - TEST_LAT))
lon_idx = np.argmin(np.abs(grid_lons - TEST_LON))
cluster_id = int(cluster_map[lat_idx, lon_idx])

cluster_onehot = np.zeros(5)
cluster_onehot[cluster_id] = 1
x = np.concatenate([[sst_val, ssh_val], cluster_onehot]).reshape(1, -1).astype(np.float32)
x_norm = (x - norm["X_mean"]) / norm["X_std"]
x_t = torch.tensor(x_norm, dtype=torch.float32).to(device)

with torch.no_grad():
    pred_norm = model(x_t).cpu().numpy()[0]
pred_temp = pred_norm * norm["y_std"] + norm["y_mean"]
pred_corrected = [pred_temp[i] - correction[d] for i, d in enumerate(depths)]

print(f"\n{'='*50}")
print(f"LIVE PREDICTION — {yesterday} — Bay of Bengal ({TEST_LAT}, {TEST_LON})")
print(f"{'='*50}")
for d, t in zip(depths, pred_corrected):
    print(f"  {d:>5}m: {t:.2f} C")