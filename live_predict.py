import copernicusmarine
import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os
from datetime import datetime, timedelta, timezone

depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BAY_LON_MIN = 77.0
LIVE_DIR = "data/live_cache"
os.makedirs(LIVE_DIR, exist_ok=True)


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


# ---------- Load both models once, at import time ----------
print("Loading trained models...")
bay_model = MLP(n_inputs=7).to(device)
bay_model.load_state_dict(torch.load("data/best_mlp_cluster.pt"))
bay_model.eval()
bay_norm = np.load("data/normalization_stats_cluster.npz")
bay_correction = dict(zip(
    pd.read_csv("data/depth_bias_correction_cluster.csv")["depth"],
    pd.read_csv("data/depth_bias_correction_cluster.csv")["correction_c"]
))
bay_cluster_map = np.load("data/cluster_map.npy")
_bay_grid = xr.open_dataarray("data/processed_sst_4yr.nc")
bay_grid_lats, bay_grid_lons = _bay_grid.latitude.values, _bay_grid.longitude.values

arabian_model = MLP(n_inputs=16).to(device)
arabian_model.load_state_dict(torch.load("data/best_mlp_arabian_eddy.pt"))
arabian_model.eval()
arabian_norm = np.load("data/normalization_stats_arabian_eddy.npz")
arabian_correction = dict(zip(
    pd.read_csv("data/depth_bias_correction_arabian_eddy.csv")["depth"],
    pd.read_csv("data/depth_bias_correction_arabian_eddy.csv")["correction_c"]
))
arabian_cluster_map = np.load("data/cluster_map_arabian_curl.npy")
_ar_grid = xr.open_dataarray("data/processed_sst_full.nc").sel(longitude=slice(None, BAY_LON_MIN))
ar_grid_lats, ar_grid_lons = _ar_grid.latitude.values, _ar_grid.longitude.values
print("Models loaded.\n")


def _get_cluster_id(lat, lon, cluster_map, grid_lats, grid_lons):
    lat_idx = np.argmin(np.abs(grid_lats - lat))
    lon_idx = np.argmin(np.abs(grid_lons - lon))
    return int(cluster_map[lat_idx, lon_idx])


def _fetch_latest(dataset_id, variable, lat, lon, filename, max_days_back=14, hourly=False):
    """
    Tries progressively older dates until the live dataset actually has data -
    handles the fact that different variables have different real-world lag
    (we found SST/SSH ~2 days, SSS ~7 days) without hardcoding any single value.
    """
    today = datetime.now(timezone.utc).date()
    out_path = f"{LIVE_DIR}/{filename}"

    for days_back in range(2, max_days_back + 1):
        target_date = today - timedelta(days=days_back)
        start = f"{target_date}T12:00:00" if hourly else str(target_date)
        end = start
        try:
            copernicusmarine.subset(
                dataset_id=dataset_id, variables=[variable],
                minimum_longitude=lon - 0.5, maximum_longitude=lon + 0.5,
                minimum_latitude=lat - 0.5, maximum_latitude=lat + 0.5,
                start_datetime=start, end_datetime=end,
                output_directory=LIVE_DIR, output_filename=filename,
            )
            return out_path, target_date
        except Exception:
            continue
    raise RuntimeError(f"Could not fetch {variable} from {dataset_id} within {max_days_back} days")


def predict_live(lat, lon):
    """
    THE MAIN FUNCTION. Give it a real point in the Bay of Bengal or Arabian Sea,
    get back a live prediction using the freshest real satellite data available,
    run through the correct trained, validated model. No Argo comparison is
    possible here (real measurements for "today" don't exist yet) - that's an
    honest, expected limitation, not a bug.
    """
    if lon < BAY_LON_MIN:
        region = "Arabian Sea"

        sst_path, sst_date = _fetch_latest("METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2", "analysed_sst", lat, lon, "live_sst.nc")
        ssh_path, ssh_date = _fetch_latest("cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D", "sla", lat, lon, "live_ssh.nc")
        sss_path, sss_date = _fetch_latest("cmems_obs-mob_glo_phy-sss_nrt_multi_P1D", "sos", lat, lon, "live_sss.nc")
        mld_path, mld_date = _fetch_latest("cmems_mod_glo_phy_anfc_0.083deg_P1D-m", "mlotst", lat, lon, "live_mld.nc")
        curl_path, curl_date = _fetch_latest("cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H", "stress_curl", lat, lon, "live_curl.nc", hourly=True)

        sst_val = float(xr.open_dataset(sst_path)["analysed_sst"].sel(latitude=lat, longitude=lon, method="nearest").isel(time=0))
        if sst_val > 100:
            sst_val -= 273.15
        ssh_val = float(xr.open_dataset(ssh_path)["sla"].sel(latitude=lat, longitude=lon, method="nearest").isel(time=0))
        sss_val = float(xr.open_dataset(sss_path)["sos"].squeeze(drop=True).sel(latitude=lat, longitude=lon, method="nearest"))
        mld_val = float(xr.open_dataset(mld_path)["mlotst"].sel(latitude=lat, longitude=lon, method="nearest").isel(time=0))
        curl_val = float(xr.open_dataset(curl_path)["stress_curl"].sel(latitude=lat, longitude=lon, method="nearest").isel(time=0))

        ssh_patch = xr.open_dataset(ssh_path)["sla"].isel(time=0)
        g, omega = 9.81, 7.2921e-5
        f = 2 * omega * np.sin(np.deg2rad(lat))
        lat_rad = np.deg2rad(ssh_patch.latitude)
        dx = 111320 * np.cos(lat_rad)
        dy = 110540
        d2_dx2 = ssh_patch.differentiate("longitude").differentiate("longitude") / (dx ** 2)
        d2_dy2 = ssh_patch.differentiate("latitude").differentiate("latitude") / (dy ** 2)
        vort_val = float(((9.81 / f) * (d2_dx2 + d2_dy2)).sel(latitude=lat, longitude=lon, method="nearest"))

        cluster_id = _get_cluster_id(lat, lon, arabian_cluster_map, ar_grid_lats, ar_grid_lons)
        if cluster_id == -1:
            return None
        cluster_onehot = np.zeros(10)
        cluster_onehot[cluster_id] = 1
        x = np.concatenate([[sst_val, ssh_val, curl_val, mld_val, sss_val, vort_val], cluster_onehot]).reshape(1, -1).astype(np.float32)
        x_norm = (x - arabian_norm["X_mean"]) / arabian_norm["X_std"]
        x_t = torch.tensor(x_norm, dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_norm = arabian_model(x_t).cpu().numpy()[0]
        pred_temp = pred_norm * arabian_norm["y_std"] + arabian_norm["y_mean"]
        pred_corrected = [round(float(pred_temp[i] - arabian_correction[d]), 2) for i, d in enumerate(depths)]

        surface_state = {
            "sst_c": round(sst_val, 2), "ssh_m": round(ssh_val, 3),
            "wind_stress_curl": curl_val, "mld_m": round(mld_val, 1),
            "sss_psu": round(sss_val, 2), "eddy_vorticity": vort_val,
        }
        data_dates = {"sst": str(sst_date), "ssh": str(ssh_date), "sss": str(sss_date), "mld": str(mld_date), "curl": str(curl_date)}

    else:
        region = "Bay of Bengal"

        sst_path, sst_date = _fetch_latest("METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2", "analysed_sst", lat, lon, "live_sst_bay.nc")
        ssh_path, ssh_date = _fetch_latest("cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D", "sla", lat, lon, "live_ssh_bay.nc")

        sst_val = float(xr.open_dataset(sst_path)["analysed_sst"].sel(latitude=lat, longitude=lon, method="nearest").isel(time=0))
        if sst_val > 100:
            sst_val -= 273.15
        ssh_val = float(xr.open_dataset(ssh_path)["sla"].sel(latitude=lat, longitude=lon, method="nearest").isel(time=0))

        cluster_id = _get_cluster_id(lat, lon, bay_cluster_map, bay_grid_lats, bay_grid_lons)
        if cluster_id == -1:
            return None
        cluster_onehot = np.zeros(5)
        cluster_onehot[cluster_id] = 1
        x = np.concatenate([[sst_val, ssh_val], cluster_onehot]).reshape(1, -1).astype(np.float32)
        x_norm = (x - bay_norm["X_mean"]) / bay_norm["X_std"]
        x_t = torch.tensor(x_norm, dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_norm = bay_model(x_t).cpu().numpy()[0]
        pred_temp = pred_norm * bay_norm["y_std"] + bay_norm["y_mean"]
        pred_corrected = [round(float(pred_temp[i] - bay_correction[d]), 2) for i, d in enumerate(depths)]

        surface_state = {"sst_c": round(sst_val, 2), "ssh_m": round(ssh_val, 3)}
        data_dates = {"sst": str(sst_date), "ssh": str(ssh_date)}

    return {
        "location": {"lat": lat, "lon": lon, "region": region},
        "mode": "live",
        "data_dates": data_dates,
        "surface_state": surface_state,
        "profile": {"depths_m": depths, "predicted_temp_c": pred_corrected},
    }


if __name__ == "__main__":
    import json
    print("Testing live prediction - Bay of Bengal point...")
    result_bay = predict_live(15.0, 88.0)
    print(json.dumps(result_bay, indent=2))

    print("\nTesting live prediction - Arabian Sea point...")
    result_arabian = predict_live(17.0, 58.0)
    print(json.dumps(result_arabian, indent=2))
    