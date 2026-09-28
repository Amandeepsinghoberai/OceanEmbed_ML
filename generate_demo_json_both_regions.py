import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import json
import os

depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BAY_LON_MIN = 77.0


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


# ======================== BAY OF BENGAL SETUP ========================
bay_model = MLP(n_inputs=7).to(device)
bay_model.load_state_dict(torch.load("data/best_mlp_cluster.pt"))
bay_model.eval()

bay_norm = np.load("data/normalization_stats_cluster.npz")
bay_X_mean, bay_X_std = bay_norm["X_mean"], bay_norm["X_std"]
bay_y_mean, bay_y_std = bay_norm["y_mean"], bay_norm["y_std"]

bay_correction_df = pd.read_csv("data/depth_bias_correction_cluster.csv")
bay_correction = dict(zip(bay_correction_df["depth"], bay_correction_df["correction_c"]))

bay_sst = xr.open_dataarray("data/processed_sst_4yr.nc")
bay_ssh = xr.open_dataarray("data/processed_ssh_4yr.nc")
bay_cluster_map = np.load("data/cluster_map.npy")
bay_grid_lats = bay_sst.latitude.values
bay_grid_lons = bay_sst.longitude.values

# Ensemble - trained ONLY on Bay of Bengal data, only valid for Bay of Bengal confidence
ensemble_models = []
for i in range(7):
    m = MLP(n_inputs=2).to(device)
    m.load_state_dict(torch.load(f"data/ensemble_model_{i}.pt"))
    m.eval()
    ensemble_models.append(m)
ens_norm = np.load("data/normalization_stats_4yr.npz")
ens_X_mean, ens_X_std = ens_norm["X_mean"], ens_norm["X_std"]
ens_y_mean, ens_y_std = ens_norm["y_mean"], ens_norm["y_std"]
ensemble_results = np.load("data/ensemble_results.npz")
max_uncertainty_seen = float(ensemble_results["ensemble_std"].max())


def get_bay_cluster_id(lat, lon):
    lat_idx = np.argmin(np.abs(bay_grid_lats - lat))
    lon_idx = np.argmin(np.abs(bay_grid_lons - lon))
    return int(bay_cluster_map[lat_idx, lon_idx])


def predict_bay(lat, lon, date):
    sst_val = float(bay_sst.sel(latitude=lat, longitude=lon, time=date, method="nearest"))
    ssh_val = float(bay_ssh.sel(latitude=lat, longitude=lon, time=date, method="nearest"))
    cluster_id = get_bay_cluster_id(lat, lon)
    if np.isnan(sst_val) or np.isnan(ssh_val) or cluster_id == -1:
        return None

    cluster_onehot = np.zeros(5)
    cluster_onehot[cluster_id] = 1
    x = np.concatenate([[sst_val, ssh_val], cluster_onehot]).reshape(1, -1).astype(np.float32)
    x_norm = (x - bay_X_mean) / bay_X_std
    x_t = torch.tensor(x_norm, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_norm = bay_model(x_t).cpu().numpy()[0]
    pred_temp = pred_norm * bay_y_std + bay_y_mean
    pred_corrected = np.array([pred_temp[i] - bay_correction[d] for i, d in enumerate(depths)])

    # Real ensemble-based confidence (valid here - ensemble trained on this exact region)
    x2 = np.array([[sst_val, ssh_val]], dtype=np.float32)
    x2_norm = (x2 - ens_X_mean) / ens_X_std
    x2_t = torch.tensor(x2_norm, dtype=torch.float32).to(device)
    ens_preds = []
    with torch.no_grad():
        for m in ensemble_models:
            p = m(x2_t).cpu().numpy()[0] * ens_y_std + ens_y_mean
            ens_preds.append(p)
    ens_std = np.stack(ens_preds, axis=0).std(axis=0)
    confidence_pct = np.clip(100 * (1 - ens_std / max_uncertainty_seen), 0, 100)

    return sst_val, ssh_val, pred_corrected, confidence_pct


# ======================== ARABIAN SEA SETUP ========================
arabian_model = MLP(n_inputs=16).to(device)
arabian_model.load_state_dict(torch.load("data/best_mlp_arabian_eddy.pt"))
arabian_model.eval()

arabian_norm = np.load("data/normalization_stats_arabian_eddy.npz")
ar_X_mean, ar_X_std = arabian_norm["X_mean"], arabian_norm["X_std"]
ar_y_mean, ar_y_std = arabian_norm["y_mean"], arabian_norm["y_std"]

arabian_correction_df = pd.read_csv("data/depth_bias_correction_arabian_eddy.csv")
arabian_correction = dict(zip(arabian_correction_df["depth"], arabian_correction_df["correction_c"]))

ar_sst = xr.open_dataarray("data/processed_sst_full.nc").sel(longitude=slice(None, BAY_LON_MIN))
ar_ssh = xr.open_dataarray("data/processed_ssh_full.nc").sel(longitude=slice(None, BAY_LON_MIN))
ar_curl = xr.open_dataarray("data/processed_curl_arabian.nc")
ar_mld = xr.open_dataarray("data/processed_mld_arabian.nc")
ar_sss = xr.open_dataarray("data/processed_sss_arabian.nc")
ar_vort = xr.open_dataarray("data/processed_vorticity_arabian.nc")
ar_cluster_map = np.load("data/cluster_map_arabian_curl.npy")
ar_grid_lats = ar_sst.latitude.values
ar_grid_lons = ar_sst.longitude.values

# Confidence for Arabian Sea: no valid ensemble exists for this region (it was never
# trained on Arabian Sea data), so we use the REAL, already-validated per-depth
# correlation from this exact model's Argo validation as an honest confidence proxy
arabian_depth_correlation = {
    0: 0.808, 5: 0.948, 10: 0.919, 20: 0.875, 30: 0.822, 50: 0.821, 75: 0.819,
    100: 0.827, 125: 0.844, 150: 0.865, 200: 0.872, 300: 0.886, 500: 0.873,
    700: 0.891, 1000: 0.879,
}


def get_arabian_cluster_id(lat, lon):
    lat_idx = np.argmin(np.abs(ar_grid_lats - lat))
    lon_idx = np.argmin(np.abs(ar_grid_lons - lon))
    return int(ar_cluster_map[lat_idx, lon_idx])


def predict_arabian(lat, lon, date):
    try:
        sst_val = float(ar_sst.sel(latitude=lat, longitude=lon, time=date, method="nearest"))
        ssh_val = float(ar_ssh.sel(latitude=lat, longitude=lon, time=date, method="nearest"))
        curl_val = float(ar_curl.sel(latitude=lat, longitude=lon, time=date, method="nearest"))
        mld_val = float(ar_mld.sel(latitude=lat, longitude=lon, time=date, method="nearest"))
        sss_val = float(ar_sss.sel(latitude=lat, longitude=lon, time=date, method="nearest"))
        vort_val = float(ar_vort.sel(latitude=lat, longitude=lon, time=date, method="nearest"))
    except Exception:
        return None

    if any(np.isnan(v) for v in [sst_val, ssh_val, curl_val, mld_val, sss_val, vort_val]):
        return None

    cluster_id = get_arabian_cluster_id(lat, lon)
    if cluster_id == -1:
        return None

    cluster_onehot = np.zeros(10)
    cluster_onehot[cluster_id] = 1
    x = np.concatenate([[sst_val, ssh_val, curl_val, mld_val, sss_val, vort_val], cluster_onehot]).reshape(1, -1).astype(np.float32)
    x_norm = (x - ar_X_mean) / ar_X_std
    x_t = torch.tensor(x_norm, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_norm = arabian_model(x_t).cpu().numpy()[0]
    pred_temp = pred_norm * ar_y_std + ar_y_mean
    pred_corrected = np.array([pred_temp[i] - arabian_correction[d] for i, d in enumerate(depths)])

    confidence_pct = np.array([arabian_depth_correlation[d] * 100 for d in depths])

    return sst_val, ssh_val, curl_val, mld_val, sss_val, vort_val, pred_corrected, confidence_pct


# ======================== ARGO MATCHING (shared, both regions) ========================
def get_argo_match(lat, lon, date, tolerance_days=3, tolerance_deg=0.5):
    profiles_dir = "data/argo_full/profiles"
    best_match = None
    best_dist = None

    for fname in os.listdir(profiles_dir):
        try:
            argo = xr.open_dataset(os.path.join(profiles_dir, fname))
            a_lat = float(argo["LATITUDE"].values[0])
            a_lon = float(argo["LONGITUDE"].values[0])
            a_date = pd.Timestamp(argo["JULD"].values[0])

            day_diff = abs((a_date - pd.Timestamp(date)).days)
            spatial_dist = ((a_lat - lat) ** 2 + (a_lon - lon) ** 2) ** 0.5

            if day_diff <= tolerance_days and spatial_dist <= tolerance_deg:
                if best_dist is None or spatial_dist < best_dist:
                    pres = argo["PRES"].values[0]
                    temp = argo["TEMP"].values[0]
                    valid_p = ~np.isnan(pres) & ~np.isnan(temp)
                    if valid_p.sum() < 2:
                        continue
                    argo_interp = np.interp(depths, pres[valid_p], temp[valid_p])
                    best_match = argo_interp
                    best_dist = spatial_dist
        except Exception:
            continue

    return best_match


# ======================== GENERATE ALL DEMO FILES ========================
os.makedirs("data/demo_json", exist_ok=True)

bay_points = [
    {"lat": 13.14, "lon": 86.75, "date": "2023-01-26", "region": "Bay of Bengal"},
    {"lat": 13.90, "lon": 84.57, "date": "2023-06-16", "region": "Bay of Bengal"},
    {"lat": 12.57, "lon": 90.12, "date": "2023-11-20", "region": "Bay of Bengal"},
]

arabian_points = [
    {"lat": 19.36, "lon": 58.65, "date": "2022-01-08", "region": "Arabian Sea"},
    {"lat": 25.02, "lon": 59.00, "date": "2022-06-24", "region": "Arabian Sea"},
    {"lat": 23.93, "lon": 62.25, "date": "2022-09-24", "region": "Arabian Sea"},
]

file_counter = 1

print("=" * 60)
print("BAY OF BENGAL DEMO POINTS")
print("=" * 60)
for point in bay_points:
    result = predict_bay(point["lat"], point["lon"], point["date"])
    if result is None:
        print(f"Skipping {point} - no valid data")
        continue
    sst_val, ssh_val, pred_temp, confidence_pct = result
    argo_temp = get_argo_match(point["lat"], point["lon"], point["date"])

    if argo_temp is not None:
        rmse = float(np.sqrt(np.mean((pred_temp - argo_temp) ** 2)))
        bias = float(np.mean(pred_temp - argo_temp))
        correlation = float(np.corrcoef(pred_temp, argo_temp)[0, 1])
        argo_list = [round(float(t), 2) for t in argo_temp]
    else:
        rmse, bias, correlation, argo_list = None, None, None, None
        print(f"  No real Argo match nearby for {point['date']}")

    output = {
        "location": {"lat": point["lat"], "lon": point["lon"], "region": point["region"]},
        "date": point["date"],
        "model_used": "bay_of_bengal_champion",
        "surface_state": {"sst_c": round(sst_val, 2), "ssh_m": round(ssh_val, 3)},
        "profile": {
            "depths_m": depths,
            "predicted_temp_c": [round(float(t), 2) for t in pred_temp],
            "argo_temp_c": argo_list,
            "confidence_pct": [round(float(c), 1) for c in confidence_pct],
        },
        "metrics": {
            "rmse_c": round(rmse, 3) if rmse is not None else None,
            "bias_c": round(bias, 3) if bias is not None else None,
            "correlation": round(correlation, 3) if correlation is not None else None,
        },
    }

    filename = f"data/demo_json/demo_{file_counter}.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {filename} ({point['region']}, {point['date']})")
    file_counter += 1

print("\n" + "=" * 60)
print("ARABIAN SEA DEMO POINTS")
print("=" * 60)
for point in arabian_points:
    result = predict_arabian(point["lat"], point["lon"], point["date"])
    if result is None:
        print(f"Skipping {point} - no valid data")
        continue
    sst_val, ssh_val, curl_val, mld_val, sss_val, vort_val, pred_temp, confidence_pct = result
    argo_temp = get_argo_match(point["lat"], point["lon"], point["date"])

    if argo_temp is not None:
        rmse = float(np.sqrt(np.mean((pred_temp - argo_temp) ** 2)))
        bias = float(np.mean(pred_temp - argo_temp))
        correlation = float(np.corrcoef(pred_temp, argo_temp)[0, 1])
        argo_list = [round(float(t), 2) for t in argo_temp]
    else:
        rmse, bias, correlation, argo_list = None, None, None, None
        print(f"  No real Argo match nearby for {point['date']}")

    output = {
        "location": {"lat": point["lat"], "lon": point["lon"], "region": point["region"]},
        "date": point["date"],
        "model_used": "arabian_sea_champion",
        "surface_state": {
            "sst_c": round(sst_val, 2), "ssh_m": round(ssh_val, 3),
            "wind_stress_curl": curl_val, "mld_m": round(mld_val, 1),
            "sss_psu": round(sss_val, 2), "eddy_vorticity": vort_val,
        },
        "profile": {
            "depths_m": depths,
            "predicted_temp_c": [round(float(t), 2) for t in pred_temp],
            "argo_temp_c": argo_list,
            "confidence_pct": [round(float(c), 1) for c in confidence_pct],
        },
        "metrics": {
            "rmse_c": round(rmse, 3) if rmse is not None else None,
            "bias_c": round(bias, 3) if bias is not None else None,
            "correlation": round(correlation, 3) if correlation is not None else None,
        },
    }

    filename = f"data/demo_json/demo_{file_counter}.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {filename} ({point['region']}, {point['date']})")
    file_counter += 1

print("\nAll demo JSON files generated in data/demo_json/")