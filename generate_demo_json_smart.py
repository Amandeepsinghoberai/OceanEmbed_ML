import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import json
import os
import gsw

depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BAY_LON_MIN = 77.0  # the one boundary that decides which model to use


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


# ======================== LOAD BOTH MODELS ONCE ========================
print("Loading Bay of Bengal model...")
bay_model = MLP(n_inputs=7).to(device)
bay_model.load_state_dict(torch.load("data/best_mlp_cluster.pt"))
bay_model.eval()
bay_norm = np.load("data/normalization_stats_cluster.npz")
bay_correction = dict(zip(
    pd.read_csv("data/depth_bias_correction_cluster.csv")["depth"],
    pd.read_csv("data/depth_bias_correction_cluster.csv")["correction_c"]
))
bay_sst = xr.open_dataarray("data/processed_sst_4yr.nc")
bay_ssh = xr.open_dataarray("data/processed_ssh_4yr.nc")
bay_cluster_map = np.load("data/cluster_map.npy")

print("Loading Arabian Sea model...")
arabian_model = MLP(n_inputs=16).to(device)
arabian_model.load_state_dict(torch.load("data/best_mlp_arabian_eddy.pt"))
arabian_model.eval()
arabian_norm = np.load("data/normalization_stats_arabian_eddy.npz")
arabian_correction = dict(zip(
    pd.read_csv("data/depth_bias_correction_arabian_eddy.csv")["depth"],
    pd.read_csv("data/depth_bias_correction_arabian_eddy.csv")["correction_c"]
))
ar_sst = xr.open_dataarray("data/processed_sst_full.nc").sel(longitude=slice(None, BAY_LON_MIN))
ar_ssh = xr.open_dataarray("data/processed_ssh_full.nc").sel(longitude=slice(None, BAY_LON_MIN))
ar_curl = xr.open_dataarray("data/processed_curl_arabian.nc")
ar_mld = xr.open_dataarray("data/processed_mld_arabian.nc")
ar_sss = xr.open_dataarray("data/processed_sss_arabian.nc")
ar_vort = xr.open_dataarray("data/processed_vorticity_arabian.nc")
ar_cluster_map = np.load("data/cluster_map_arabian_curl.npy")

arabian_depth_correlation = {
    0: 0.808, 5: 0.948, 10: 0.919, 20: 0.875, 30: 0.822, 50: 0.821, 75: 0.819,
    100: 0.827, 125: 0.844, 150: 0.865, 200: 0.872, 300: 0.886, 500: 0.873,
    700: 0.891, 1000: 0.879,
}

print("Loading confidence ensemble (Bay of Bengal only)...")
ensemble_models = []
for i in range(7):
    m = MLP(n_inputs=2).to(device)
    m.load_state_dict(torch.load(f"data/ensemble_model_{i}.pt"))
    m.eval()
    ensemble_models.append(m)
ens_norm = np.load("data/normalization_stats_4yr.npz")
max_uncertainty_seen = float(np.load("data/ensemble_results.npz")["ensemble_std"].max())

print("All models loaded. Ready to predict for any point in the region.\n")


def get_cluster_id(lat, lon, cluster_map, grid_lats, grid_lons):
    lat_idx = np.argmin(np.abs(grid_lats - lat))
    lon_idx = np.argmin(np.abs(grid_lons - lon))
    return int(cluster_map[lat_idx, lon_idx])


def compute_pressure_profile(lat, depths):
    """
    Real pressure at each depth, using the standard oceanographic
    formula (adjusted for latitude, since Earth's shape means gravity
    varies slightly by latitude). This is a direct calculation from
    physics, not a model prediction - depth alone determines pressure
    almost completely.
    """
    z = [-d for d in depths]  # gsw expects negative depth (height convention)
    pressure_dbar = gsw.p_from_z(z, lat)
    return [round(float(p), 1) for p in pressure_dbar]


def predict(lat, lon, date):
    """
    THE SMART PART: automatically picks the right model based on which
    side of 77 degrees longitude the point falls on. Everything downstream
    (the caller) doesn't need to know or care which model was used.
    """
    if lon < BAY_LON_MIN:
        region = "Arabian Sea"
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

        cluster_id = get_cluster_id(lat, lon, ar_cluster_map, ar_sst.latitude.values, ar_sst.longitude.values)
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
        pred_corrected = np.array([pred_temp[i] - arabian_correction[d] for i, d in enumerate(depths)])
        confidence_pct = np.array([arabian_depth_correlation[d] * 100 for d in depths])

        surface_state = {
            "sst_c": round(sst_val, 2), "ssh_m": round(ssh_val, 3),
            "wind_stress_curl": curl_val, "mld_m": round(mld_val, 1),
            "sss_psu": round(sss_val, 2), "eddy_vorticity": vort_val,
        }

    else:
        region = "Bay of Bengal"
        try:
            sst_val = float(bay_sst.sel(latitude=lat, longitude=lon, time=date, method="nearest"))
            ssh_val = float(bay_ssh.sel(latitude=lat, longitude=lon, time=date, method="nearest"))
        except Exception:
            return None

        if np.isnan(sst_val) or np.isnan(ssh_val):
            return None

        cluster_id = get_cluster_id(lat, lon, bay_cluster_map, bay_sst.latitude.values, bay_sst.longitude.values)
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
        pred_corrected = np.array([pred_temp[i] - bay_correction[d] for i, d in enumerate(depths)])

        x2 = np.array([[sst_val, ssh_val]], dtype=np.float32)
        x2_norm = (x2 - ens_norm["X_mean"]) / ens_norm["X_std"]
        x2_t = torch.tensor(x2_norm, dtype=torch.float32).to(device)
        ens_preds = []
        with torch.no_grad():
            for m in ensemble_models:
                p = m(x2_t).cpu().numpy()[0] * ens_norm["y_std"] + ens_norm["y_mean"]
                ens_preds.append(p)
        ens_std = np.stack(ens_preds, axis=0).std(axis=0)
        confidence_pct = np.clip(100 * (1 - ens_std / max_uncertainty_seen), 0, 100)

        surface_state = {"sst_c": round(sst_val, 2), "ssh_m": round(ssh_val, 3)}

    pressure_profile = compute_pressure_profile(lat, depths)

    return {
        "region": region,
        "surface_state": surface_state,
        "predicted_temp_c": pred_corrected,
        "confidence_pct": confidence_pct,
        "pressure_dbar": pressure_profile,
    }


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
                    best_match = np.interp(depths, pres[valid_p], temp[valid_p])
                    best_dist = spatial_dist
        except Exception:
            continue
    return best_match


# ======================== GENERATE DEMO FILES (any mix of points, any region) ========================
demo_points = [
    {"lat": 13.14, "lon": 86.75, "date": "2023-01-26"},
    {"lat": 13.90, "lon": 84.57, "date": "2023-06-16"},
    {"lat": 12.57, "lon": 90.12, "date": "2023-11-20"},
    {"lat": 19.36, "lon": 58.65, "date": "2022-01-08"},
    {"lat": 25.02, "lon": 59.00, "date": "2022-06-24"},
    {"lat": 23.93, "lon": 62.25, "date": "2022-09-24"},
]

os.makedirs("data/demo_json", exist_ok=True)
file_counter = 1

for point in demo_points:
    result = predict(point["lat"], point["lon"], point["date"])
    if result is None:
        print(f"Skipping {point} - no valid data at this point/date")
        continue

    argo_temp = get_argo_match(point["lat"], point["lon"], point["date"])
    pred_temp = result["predicted_temp_c"]

    if argo_temp is not None:
        rmse = float(np.sqrt(np.mean((pred_temp - argo_temp) ** 2)))
        bias = float(np.mean(pred_temp - argo_temp))
        correlation = float(np.corrcoef(pred_temp, argo_temp)[0, 1])
        argo_list = [round(float(t), 2) for t in argo_temp]
    else:
        rmse, bias, correlation, argo_list = None, None, None, None

    output = {
        "location": {"lat": point["lat"], "lon": point["lon"], "region": result["region"]},
        "date": point["date"],
        "surface_state": result["surface_state"],
        "profile": {
            "depths_m": depths,
            "pressure_dbar": result["pressure_dbar"],
            "predicted_temp_c": [round(float(t), 2) for t in pred_temp],
            "argo_temp_c": argo_list,
            "confidence_pct": [round(float(c), 1) for c in result["confidence_pct"]],
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
    print(f"Saved {filename}: {result['region']}, {point['date']} (auto-detected)")
    file_counter += 1

print("\nDone. The 'predict()' function above is exactly what your backend's")
print("/api/predict endpoint should call - it already picks the right model")
print("automatically based on longitude, no separate code paths needed.")