import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import json
import os

depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLUSTERS = 5


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


# ---------- Load the final production model ----------
model = MLP(n_inputs=7).to(device)
model.load_state_dict(torch.load("data/best_mlp_cluster.pt"))
model.eval()

norm = np.load("data/normalization_stats_cluster.npz")
X_mean, X_std = norm["X_mean"], norm["X_std"]
y_mean, y_std = norm["y_mean"], norm["y_std"]

# ---------- Load the bias correction table ----------
correction_df = pd.read_csv("data/depth_bias_correction_cluster.csv")
correction = dict(zip(correction_df["depth"], correction_df["correction_c"]))

# ---------- Load the 7-model ensemble for confidence estimates ----------
ensemble_models = []
for i in range(7):
    m = MLP(n_inputs=2).to(device)
    m.load_state_dict(torch.load(f"data/ensemble_model_{i}.pt"))
    m.eval()
    ensemble_models.append(m)

ensemble_norm = np.load("data/normalization_stats_4yr.npz")
ens_X_mean, ens_X_std = ensemble_norm["X_mean"], ensemble_norm["X_std"]
ens_y_mean, ens_y_std = ensemble_norm["y_mean"], ensemble_norm["y_std"]

# FIX: use the actual highest disagreement ever recorded across all real test
# points and depths, not the average - the average was too small a ceiling,
# causing roughly half of all points to clip to a flat 0% confidence
ensemble_results = np.load("data/ensemble_results.npz")
max_uncertainty_seen = float(ensemble_results["ensemble_std"].max())
print(f"Using real max ensemble disagreement as confidence ceiling: {max_uncertainty_seen:.4f}")

# ---------- Surface data + cluster map ----------
sst = xr.open_dataarray("data/processed_sst_4yr.nc")
ssh = xr.open_dataarray("data/processed_ssh_4yr.nc")
cluster_map = np.load("data/cluster_map.npy")
grid_lats = sst.latitude.values
grid_lons = sst.longitude.values


def get_cluster_id(lat, lon):
    lat_idx = np.argmin(np.abs(grid_lats - lat))
    lon_idx = np.argmin(np.abs(grid_lons - lon))
    return int(cluster_map[lat_idx, lon_idx])


def predict_final(lat, lon, date):
    sst_val = float(sst.sel(latitude=lat, longitude=lon, time=date, method="nearest"))
    ssh_val = float(ssh.sel(latitude=lat, longitude=lon, time=date, method="nearest"))
    cluster_id = get_cluster_id(lat, lon)

    if np.isnan(sst_val) or np.isnan(ssh_val) or cluster_id == -1:
        return None

    cluster_onehot = np.zeros(N_CLUSTERS)
    cluster_onehot[cluster_id] = 1
    x = np.concatenate([[sst_val, ssh_val], cluster_onehot]).reshape(1, -1).astype(np.float32)
    x_norm = (x - X_mean) / X_std
    x_t = torch.tensor(x_norm, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_norm = model(x_t).cpu().numpy()[0]
    pred_temp = pred_norm * y_std + y_mean
    pred_temp_corrected = np.array([pred_temp[i] - correction[d] for i, d in enumerate(depths)])

    x2 = np.array([[sst_val, ssh_val]], dtype=np.float32)
    x2_norm = (x2 - ens_X_mean) / ens_X_std
    x2_t = torch.tensor(x2_norm, dtype=torch.float32).to(device)
    ens_preds = []
    with torch.no_grad():
        for m in ensemble_models:
            p = m(x2_t).cpu().numpy()[0] * ens_y_std + ens_y_mean
            ens_preds.append(p)
    ens_preds = np.stack(ens_preds, axis=0)
    ens_std = ens_preds.std(axis=0)
    confidence_pct = np.clip(100 * (1 - ens_std / max_uncertainty_seen), 0, 100)

    return sst_val, ssh_val, pred_temp_corrected, confidence_pct


def get_argo_match(lat, lon, date, tolerance_days=3, tolerance_deg=0.5):
    profiles_dir = "data/argo/profiles"
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
                    argo_interp = np.interp(depths, pres[valid_p], temp[valid_p])
                    best_match = argo_interp
                    best_dist = spatial_dist
        except Exception:
            continue

    return best_match


# ---------- Real demo points, one per season, all with confirmed real Argo profiles ----------
demo_points = [
    {"lat": 13.14, "lon": 86.75, "date": "2023-01-26", "region": "Bay of Bengal"},
    {"lat": 13.90, "lon": 84.57, "date": "2023-06-16", "region": "Bay of Bengal"},
    {"lat": 12.57, "lon": 90.12, "date": "2023-11-20", "region": "Bay of Bengal"},
]

os.makedirs("data/demo_json", exist_ok=True)

for i, point in enumerate(demo_points):
    result = predict_final(point["lat"], point["lon"], point["date"])
    if result is None:
        print(f"Point {i+1} ({point['lat']}, {point['lon']}, {point['date']}): no valid data, skipping")
        continue

    sst_val, ssh_val, pred_temp, confidence_pct = result
    argo_temp = get_argo_match(point["lat"], point["lon"], point["date"])

    if argo_temp is not None:
        rmse = float(np.sqrt(np.mean((pred_temp - argo_temp) ** 2)))
        bias = float(np.mean(pred_temp - argo_temp))
        correlation = float(np.corrcoef(pred_temp, argo_temp)[0, 1])
        argo_list = [round(float(t), 2) for t in argo_temp]
    else:
        rmse, bias, correlation = None, None, None
        argo_list = None
        print(f"Point {i+1}: no real Argo match nearby - using model prediction only")

    output = {
        "location": {"lat": point["lat"], "lon": point["lon"], "region": point["region"]},
        "date": point["date"],
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

    filename = f"data/demo_json/demo_{i+1}.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {filename}:")
    print(json.dumps(output, indent=2))

print("\nAll demo JSON files generated in data/demo_json/")