import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import os
import pandas as pd

depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 48), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(48, 48), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(48, 15)
        )

    def forward(self, x):
        return self.net(x)


model = MLP().to(device)
model.load_state_dict(torch.load("data/best_mlp_4yr.pt"))
model.eval()

norm = np.load("data/normalization_stats_4yr.npz")
X_mean, X_std = norm["X_mean"], norm["X_std"]
y_mean, y_std = norm["y_mean"], norm["y_std"]

# Use the 4-year surface data - covers 2020-2023, includes all our Argo dates (2023)
sst = xr.open_dataarray("data/processed_sst_4yr.nc")
ssh = xr.open_dataarray("data/processed_ssh_4yr.nc")


def predict_profile(lat, lon, date):
    try:
        sst_val = float(sst.sel(latitude=lat, longitude=lon, time=date, method="nearest"))
        ssh_val = float(ssh.sel(latitude=lat, longitude=lon, time=date, method="nearest"))
    except Exception:
        return None

    if np.isnan(sst_val) or np.isnan(ssh_val):
        return None

    x = np.array([[sst_val, ssh_val]], dtype=np.float32)
    x_norm = (x - X_mean) / X_std
    x_t = torch.tensor(x_norm, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_norm = model(x_t).cpu().numpy()[0]
    return pred_norm * y_std + y_mean


profiles_dir = "data/argo/profiles"
files = os.listdir(profiles_dir)
print(f"Validating 4-year model against {len(files)} real Argo profiles...\n")

results = []
matched_profiles = 0
skipped_profiles = 0

for fname in files:
    path = os.path.join(profiles_dir, fname)
    try:
        argo = xr.open_dataset(path)

        lat = float(argo["LATITUDE"].values[0])
        lon = float(argo["LONGITUDE"].values[0])
        date = pd.Timestamp(argo["JULD"].values[0])

        pres = argo["PRES"].values[0]
        temp = argo["TEMP"].values[0]
        temp_qc = argo["TEMP_QC"].values[0]

        pred = predict_profile(lat, lon, date)
        if pred is None:
            skipped_profiles += 1
            continue

        matched_this_profile = False
        for p, t, qc in zip(pres, temp, temp_qc):
            if np.isnan(p) or np.isnan(t):
                continue
            qc_str = qc.decode() if isinstance(qc, bytes) else str(qc)
            if qc_str not in ("1", "2"):
                continue
            if p < 0 or p > 1000:
                continue

            model_temp = float(np.interp(p, depths, pred))
            nearest_bin = min(depths, key=lambda d: abs(d - p))

            results.append({
                "depth_bin": nearest_bin,
                "argo_pres": p,
                "argo_temp": t,
                "model_temp": model_temp,
                "lat": lat, "lon": lon, "date": date
            })
            matched_this_profile = True

        if matched_this_profile:
            matched_profiles += 1
        else:
            skipped_profiles += 1

    except Exception:
        skipped_profiles += 1
        continue

print(f"Matched profiles: {matched_profiles}")
print(f"Skipped profiles: {skipped_profiles}")
print(f"Total real depth measurements compared: {len(results)}")

df = pd.DataFrame(results)
df.to_csv("data/argo_validation_results_4yr.csv", index=False)
print("\nSaved full comparison table to data/argo_validation_results_4yr.csv")

print(f"\n{'Depth (m)':>10} {'N':>6} {'RMSE (C)':>10} {'Bias (C)':>10} {'Correlation':>12}")
for d in depths:
    subset = df[df["depth_bin"] == d]
    if len(subset) < 3:
        print(f"{d:>10} {len(subset):>6}   (too few real points to score)")
        continue
    err = subset["model_temp"] - subset["argo_temp"]
    rmse = np.sqrt(np.mean(err ** 2))
    bias = np.mean(err)
    corr = np.corrcoef(subset["argo_temp"], subset["model_temp"])[0, 1]
    print(f"{d:>10} {len(subset):>6} {rmse:>10.3f} {bias:>10.3f} {corr:>12.3f}")

overall_err = df["model_temp"] - df["argo_temp"]
overall_rmse = np.sqrt(np.mean(overall_err ** 2))
overall_bias = np.mean(overall_err)
overall_corr = np.corrcoef(df["argo_temp"], df["model_temp"])[0, 1]
print(f"\n{'OVERALL':>10} {len(df):>6} {overall_rmse:>10.3f} {overall_bias:>10.3f} {overall_corr:>12.3f}")

print("\n" + "=" * 60)
print("COMPARISON TO PREVIOUS (1-year model) ARGO VALIDATION")
print("=" * 60)
print("1-year model: 567 profiles matched, 65,944 points, OVERALL RMSE 0.759")
print(f"4-year model: {matched_profiles} profiles matched, {len(df)} points, OVERALL RMSE {overall_rmse:.3f}")