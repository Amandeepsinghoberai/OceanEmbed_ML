import pandas as pd
import numpy as np

depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

df = pd.read_csv("data/argo_validation_results.csv")
df["profile_id"] = df["lat"].astype(str) + "_" + df["lon"].astype(str) + "_" + df["date"].astype(str)
profile_ids = df["profile_id"].unique()

np.random.seed(42)
np.random.shuffle(np.array(profile_ids, dtype=object))
split = int(len(profile_ids) * 0.5)
calib_ids = set(profile_ids[:split])
report_ids = set(profile_ids[split:])

calib_df = df[df["profile_id"].isin(calib_ids)]
report_df = df[df["profile_id"].isin(report_ids)].copy()

# Learn a slope + intercept per depth (not just a fixed shift) from HALF the data
calibration = {}
for d in depths:
    subset = calib_df[calib_df["depth_bin"] == d]
    if len(subset) >= 20:
        slope, intercept = np.polyfit(subset["model_temp"], subset["argo_temp"], 1)
        calibration[d] = (slope, intercept)
    else:
        calibration[d] = (1.0, 0.0)

print("Per-depth calibration learned (slope, intercept):")
for d in depths:
    s, i = calibration[d]
    print(f"  {d:>5}m: slope={s:.3f}, intercept={i:+.3f}")

def calibrate(row):
    s, i = calibration[row["depth_bin"]]
    return row["model_temp"] * s + i

report_df["model_temp_calibrated"] = report_df.apply(calibrate, axis=1)

print(f"\n{'Depth (m)':>10} {'N':>6} {'RMSE raw':>10} {'RMSE calib':>11} {'Corr raw':>10} {'Corr calib':>11}")
for d in depths:
    subset = report_df[report_df["depth_bin"] == d]
    if len(subset) < 20:
        continue
    err_raw = subset["model_temp"] - subset["argo_temp"]
    err_cal = subset["model_temp_calibrated"] - subset["argo_temp"]
    print(f"{d:>10} {len(subset):>6} "
          f"{np.sqrt(np.mean(err_raw**2)):>10.3f} {np.sqrt(np.mean(err_cal**2)):>11.3f} "
          f"{np.corrcoef(subset['argo_temp'], subset['model_temp'])[0,1]:>10.3f} "
          f"{np.corrcoef(subset['argo_temp'], subset['model_temp_calibrated'])[0,1]:>11.3f}")