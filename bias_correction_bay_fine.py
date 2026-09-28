import pandas as pd
import numpy as np

depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

df = pd.read_csv("data/argo_validation_results_bay_fine.csv")

df["profile_id"] = df["lat"].astype(str) + "_" + df["lon"].astype(str) + "_" + df["date"].astype(str)
profile_ids = df["profile_id"].unique()

np.random.seed(42)
np.random.shuffle(profile_ids)
split = int(len(profile_ids) * 0.5)
calib_ids = set(profile_ids[:split])
report_ids = set(profile_ids[split:])

calib_df = df[df["profile_id"].isin(calib_ids)]
report_df = df[df["profile_id"].isin(report_ids)]

print(f"Calibration profiles: {len(calib_ids)} ({len(calib_df)} points)")
print(f"Report profiles:      {len(report_ids)} ({len(report_df)} points)")

correction = {}
for d in depths:
    subset = calib_df[calib_df["depth_bin"] == d]
    if len(subset) >= 5:
        correction[d] = float((subset["model_temp"] - subset["argo_temp"]).mean())
    else:
        correction[d] = 0.0

print("\nPer-depth correction learned (10-cluster Bay of Bengal model):")
for d in depths:
    print(f"  {d:>5}m: subtract {correction[d]:+.3f} C from future predictions")

report_df = report_df.copy()
report_df["model_temp_corrected"] = report_df.apply(
    lambda row: row["model_temp"] - correction[row["depth_bin"]], axis=1
)

print(f"\n{'Depth (m)':>10} {'N':>7} {'RMSE before':>12} {'RMSE after':>11} {'Corr before':>12} {'Corr after':>11}")
for d in depths:
    subset = report_df[report_df["depth_bin"] == d]
    if len(subset) < 5:
        continue
    err_before = subset["model_temp"] - subset["argo_temp"]
    err_after = subset["model_temp_corrected"] - subset["argo_temp"]
    rmse_before = np.sqrt(np.mean(err_before ** 2))
    rmse_after = np.sqrt(np.mean(err_after ** 2))
    corr_before = np.corrcoef(subset["argo_temp"], subset["model_temp"])[0, 1]
    corr_after = np.corrcoef(subset["argo_temp"], subset["model_temp_corrected"])[0, 1]
    print(f"{d:>10} {len(subset):>7} {rmse_before:>12.3f} {rmse_after:>11.3f} {corr_before:>12.3f} {corr_after:>11.3f}")

err_before = report_df["model_temp"] - report_df["argo_temp"]
err_after = report_df["model_temp_corrected"] - report_df["argo_temp"]
print(f"\n{'OVERALL':>10} {len(report_df):>7} "
      f"{np.sqrt(np.mean(err_before**2)):>12.3f} {np.sqrt(np.mean(err_after**2)):>11.3f} "
      f"{np.corrcoef(report_df['argo_temp'], report_df['model_temp'])[0,1]:>12.3f} "
      f"{np.corrcoef(report_df['argo_temp'], report_df['model_temp_corrected'])[0,1]:>11.3f}")

correction_df = pd.DataFrame(list(correction.items()), columns=["depth", "correction_c"])
correction_df.to_csv("data/depth_bias_correction_bay_fine.csv", index=False)
print("\nSaved correction table to data/depth_bias_correction_bay_fine.csv")