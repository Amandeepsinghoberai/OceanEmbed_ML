import pandas as pd
import xarray as xr
import numpy as np

df = pd.read_csv("data/ar_index_global_prof_latest.txt", comment="#")
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d%H%M%S", errors="coerce")

# Find the exact same profile that returned the bad value
mask = (df["latitude"].round(2) == 14.35) & (df["longitude"].round(2) == 88.08)
match = df[mask]
print(match[["file", "date"]])

file_path = f"data/recent_argo_cache/{match.iloc[0]['file'].split('/')[-1]}"
argo = xr.open_dataset(file_path)

pres = argo["PRES"].values[0]
temp = argo["TEMP"].values[0]
temp_qc = argo["TEMP_QC"].values[0]
qc_str = [q.decode() if isinstance(q, bytes) else str(q) for q in temp_qc]

print("\nFirst 15 real readings in this profile (pressure, temp, QC flag):")
for p, t, q in list(zip(pres, temp, qc_str))[:15]:
    print(f"  Pressure: {p}, Temp: {t}, QC: {q}")