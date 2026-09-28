import pandas as pd
import xarray as xr

df = pd.read_csv("data/ar_index_global_prof_latest.txt", comment="#")
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d%H%M%S", errors="coerce")

# Try files from 5-10 days ago instead of the most recent 1-2 days
mask = (df["date"] >= pd.Timestamp.now() - pd.Timedelta(days=10)) & (df["date"] <= pd.Timestamp.now() - pd.Timedelta(days=5))
candidates = df[mask]
print(f"Testing {min(5, len(candidates))} files from 5-10 days ago...")

for _, row in candidates.head(5).iterrows():
    url = f"https://data-argo.ifremer.fr/dac/{row['file']}"
    try:
        ds = xr.open_dataset(url)
        print(f"SUCCESS: {row['file']} (from {row['date'].date()})")
    except Exception as e:
        print(f"FAILED: {row['file']} - {type(e).__name__}")