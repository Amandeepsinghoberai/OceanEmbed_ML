import pandas as pd
import requests

print("Fetching the current Argo global index...")
url = "https://data-argo.ifremer.fr/ar_index_global_prof.txt"
r = requests.get(url, timeout=60)
with open("data/ar_index_global_prof_latest.txt", "wb") as f:
    f.write(r.content)

df = pd.read_csv("data/ar_index_global_prof_latest.txt", comment="#")
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d%H%M%S", errors="coerce")

# Check what's actually available recently, in our region
mask = (
    (df["latitude"] >= 5) & (df["latitude"] <= 30) &
    (df["longitude"] >= 45) & (df["longitude"] <= 100) &
    (df["date"] >= pd.Timestamp.now() - pd.Timedelta(days=30))
)
recent = df[mask]
print(f"\nArgo profiles in our region in the last 30 days: {len(recent)}")
print(f"\nMost recent 10:")
print(recent.sort_values("date", ascending=False)[["latitude", "longitude", "date"]].head(10))