import pandas as pd
import os

index_path = "data/argo/ar_index_global_prof.txt"

print("Loading and filtering the Argo global index for the full region, 2020-2023...")
df = pd.read_csv(index_path, comment="#")
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d%H%M%S", errors="coerce")

mask = (
    (df["latitude"] >= 5) & (df["latitude"] <= 30) &
    (df["longitude"] >= 45) & (df["longitude"] <= 105) &
    (df["date"] >= "2020-01-01") & (df["date"] <= "2023-12-31")
)
matches = df[mask].copy()

print(f"\nFound {len(matches)} Argo profiles in the full region (Bay of Bengal + Arabian Sea), 2020-2023.")

# Quick breakdown: roughly how many fall in each sea, using 77E as a rough dividing line
bay_count = (matches["longitude"] >= 77).sum()
arabian_count = (matches["longitude"] < 77).sum()
print(f"  Roughly in Bay of Bengal side (lon >= 77E): {bay_count}")
print(f"  Roughly in Arabian Sea side (lon < 77E):    {arabian_count}")

print(matches[["file", "date", "latitude", "longitude"]].head())

os.makedirs("data/argo_full", exist_ok=True)
matches.to_csv("data/argo_full/argo_index_full_2020_2023.csv", index=False)
print("\nSaved filtered index to data/argo_full/argo_index_full_2020_2023.csv")