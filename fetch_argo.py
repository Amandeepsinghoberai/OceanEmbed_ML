import requests
import pandas as pd
import os

os.makedirs("data/argo", exist_ok=True)

# Step 1: Download the official Argo global profile index
# This is a small text file listing every profile ever recorded - not the actual data yet
index_url = "https://data-argo.ifremer.fr/ar_index_global_prof.txt"
index_path = "data/argo/ar_index_global_prof.txt"

if not os.path.exists(index_path):
    print("Downloading Argo global profile index...")
    r = requests.get(index_url, stream=True)
    r.raise_for_status()
    with open(index_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Index downloaded.")
else:
    print("Index already downloaded, reusing it.")

# Step 2: Load and filter the index for our region and dates
print("Loading and filtering index...")
df = pd.read_csv(index_path, comment="#")
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d%H%M%S", errors="coerce")

mask = (
    (df["latitude"] >= 5) & (df["latitude"] <= 22) &
    (df["longitude"] >= 80) & (df["longitude"] <= 100) &
    (df["date"] >= "2023-01-01") & (df["date"] <= "2023-12-31")
)
matches = df[mask].copy()

print(f"\nFound {len(matches)} Argo profiles in Bay of Bengal, 2023.")
print(matches[["file", "date", "latitude", "longitude"]].head())

matches.to_csv("data/argo/argo_index_bob_2023.csv", index=False)
print("\nSaved filtered index to data/argo/argo_index_bob_2023.csv")
print("Run the download step next once this count looks reasonable.")