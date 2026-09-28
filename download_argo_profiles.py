import pandas as pd
import requests
import os

os.makedirs("data/argo/profiles", exist_ok=True)

matches = pd.read_csv("data/argo/argo_index_bob_2023.csv")
base_url = "https://data-argo.ifremer.fr/dac/"

print(f"Downloading {len(matches)} Argo profile files...")

downloaded = 0
failed = 0

for i, row in matches.iterrows():
    file_path = row["file"]
    filename = file_path.split("/")[-1]
    local_path = f"data/argo/profiles/{filename}"

    if os.path.exists(local_path):
        downloaded += 1
        continue

    url = base_url + file_path
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
        downloaded += 1
    except Exception as e:
        print(f"Failed: {file_path} -> {e}")
        failed += 1

    if downloaded % 50 == 0 and downloaded > 0:
        print(f"  {downloaded}/{len(matches)} downloaded...")

print(f"\nDone. Downloaded: {downloaded}, Failed: {failed}")
print(f"Files saved to data/argo/profiles/")