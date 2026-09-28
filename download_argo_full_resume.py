import pandas as pd
import requests
import os
import time

matches = pd.read_csv("data/argo_full/argo_index_full_2020_2023.csv")

os.makedirs("data/argo_full/profiles", exist_ok=True)
base_url = "https://data-argo.ifremer.fr/dac/"

downloaded = 0
failed = 0
already_had = 0
failed_files = []

for i, row in matches.iterrows():
    file_path = row["file"]
    filename = file_path.split("/")[-1]
    local_path = f"data/argo_full/profiles/{filename}"

    if os.path.exists(local_path):
        already_had += 1
        downloaded += 1
        continue

    url = base_url + file_path
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
        downloaded += 1
        # Small pause between actual downloads (not skipped/cached files) -
        # this is what should keep us under the server's rate limit this time
        time.sleep(0.1)
    except Exception as e:
        failed += 1
        failed_files.append(file_path)

    total_done = downloaded + failed
    if total_done % 200 == 0:
        print(f"  {total_done}/{len(matches)} processed... ({downloaded} ok, {failed} failed)")

print(f"\nDone. Downloaded: {downloaded} (of which {already_had} were already present), Failed: {failed}")
if failed_files:
    print(f"\nFirst few failed files (for reference): {failed_files[:5]}")
print(f"Files saved to data/argo_full/profiles/")