import pandas as pd
import requests
import os

# Set this to True to download all 15,283 profiles (several hours),
# or False to download a random representative sample instead (much faster)
DOWNLOAD_ALL = True
SAMPLE_SIZE = 5000  # only used if DOWNLOAD_ALL is False

matches = pd.read_csv("data/argo_full/argo_index_full_2020_2023.csv")

if not DOWNLOAD_ALL:
    matches = matches.sample(n=min(SAMPLE_SIZE, len(matches)), random_state=42)
    print(f"Downloading a random sample of {len(matches)} profiles...")
else:
    print(f"Downloading all {len(matches)} profiles...")

os.makedirs("data/argo_full/profiles", exist_ok=True)
base_url = "https://data-argo.ifremer.fr/dac/"

downloaded = 0
failed = 0
already_had = 0

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
    except Exception as e:
        failed += 1

    total_done = downloaded + failed
    if total_done % 200 == 0:
        print(f"  {total_done}/{len(matches)} processed... ({downloaded} ok, {failed} failed)")

print(f"\nDone. Downloaded: {downloaded} (of which {already_had} were already present), Failed: {failed}")
print(f"Files saved to data/argo_full/profiles/")