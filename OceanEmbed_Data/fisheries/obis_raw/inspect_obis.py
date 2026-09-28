import json
import glob
from collections import Counter
from pathlib import Path

files = sorted(glob.glob("obis_fish_*.json"))

total = 0
species = Counter()
years = Counter()
datasets = Counter()
taxon_ranks = Counter()
status = Counter()

depth_count = 0
sst_count = 0
sss_count = 0
coordinates_count = 0
species_id_count = 0

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        records = json.load(f)

    for r in records:
        total += 1

        name = r.get("scientificName") or r.get("originalScientificName")
        if name:
            species[name] += 1

        year = r.get("date_year") or r.get("year")
        if year:
            years[str(year)] += 1

        dataset = r.get("datasetName") or r.get("dataset_id")
        if dataset:
            datasets[dataset] += 1

        rank = r.get("taxonRank")
        if rank:
            taxon_ranks[rank] += 1

        occurrence = r.get("occurrenceStatus")
        if occurrence:
            status[str(occurrence)] += 1

        if r.get("depth") is not None or (
            r.get("minimumDepthInMeters") is not None
            and r.get("maximumDepthInMeters") is not None
        ):
            depth_count += 1

        if r.get("sst") is not None:
            sst_count += 1

        if r.get("sss") is not None:
            sss_count += 1

        if (
            r.get("decimalLatitude") is not None
            and r.get("decimalLongitude") is not None
        ):
            coordinates_count += 1

        if r.get("speciesid") is not None or r.get("scientificNameID") is not None:
            species_id_count += 1


print("\n========== OBIS FISH DATASET REPORT ==========\n")

print(f"Files: {len(files)}")
print(f"Total records: {total}")
print(f"Records with coordinates: {coordinates_count}")
print(f"Records with depth information: {depth_count}")
print(f"Records with SST: {sst_count}")
print(f"Records with SSS: {sss_count}")
print(f"Records with species/taxon ID: {species_id_count}")

print("\n--- Taxonomic ranks ---")
for k, v in taxon_ranks.most_common():
    print(f"{k}: {v}")

print("\n--- Years (top 20) ---")
for k, v in years.most_common(20):
    print(f"{k}: {v}")

print("\n--- Occurrence status ---")
for k, v in status.most_common():
    print(f"{k}: {v}")

print("\n--- Top 30 species/taxa ---")
for k, v in species.most_common(30):
    print(f"{k}: {v}")

print("\n--- Top 20 datasets ---")
for k, v in datasets.most_common(20):
    print(f"{k}: {v}")

print("\n==============================================\n")