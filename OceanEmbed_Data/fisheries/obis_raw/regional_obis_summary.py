import json
from pathlib import Path
from collections import Counter


RAW_DIR = Path(".")

# Your project domain
LAT_MIN = 5.0
LAT_MAX = 30.0
LON_MIN = 45.0
LON_MAX = 105.0


def get_lat(record):
    value = record.get("decimalLatitude")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_lon(record):
    value = record.get("decimalLongitude")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_species_level(record):
    rank = str(record.get("taxonRank") or "").strip().lower()
    return rank in {"species", "subspecies"}


def get_species_name(record):
    name = record.get("scientificName")
    if name:
        return str(name).strip()

    name = record.get("acceptedScientificName")
    if name:
        return str(name).strip()

    return None


# ---------------------------------------------------------
# Load all corrected OBIS batches
# ---------------------------------------------------------

files = sorted(RAW_DIR.glob("obis_fish_*.json"))

print(f"Files found: {len(files)}")

records = []

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        batch = json.load(f)

    records.extend(batch)

print(f"Total records loaded: {len(records)}")


# ---------------------------------------------------------
# Regional classification
#
# Arabian Sea:
#   5-30 N, 45-77 E
#
# Bay of Bengal:
#   5-30 N, 77-105 E
#
# NOTE:
# The 77 E boundary follows the current OceanEmbed
# regional split used by the live inference system.
# ---------------------------------------------------------

arabian = []
bay = []
outside = []

for record in records:

    lat = get_lat(record)
    lon = get_lon(record)

    if lat is None or lon is None:
        outside.append(record)
        continue

    if not (LAT_MIN <= lat <= LAT_MAX and
            LON_MIN <= lon <= LON_MAX):
        outside.append(record)
        continue

    if lon < 77:
        arabian.append(record)
    else:
        bay.append(record)


# ---------------------------------------------------------
# Summary function
# ---------------------------------------------------------

def summarize(name, data):

    species_records = [
        r for r in data
        if is_species_level(r)
    ]

    depth_records = [
        r for r in data
        if r.get("minimumDepthInMeters") is not None
        or r.get("depthInMeters") is not None
        or r.get("maximumDepthInMeters") is not None
    ]

    sst_records = [
        r for r in data
        if r.get("seaSurfaceTemperature") is not None
    ]

    sss_records = [
        r for r in data
        if r.get("seaWaterSalinity") is not None
    ]

    species_counter = Counter()

    for record in species_records:
        species = get_species_name(record)

        if species:
            species_counter[species] += 1

    years = Counter()

    for record in data:
        date = (
            record.get("eventDate")
            or record.get("dateIdentified")
            or record.get("year")
        )

        if date:
            year = str(date)[:4]

            if year.isdigit():
                years[year] += 1

    datasets = Counter()

    for record in data:
        dataset = (
            record.get("datasetName")
            or record.get("datasetID")
            or "Unknown"
        )

        datasets[str(dataset)] += 1

    print("\n========================================")
    print(name)
    print("========================================")

    print(f"Total records: {len(data)}")
    print(f"Species/subspecies records: {len(species_records)}")
    print(f"Records with depth: {len(depth_records)}")
    print(f"Records with SST: {len(sst_records)}")
    print(f"Records with SSS: {len(sss_records)}")

    print("\nTop 20 species:")
    for species, count in species_counter.most_common(20):
        print(f"{species}: {count}")

    print("\nTop 10 years:")
    for year, count in years.most_common(10):
        print(f"{year}: {count}")

    print("\nTop 10 datasets:")
    for dataset, count in datasets.most_common(10):
        print(f"{dataset}: {count}")


# ---------------------------------------------------------
# Print regional results
# ---------------------------------------------------------

summarize("ARABIAN SEA", arabian)
summarize("BAY OF BENGAL", bay)

print("\n========================================")
print("OUTSIDE / UNCLASSIFIED")
print("========================================")
print(f"Records: {len(outside)}")

print("\n========================================")
print("FINAL SUMMARY")
print("========================================")
print(f"Total loaded: {len(records)}")
print(f"Arabian Sea: {len(arabian)}")
print(f"Bay of Bengal: {len(bay)}")
print(f"Outside/unclassified: {len(outside)}")
print(f"Regional total: {len(arabian) + len(bay)}")