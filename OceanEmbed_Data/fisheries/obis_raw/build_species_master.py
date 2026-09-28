import json
import csv
from pathlib import Path
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

RAW_DIR = Path(".")

LAT_MIN = 5.0
LAT_MAX = 30.0
LON_MIN = 45.0
LON_MAX = 105.0

GRID_SIZE = 0.25
REGION_SPLIT_LON = 77.0

OUTPUT_FILE = RAW_DIR / "obis_species_master.csv"


# ============================================================
# HELPERS
# ============================================================

def get_float(record, key):
    value = record.get(key)

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_lat(record):
    return get_float(record, "decimalLatitude")


def get_lon(record):
    return get_float(record, "decimalLongitude")


def get_depth(record):
    """
    Return a usable representative depth.

    Priority:
    1. depthInMeters
    2. midpoint of minimumDepthInMeters and maximumDepthInMeters
    3. minimum depth
    4. maximum depth
    """

    depth = get_float(record, "depthInMeters")

    if depth is not None:
        return depth

    minimum = get_float(record, "minimumDepthInMeters")
    maximum = get_float(record, "maximumDepthInMeters")

    if minimum is not None and maximum is not None:
        return (minimum + maximum) / 2.0

    if minimum is not None:
        return minimum

    if maximum is not None:
        return maximum

    return None


def is_species_level(record):
    rank = str(record.get("taxonRank") or "").strip().lower()

    return rank in {
        "species",
        "subspecies",
    }


def get_species_name(record):
    """
    Prefer scientificName.
    Fall back to acceptedScientificName.
    """

    name = record.get("scientificName")

    if name:
        return str(name).strip()

    name = record.get("acceptedScientificName")

    if name:
        return str(name).strip()

    return None


def get_accepted_name(record):
    name = record.get("acceptedScientificName")

    if name:
        return str(name).strip()

    return None


def get_taxon_id(record):
    """
    OBIS/WoRMS identifiers may appear under different fields.
    """

    possible_fields = [
        "taxonID",
        "acceptedTaxonID",
        "scientificNameID",
    ]

    for field in possible_fields:

        value = record.get(field)

        if value:
            return str(value).strip()

    return None


def get_dataset(record):
    dataset = record.get("datasetName")

    if dataset:
        return str(dataset).strip()

    dataset = record.get("datasetID")

    if dataset:
        return str(dataset).strip()

    return "Unknown"


def get_grid_cell(lat, lon):

    lat_index = int((lat - LAT_MIN) // GRID_SIZE)
    lon_index = int((lon - LON_MIN) // GRID_SIZE)

    lat_min = LAT_MIN + lat_index * GRID_SIZE
    lon_min = LON_MIN + lon_index * GRID_SIZE

    return (
        round(lat_min, 2),
        round(lon_min, 2),
    )


def get_region(lon):

    if lon < REGION_SPLIT_LON:
        return "Arabian Sea"

    return "Bay of Bengal"


# ============================================================
# LOAD RECORDS
# ============================================================

files = sorted(RAW_DIR.glob("obis_fish_*.json"))

print("=" * 70)
print("BUILDING OBIS SPECIES MASTER")
print("=" * 70)

print(f"Files found: {len(files)}")

records_loaded = 0
species_records = 0
records_skipped = 0


# ============================================================
# SPECIES DATABASE
# ============================================================

species_data = defaultdict(
    lambda: {
        "scientific_name": None,
        "accepted_scientific_name": None,
        "taxon_id": None,

        "total_observations": 0,

        "arabian_observations": 0,
        "bay_observations": 0,

        "arabian_depth_records": 0,
        "bay_depth_records": 0,

        "arabian_cells": set(),
        "bay_cells": set(),

        "depth_min": None,
        "depth_max": None,

        "datasets": set(),
    }
)


# ============================================================
# PROCESS FILES
# ============================================================

for file in files:

    print(f"Loading: {file.name}")

    try:

        with open(file, "r", encoding="utf-8") as f:
            batch = json.load(f)

    except Exception as e:

        print(f"ERROR reading {file.name}: {e}")
        continue

    for record in batch:

        records_loaded += 1

        # ----------------------------------------------------
        # Only species/subspecies records
        # ----------------------------------------------------

        if not is_species_level(record):

            records_skipped += 1
            continue

        scientific_name = get_species_name(record)

        if not scientific_name:

            records_skipped += 1
            continue

        lat = get_lat(record)
        lon = get_lon(record)

        if lat is None or lon is None:

            records_skipped += 1
            continue

        # ----------------------------------------------------
        # Domain check
        # ----------------------------------------------------

        if not (
            LAT_MIN <= lat <= LAT_MAX
            and LON_MIN <= lon <= LON_MAX
        ):

            records_skipped += 1
            continue

        species_records += 1

        # ----------------------------------------------------
        # Species entry
        # ----------------------------------------------------

        data = species_data[scientific_name]

        data["scientific_name"] = scientific_name

        accepted_name = get_accepted_name(record)

        if accepted_name:
            data["accepted_scientific_name"] = accepted_name

        taxon_id = get_taxon_id(record)

        if taxon_id:
            data["taxon_id"] = taxon_id

        data["total_observations"] += 1

        # ----------------------------------------------------
        # Region
        # ----------------------------------------------------

        region = get_region(lon)

        cell = get_grid_cell(lat, lon)

        if region == "Arabian Sea":

            data["arabian_observations"] += 1
            data["arabian_cells"].add(cell)

        else:

            data["bay_observations"] += 1
            data["bay_cells"].add(cell)

        # ----------------------------------------------------
        # Depth
        # ----------------------------------------------------

        depth = get_depth(record)

        if depth is not None and depth >= 0:

            if region == "Arabian Sea":

                data["arabian_depth_records"] += 1

            else:

                data["bay_depth_records"] += 1

            if data["depth_min"] is None:
                data["depth_min"] = depth
            else:
                data["depth_min"] = min(
                    data["depth_min"],
                    depth,
                )

            if data["depth_max"] is None:
                data["depth_max"] = depth
            else:
                data["depth_max"] = max(
                    data["depth_max"],
                    depth,
                )

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        dataset = get_dataset(record)

        data["datasets"].add(dataset)


# ============================================================
# SORT SPECIES
# ============================================================

species_list = list(species_data.values())

species_list.sort(
    key=lambda item: item["total_observations"],
    reverse=True,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("SPECIES MASTER SUMMARY")
print("=" * 70)

print(f"Records loaded: {records_loaded:,}")
print(f"Species-level records used: {species_records:,}")
print(f"Records skipped: {records_skipped:,}")

print(
    f"Unique species/subspecies: "
    f"{len(species_list):,}"
)

arabian_species = sum(
    1
    for item in species_list
    if item["arabian_observations"] > 0
)

bay_species = sum(
    1
    for item in species_list
    if item["bay_observations"] > 0
)

both_regions = sum(
    1
    for item in species_list
    if (
        item["arabian_observations"] > 0
        and item["bay_observations"] > 0
    )
)

print(
    f"Arabian Sea species: "
    f"{arabian_species:,}"
)

print(
    f"Bay of Bengal species: "
    f"{bay_species:,}"
)

print(
    f"Species present in both regions: "
    f"{both_regions:,}"
)


# ============================================================
# TOP SPECIES
# ============================================================

print()
print("Top 30 species by total observations:")

for rank, item in enumerate(species_list[:30], 1):

    name = item["scientific_name"]

    total = item["total_observations"]

    arabian = item["arabian_observations"]

    bay = item["bay_observations"]

    cells = (
        len(item["arabian_cells"])
        + len(item["bay_cells"])
    )

    depth = (
        item["arabian_depth_records"]
        + item["bay_depth_records"]
    )

    print(
        f"{rank:2d}. "
        f"{name}: "
        f"{total:,} observations | "
        f"AS={arabian:,} | "
        f"BoB={bay:,} | "
        f"cells={cells:,} | "
        f"depth={depth:,}"
    )


# ============================================================
# MODELING CANDIDATES
# ============================================================

print()
print("=" * 70)
print("STRONGER HABITAT-MODELING CANDIDATES")
print("=" * 70)

print(
    "Criteria: ≥20 depth observations AND ≥5 occupied grid cells"
)

for region in ["Arabian Sea", "Bay of Bengal"]:

    candidates = []

    for item in species_list:

        if region == "Arabian Sea":

            depth_count = item["arabian_depth_records"]
            cell_count = len(item["arabian_cells"])

        else:

            depth_count = item["bay_depth_records"]
            cell_count = len(item["bay_cells"])

        if depth_count >= 20 and cell_count >= 5:

            candidates.append(
                (
                    item["scientific_name"],
                    depth_count,
                    cell_count,
                    item["total_observations"],
                )
            )

    candidates.sort(
        key=lambda x: (
            x[1],
            x[2],
        ),
        reverse=True,
    )

    print()
    print(region)
    print(
        f"Candidates: {len(candidates):,}"
    )

    for rank, candidate in enumerate(
        candidates[:30],
        1,
    ):

        name, depth, cells, total = candidate

        print(
            f"{rank:2d}. "
            f"{name}: "
            f"depth={depth:,}, "
            f"cells={cells:,}, "
            f"total={total:,}"
        )


# ============================================================
# WRITE CSV
# ============================================================

fieldnames = [
    "scientific_name",
    "accepted_scientific_name",
    "taxon_id",

    "total_observations",

    "arabian_observations",
    "bay_observations",

    "arabian_depth_records",
    "bay_depth_records",

    "arabian_grid_cells",
    "bay_grid_cells",
    "total_grid_cells",

    "depth_min_m",
    "depth_max_m",

    "dataset_count",
    "datasets",

    # These are intentionally blank for now.
    # They will be populated during the
    # India/local-name enrichment stage.
    "english_common_name",
    "india_common_name",
    "hindi_name",
    "malayalam_name",
    "tamil_name",
    "telugu_name",
    "odia_name",
    "bengali_name",
    "marathi_name",
    "gujarati_name",
    "kannada_name",
    "other_local_names",

    "name_source",
    "name_confidence",
]


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for item in species_list:

        datasets = sorted(item["datasets"])

        writer.writerow(
            {
                "scientific_name":
                    item["scientific_name"],

                "accepted_scientific_name":
                    item["accepted_scientific_name"]
                    or "",

                "taxon_id":
                    item["taxon_id"]
                    or "",

                "total_observations":
                    item["total_observations"],

                "arabian_observations":
                    item["arabian_observations"],

                "bay_observations":
                    item["bay_observations"],

                "arabian_depth_records":
                    item["arabian_depth_records"],

                "bay_depth_records":
                    item["bay_depth_records"],

                "arabian_grid_cells":
                    len(item["arabian_cells"]),

                "bay_grid_cells":
                    len(item["bay_cells"]),

                "total_grid_cells":
                    len(item["arabian_cells"])
                    | len(item["bay_cells"]),

                "depth_min_m":
                    (
                        f"{item['depth_min']:.2f}"
                        if item["depth_min"] is not None
                        else ""
                    ),

                "depth_max_m":
                    (
                        f"{item['depth_max']:.2f}"
                        if item["depth_max"] is not None
                        else ""
                    ),

                "dataset_count":
                    len(datasets),

                "datasets":
                    " | ".join(datasets),

                "english_common_name":
                    "",

                "india_common_name":
                    "",

                "hindi_name":
                    "",

                "malayalam_name":
                    "",

                "tamil_name":
                    "",

                "telugu_name":
                    "",

                "odia_name":
                    "",

                "bengali_name":
                    "",

                "marathi_name":
                    "",

                "gujarati_name":
                    "",

                "kannada_name":
                    "",

                "other_local_names":
                    "",

                "name_source":
                    "",

                "name_confidence":
                    "",
            }
        )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("OUTPUT")
print("=" * 70)

print(
    f"Species master written to:\n"
    f"{OUTPUT_FILE.resolve()}"
)

print()
print(
    "The local/Indian-name columns were intentionally "
    "left blank for the verified name-enrichment stage."
)

print()
print("Analysis complete.")