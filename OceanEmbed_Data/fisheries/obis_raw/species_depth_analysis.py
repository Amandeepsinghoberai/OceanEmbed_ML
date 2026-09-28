import json
from pathlib import Path
from collections import Counter, defaultdict

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
    Prefer depthInMeters.
    If unavailable, use the midpoint of minimum/maximum depth.
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
    return rank in {"species", "subspecies"}


def get_species_name(record):
    name = record.get("scientificName")

    if name:
        return str(name).strip()

    name = record.get("acceptedScientificName")

    if name:
        return str(name).strip()

    return None


def get_region(lon):
    if lon < REGION_SPLIT_LON:
        return "Arabian Sea"
    return "Bay of Bengal"


def grid_cell(lat, lon):
    lat_idx = int((lat - LAT_MIN) // GRID_SIZE)
    lon_idx = int((lon - LON_MIN) // GRID_SIZE)

    lat_cell = LAT_MIN + lat_idx * GRID_SIZE
    lon_cell = LON_MIN + lon_idx * GRID_SIZE

    return (
        round(lat_cell, 2),
        round(lon_cell, 2),
    )


# ============================================================
# LOAD DATA
# ============================================================

files = sorted(RAW_DIR.glob("obis_fish_*.json"))

print(f"Files found: {len(files)}")

records = []

for file in files:

    print(f"Loading: {file.name}")

    with open(file, "r", encoding="utf-8") as f:
        batch = json.load(f)

    records.extend(batch)

print(f"Total records loaded: {len(records):,}")


# ============================================================
# DATA STRUCTURES
# ============================================================

regions = {
    "Arabian Sea": {
        "records": 0,
        "depth_records": 0,
        "species_records": 0,
        "species_with_depth": set(),
        "species_cells": defaultdict(set),
        "species_depth_records": Counter(),
        "depth_bins": Counter(),
    },
    "Bay of Bengal": {
        "records": 0,
        "depth_records": 0,
        "species_records": 0,
        "species_with_depth": set(),
        "species_cells": defaultdict(set),
        "species_depth_records": Counter(),
        "depth_bins": Counter(),
    },
}


# ============================================================
# DEPTH BIN FUNCTION
# ============================================================

def depth_bin(depth):

    if depth < 0:
        return "Invalid (<0 m)"

    if depth <= 10:
        return "0–10 m"

    if depth <= 25:
        return "10–25 m"

    if depth <= 50:
        return "25–50 m"

    if depth <= 100:
        return "50–100 m"

    if depth <= 200:
        return "100–200 m"

    if depth <= 500:
        return "200–500 m"

    if depth <= 1000:
        return "500–1000 m"

    return ">1000 m"


# ============================================================
# PROCESS
# ============================================================

valid_records = 0
species_level_records = 0
depth_records_total = 0

for record in records:

    lat = get_lat(record)
    lon = get_lon(record)

    if lat is None or lon is None:
        continue

    if not (
        LAT_MIN <= lat <= LAT_MAX
        and LON_MIN <= lon <= LON_MAX
    ):
        continue

    valid_records += 1

    region = get_region(lon)
    data = regions[region]

    data["records"] += 1

    species_level = is_species_level(record)
    species = get_species_name(record)

    if species_level:
        data["species_records"] += 1
        species_level_records += 1

        cell = grid_cell(lat, lon)

        if species:
            data["species_cells"][species].add(cell)

    depth = get_depth(record)

    if depth is not None and depth >= 0:

        data["depth_records"] += 1
        depth_records_total += 1

        data["depth_bins"][depth_bin(depth)] += 1

        if species_level and species:

            data["species_with_depth"].add(species)

            data["species_depth_records"][species] += 1


# ============================================================
# DEPTH BIN ORDER
# ============================================================

depth_order = [
    "0–10 m",
    "10–25 m",
    "25–50 m",
    "50–100 m",
    "100–200 m",
    "200–500 m",
    "500–1000 m",
    ">1000 m",
    "Invalid (<0 m)",
]


# ============================================================
# REGIONAL SUMMARY
# ============================================================

for region_name in ["Arabian Sea", "Bay of Bengal"]:

    data = regions[region_name]

    print()
    print("=" * 65)
    print(region_name.upper())
    print("=" * 65)

    print(f"Total records: {data['records']:,}")
    print(f"Species-level records: {data['species_records']:,}")
    print(f"Records with usable depth: {data['depth_records']:,}")

    if data["records"]:
        print(
            f"Depth coverage: "
            f"{data['depth_records'] / data['records'] * 100:.2f}%"
        )

    print()
    print(
        f"Unique species with ≥1 depth observation: "
        f"{len(data['species_with_depth']):,}"
    )

    print(
        f"Unique species occupying ≥1 grid cell: "
        f"{len(data['species_cells']):,}"
    )

    both = (
        set(data["species_with_depth"])
        & set(data["species_cells"])
    )

    print(
        f"Species with BOTH spatial and depth coverage: "
        f"{len(both):,}"
    )

    # --------------------------------------------------------
    # DEPTH DISTRIBUTION
    # --------------------------------------------------------

    print()
    print("Depth distribution:")

    for depth_label in depth_order:

        count = data["depth_bins"].get(depth_label, 0)

        if count:
            percentage = (
                count / data["depth_records"] * 100
                if data["depth_records"]
                else 0
            )

            print(
                f"{depth_label:>12}: "
                f"{count:7,} "
                f"({percentage:6.2f}%)"
            )

    # --------------------------------------------------------
    # TOP SPECIES BY DEPTH OBSERVATIONS
    # --------------------------------------------------------

    print()
    print("Top 30 species by depth-enabled observations:")

    top_species = data["species_depth_records"].most_common(30)

    for rank, (species, count) in enumerate(top_species, 1):

        cells = len(data["species_cells"].get(species, set()))

        print(
            f"{rank:2d}. "
            f"{species}: "
            f"{count:,} depth records, "
            f"{cells:,} grid cells"
        )

    # --------------------------------------------------------
    # TOP SPECIES BY SPATIAL COVERAGE
    # --------------------------------------------------------

    print()
    print("Top 30 species by occupied 0.25° grid cells:")

    spatial_species = sorted(
        data["species_cells"].items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )[:30]

    for rank, (species, cells) in enumerate(spatial_species, 1):

        depth_count = data["species_depth_records"].get(
            species,
            0,
        )

        print(
            f"{rank:2d}. "
            f"{species}: "
            f"{len(cells):,} cells, "
            f"{depth_count:,} depth records"
        )


# ============================================================
# CROSS-REGIONAL SPECIES
# ============================================================

arabian_species = set(
    regions["Arabian Sea"]["species_cells"].keys()
)

bay_species = set(
    regions["Bay of Bengal"]["species_cells"].keys()
)

both_regions = arabian_species & bay_species

arabian_only = arabian_species - bay_species
bay_only = bay_species - arabian_species

print()
print("=" * 65)
print("CROSS-REGIONAL SPECIES COVERAGE")
print("=" * 65)

print(
    f"Arabian Sea species: {len(arabian_species):,}"
)

print(
    f"Bay of Bengal species: {len(bay_species):,}"
)

print(
    f"Species present in BOTH regions: {len(both_regions):,}"
)

print(
    f"Arabian Sea only: {len(arabian_only):,}"
)

print(
    f"Bay of Bengal only: {len(bay_only):,}"
)


# ============================================================
# SPECIES SUITABILITY CANDIDATES
# ============================================================

print()
print("=" * 65)
print("POTENTIAL SPECIES-MODELING CANDIDATES")
print("=" * 65)

print(
    "Criteria: ≥20 depth-enabled observations "
    "AND ≥5 occupied 0.25° cells"
)

for region_name in ["Arabian Sea", "Bay of Bengal"]:

    data = regions[region_name]

    candidates = []

    for species, depth_count in data["species_depth_records"].items():

        cells = len(
            data["species_cells"].get(species, set())
        )

        if depth_count >= 20 and cells >= 5:

            candidates.append(
                (
                    species,
                    depth_count,
                    cells,
                )
            )

    candidates.sort(
        key=lambda x: (x[1], x[2]),
        reverse=True,
    )

    print()
    print(region_name)

    print(
        f"Candidate species: {len(candidates):,}"
    )

    for rank, (species, depth_count, cells) in enumerate(
        candidates[:50],
        1,
    ):

        print(
            f"{rank:2d}. "
            f"{species}: "
            f"{depth_count:,} depth records, "
            f"{cells:,} grid cells"
        )


# ============================================================
# SAVE SPECIES SUMMARY CSV
# ============================================================

output_file = RAW_DIR / "obis_species_depth_summary.csv"

with open(output_file, "w", encoding="utf-8") as f:

    f.write(
        "region,species,depth_records,grid_cells\n"
    )

    for region_name in ["Arabian Sea", "Bay of Bengal"]:

        data = regions[region_name]

        all_species = set(data["species_cells"]) | set(
            data["species_depth_records"]
        )

        for species in sorted(all_species):

            depth_count = data["species_depth_records"].get(
                species,
                0,
            )

            cell_count = len(
                data["species_cells"].get(
                    species,
                    set(),
                )
            )

            safe_species = species.replace('"', '""')

            f.write(
                f'"{region_name}",'
                f'"{safe_species}",'
                f"{depth_count},"
                f"{cell_count}\n"
            )


print()
print("=" * 65)
print("OUTPUT")
print("=" * 65)

print(
    f"Species summary written to:\n"
    f"{output_file.resolve()}"
)

print()
print("Analysis complete.")