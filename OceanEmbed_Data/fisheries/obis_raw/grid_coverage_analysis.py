import json
from pathlib import Path
from collections import defaultdict, Counter

# ============================================================
# CONFIGURATION
# ============================================================

RAW_DIR = Path(".")

# OceanEmbed domain
LAT_MIN = 5.0
LAT_MAX = 30.0
LON_MIN = 45.0
LON_MAX = 105.0

# OceanEmbed grid resolution
GRID_SIZE = 0.25

# Existing OceanEmbed regional split
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


def grid_index(value, minimum):
    """
    Convert coordinate to the 0.25 degree OceanEmbed grid index.
    """
    return int((value - minimum) // GRID_SIZE)


def grid_coordinate(index, minimum):
    """
    Return the lower-left coordinate of a grid cell.
    """
    return minimum + index * GRID_SIZE


# ============================================================
# LOAD RECORDS
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
# BUILD GRID
# ============================================================

grid = defaultdict(lambda: {
    "records": 0,
    "species_records": 0,
    "depth_records": 0,
    "species": Counter(),
})


regional_grid = {
    "Arabian Sea": defaultdict(lambda: {
        "records": 0,
        "species_records": 0,
        "depth_records": 0,
        "species": Counter(),
    }),
    "Bay of Bengal": defaultdict(lambda: {
        "records": 0,
        "species_records": 0,
        "depth_records": 0,
        "species": Counter(),
    }),
}

outside = 0
valid_coordinates = 0


# ============================================================
# PROCESS RECORDS
# ============================================================

for record in records:

    lat = get_lat(record)
    lon = get_lon(record)

    if lat is None or lon is None:
        outside += 1
        continue

    # OceanEmbed domain
    if not (
        LAT_MIN <= lat <= LAT_MAX
        and LON_MIN <= lon <= LON_MAX
    ):
        outside += 1
        continue

    valid_coordinates += 1

    lat_idx = grid_index(lat, LAT_MIN)
    lon_idx = grid_index(lon, LON_MIN)

    cell = (lat_idx, lon_idx)

    species_level = is_species_level(record)

    has_depth = (
        record.get("minimumDepthInMeters") is not None
        or record.get("depthInMeters") is not None
        or record.get("maximumDepthInMeters") is not None
    )

    species = get_species_name(record)

    # Overall grid
    grid[cell]["records"] += 1

    if species_level:
        grid[cell]["species_records"] += 1

        if species:
            grid[cell]["species"][species] += 1

    if has_depth:
        grid[cell]["depth_records"] += 1

    # Regional grid
    if lon < REGION_SPLIT_LON:
        region = "Arabian Sea"
    else:
        region = "Bay of Bengal"

    regional_grid[region][cell]["records"] += 1

    if species_level:
        regional_grid[region][cell]["species_records"] += 1

        if species:
            regional_grid[region][cell]["species"][species] += 1

    if has_depth:
        regional_grid[region][cell]["depth_records"] += 1


# ============================================================
# SUMMARY
# ============================================================

total_possible_lat_cells = int(
    round((LAT_MAX - LAT_MIN) / GRID_SIZE)
)

total_possible_lon_cells = int(
    round((LON_MAX - LON_MIN) / GRID_SIZE)
)

total_possible_cells = (
    total_possible_lat_cells * total_possible_lon_cells
)

occupied_cells = len(grid)

print()
print("=" * 55)
print("0.25° × 0.25° GRID COVERAGE")
print("=" * 55)

print(f"OceanEmbed grid cells possible: {total_possible_cells:,}")
print(f"Grid cells with ≥1 OBIS record: {occupied_cells:,}")

coverage = (
    occupied_cells / total_possible_cells * 100
    if total_possible_cells
    else 0
)

print(f"Overall spatial coverage: {coverage:.2f}%")


# ============================================================
# REGIONAL COVERAGE
# ============================================================

for region in ["Arabian Sea", "Bay of Bengal"]:

    region_grid = regional_grid[region]

    occupied = len(region_grid)

    if region == "Arabian Sea":
        region_lat_min = LAT_MIN
        region_lat_max = LAT_MAX
        region_lon_min = LON_MIN
        region_lon_max = REGION_SPLIT_LON
    else:
        region_lat_min = LAT_MIN
        region_lat_max = LAT_MAX
        region_lon_min = REGION_SPLIT_LON
        region_lon_max = LON_MAX

    possible_lat = int(
        round((region_lat_max - region_lat_min) / GRID_SIZE)
    )

    possible_lon = int(
        round((region_lon_max - region_lon_min) / GRID_SIZE)
    )

    possible = possible_lat * possible_lon

    regional_coverage = (
        occupied / possible * 100
        if possible
        else 0
    )

    total_records = sum(
        cell["records"]
        for cell in region_grid.values()
    )

    species_records = sum(
        cell["species_records"]
        for cell in region_grid.values()
    )

    depth_records = sum(
        cell["depth_records"]
        for cell in region_grid.values()
    )

    print()
    print("=" * 55)
    print(region.upper())
    print("=" * 55)

    print(f"Possible grid cells: {possible:,}")
    print(f"Occupied grid cells: {occupied:,}")
    print(f"Spatial coverage: {regional_coverage:.2f}%")
    print(f"Records: {total_records:,}")
    print(f"Species-level records: {species_records:,}")
    print(f"Records with depth: {depth_records:,}")

    # Observation density
    densities = [
        cell["records"]
        for cell in region_grid.values()
    ]

    if densities:
        print(f"Minimum records/cell: {min(densities):,}")
        print(f"Maximum records/cell: {max(densities):,}")
        print(
            f"Mean records/occupied cell: "
            f"{sum(densities) / len(densities):.2f}"
        )

    # --------------------------------------------------------
    # Top 20 cells
    # --------------------------------------------------------

    print()
    print("Top 20 grid cells by observation count:")

    top_cells = sorted(
        region_grid.items(),
        key=lambda item: item[1]["records"],
        reverse=True,
    )[:20]

    for rank, (cell, data) in enumerate(top_cells, 1):

        lat_idx, lon_idx = cell

        cell_lat = grid_coordinate(
            lat_idx,
            LAT_MIN
        )

        cell_lon = grid_coordinate(
            lon_idx,
            LON_MIN
        )

        print(
            f"{rank:2d}. "
            f"Lat {cell_lat:.2f}–{cell_lat + GRID_SIZE:.2f}, "
            f"Lon {cell_lon:.2f}–{cell_lon + GRID_SIZE:.2f}: "
            f"{data['records']:,} records, "
            f"{data['species_records']:,} species-level"
        )


# ============================================================
# DENSITY DISTRIBUTION
# ============================================================

print()
print("=" * 55)
print("OBSERVATION DENSITY DISTRIBUTION")
print("=" * 55)

density_bins = {
    "1": 0,
    "2–5": 0,
    "6–10": 0,
    "11–25": 0,
    "26–50": 0,
    "51–100": 0,
    "101–250": 0,
    "251–500": 0,
    "501–1000": 0,
    "1000+": 0,
}

for cell in grid.values():

    count = cell["records"]

    if count == 1:
        density_bins["1"] += 1
    elif count <= 5:
        density_bins["2–5"] += 1
    elif count <= 10:
        density_bins["6–10"] += 1
    elif count <= 25:
        density_bins["11–25"] += 1
    elif count <= 50:
        density_bins["26–50"] += 1
    elif count <= 100:
        density_bins["51–100"] += 1
    elif count <= 250:
        density_bins["101–250"] += 1
    elif count <= 500:
        density_bins["251–500"] += 1
    elif count <= 1000:
        density_bins["501–1000"] += 1
    else:
        density_bins["1000+"] += 1

for label, count in density_bins.items():
    print(f"{label:>10} observations/cell: {count:,} cells")


# ============================================================
# SPECIES DIVERSITY PER GRID CELL
# ============================================================

print()
print("=" * 55)
print("SPECIES DIVERSITY")
print("=" * 55)

species_counts = []

for cell in grid.values():

    unique_species = len(cell["species"])

    if unique_species > 0:
        species_counts.append(unique_species)

if species_counts:

    print(
        f"Occupied cells containing species-level records: "
        f"{len(species_counts):,}"
    )

    print(
        f"Minimum unique species/cell: "
        f"{min(species_counts):,}"
    )

    print(
        f"Maximum unique species/cell: "
        f"{max(species_counts):,}"
    )

    print(
        f"Mean unique species/occupied species cell: "
        f"{sum(species_counts) / len(species_counts):.2f}"
    )


# ============================================================
# DEPTH-COVERAGE GRID
# ============================================================

depth_cells = sum(
    1
    for cell in grid.values()
    if cell["depth_records"] > 0
)

print()
print("=" * 55)
print("DEPTH COVERAGE")
print("=" * 55)

print(
    f"Grid cells containing ≥1 depth-enabled observation: "
    f"{depth_cells:,}"
)

print(
    f"Percentage of occupied cells with depth data: "
    f"{depth_cells / occupied_cells * 100:.2f}%"
    if occupied_cells
    else "N/A"
)


# ============================================================
# SAVE COMPACT GRID CSV
# ============================================================

output_file = RAW_DIR / "obis_grid_025_summary.csv"

with open(output_file, "w", encoding="utf-8") as f:

    f.write(
        "region,lat_min,lat_max,lon_min,lon_max,"
        "records,species_records,depth_records,"
        "unique_species\n"
    )

    for region in ["Arabian Sea", "Bay of Bengal"]:

        for (lat_idx, lon_idx), data in sorted(
            regional_grid[region].items()
        ):

            lat_min = grid_coordinate(lat_idx, LAT_MIN)
            lon_min = grid_coordinate(lon_idx, LON_MIN)

            unique_species = len(data["species"])

            f.write(
                f"{region},"
                f"{lat_min:.2f},"
                f"{lat_min + GRID_SIZE:.2f},"
                f"{lon_min:.2f},"
                f"{lon_min + GRID_SIZE:.2f},"
                f"{data['records']},"
                f"{data['species_records']},"
                f"{data['depth_records']},"
                f"{unique_species}\n"
            )

print()
print("=" * 55)
print("OUTPUT")
print("=" * 55)

print(f"Grid summary written to:")
print(output_file.resolve())

print()
print("Analysis complete.")