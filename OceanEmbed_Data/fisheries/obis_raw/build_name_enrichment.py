import csv
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(".")

INPUT_FILE = BASE_DIR / "obis_species_master.csv"

OUTPUT_FILE = (
    BASE_DIR / "obis_species_name_enrichment.csv"
)


# ============================================================
# NAME FIELDS
# ============================================================

NAME_FIELDS = [
    "english_common_name",
    "india_common_name",

    "hindi_name",
    "malayalam_name",
    "tamil_name",
    "telugu_name",
    "bengali_name",
    "odia_name",
    "marathi_name",
    "gujarati_name",
    "kannada_name",

    "other_local_names",

    "name_source",
    "name_confidence",
]


# ============================================================
# CHECK INPUT
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Input file not found:\n"
        f"{INPUT_FILE.resolve()}"
    )


print("=" * 70)
print("BUILDING INDIA FISH NAME ENRICHMENT TABLE")
print("=" * 70)

print(
    f"Input:\n{INPUT_FILE.resolve()}"
)


# ============================================================
# READ MASTER SPECIES FILE
# ============================================================

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)

    rows = list(reader)


print(
    f"Species records loaded: "
    f"{len(rows):,}"
)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "scientific_name",
    "accepted_scientific_name",
    "taxon_id",
    "total_observations",
    "arabian_observations",
    "bay_observations",
    "arabian_grid_cells",
    "bay_grid_cells",
    "total_grid_cells",
    "arabian_depth_records",
    "bay_depth_records",
]


missing_columns = [
    column
    for column in required_columns
    if column not in reader.fieldnames
]


if missing_columns:

    raise ValueError(
        "The following required columns are missing:\n"
        + "\n".join(
            f" - {column}"
            for column in missing_columns
        )
    )


# ============================================================
# BUILD ENRICHMENT TABLE
# ============================================================

output_rows = []


for row in rows:

    scientific_name = (
        row.get("scientific_name", "")
        .strip()
    )

    accepted_name = (
        row.get(
            "accepted_scientific_name",
            ""
        ).strip()
    )

    # --------------------------------------------------------
    # Determine regional occurrence
    # --------------------------------------------------------

    try:

        arabian_observations = int(
            row.get(
                "arabian_observations",
                0
            ) or 0
        )

    except ValueError:

        arabian_observations = 0


    try:

        bay_observations = int(
            row.get(
                "bay_observations",
                0
            ) or 0
        )

    except ValueError:

        bay_observations = 0


    regions = []

    if arabian_observations > 0:
        regions.append("Arabian Sea")

    if bay_observations > 0:
        regions.append("Bay of Bengal")


    # --------------------------------------------------------
    # Determine whether depth data exists
    # --------------------------------------------------------

    try:

        arabian_depth = int(
            row.get(
                "arabian_depth_records",
                0
            ) or 0
        )

    except ValueError:

        arabian_depth = 0


    try:

        bay_depth = int(
            row.get(
                "bay_depth_records",
                0
            ) or 0
        )

    except ValueError:

        bay_depth = 0


    total_depth_records = (
        arabian_depth
        + bay_depth
    )


    # --------------------------------------------------------
    # Preserve taxonomic information
    # --------------------------------------------------------

    enrichment = {

        "scientific_name":
            scientific_name,

        "accepted_scientific_name":
            accepted_name,

        "taxon_id":
            row.get(
                "taxon_id",
                ""
            ),

        "total_observations":
            row.get(
                "total_observations",
                "0"
            ),

        "arabian_observations":
            row.get(
                "arabian_observations",
                "0"
            ),

        "bay_observations":
            row.get(
                "bay_observations",
                "0"
            ),

        "arabian_grid_cells":
            row.get(
                "arabian_grid_cells",
                "0"
            ),

        "bay_grid_cells":
            row.get(
                "bay_grid_cells",
                "0"
            ),

        "total_grid_cells":
            row.get(
                "total_grid_cells",
                "0"
            ),

        "arabian_depth_records":
            row.get(
                "arabian_depth_records",
                "0"
            ),

        "bay_depth_records":
            row.get(
                "bay_depth_records",
                "0"
            ),

        "total_depth_records":
            str(total_depth_records),

        "regions":
            " | ".join(regions),
    }


    # --------------------------------------------------------
    # Empty verified-name fields
    # --------------------------------------------------------

    for field in NAME_FIELDS:

        enrichment[field] = ""


    # --------------------------------------------------------
    # Research status
    # --------------------------------------------------------

    enrichment[
        "name_confidence"
    ] = "PENDING_RESEARCH"


    output_rows.append(
        enrichment
    )


# ============================================================
# OUTPUT COLUMNS
# ============================================================

output_fields = [

    "scientific_name",
    "accepted_scientific_name",
    "taxon_id",

    "total_observations",

    "arabian_observations",
    "bay_observations",

    "arabian_grid_cells",
    "bay_grid_cells",
    "total_grid_cells",

    "arabian_depth_records",
    "bay_depth_records",
    "total_depth_records",

    "regions",

    # --------------------------------------------------------
    # Verified/common/local names
    # --------------------------------------------------------

    "english_common_name",
    "india_common_name",

    "hindi_name",
    "malayalam_name",
    "tamil_name",
    "telugu_name",
    "bengali_name",
    "odia_name",
    "marathi_name",
    "gujarati_name",
    "kannada_name",

    "other_local_names",

    # --------------------------------------------------------
    # Provenance
    # --------------------------------------------------------

    "name_source",
    "name_confidence",
]


# ============================================================
# WRITE OUTPUT
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=output_fields
    )

    writer.writeheader()

    writer.writerows(
        output_rows
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("ENRICHMENT TABLE CREATED")
print("=" * 70)

print(
    f"Species: "
    f"{len(output_rows):,}"
)

print(
    "Verified names added: 0"
)

print(
    "Status: "
    "PENDING_RESEARCH"
)

print()
print(
    "Output:"
)

print(
    OUTPUT_FILE.resolve()
)

print()
print(
    "No local names were guessed or automatically invented."
)

print()
print("Analysis complete.")