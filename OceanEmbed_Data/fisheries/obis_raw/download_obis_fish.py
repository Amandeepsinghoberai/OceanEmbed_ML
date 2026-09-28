import json
import time
from pathlib import Path

import requests


BASE_URL = "https://api.obis.org/v3/occurrence"

OUTPUT_DIR = Path(".")
SIZE = 10000

params = {
    "scientificname": "Actinopterygii",
    "geometry": "POLYGON((45 5,105 5,105 30,45 30,45 5))",
}

session = requests.Session()


# ---------------------------------------------------------
# Get total count
# ---------------------------------------------------------

print("Checking OBIS total count...")

count_params = {
    **params,
    "size": 0,
}

response = session.get(
    BASE_URL,
    params=count_params,
    timeout=120,
)

response.raise_for_status()

count_data = response.json()
total = count_data["total"]

print(f"OBIS reports total records: {total}")


# ---------------------------------------------------------
# Cursor-based pagination
# OBIS uses the "after" parameter.
# ---------------------------------------------------------

after = "-1"

downloaded_ids = set()

total_downloaded = 0
batch_number = 1


while total_downloaded < total:

    request_params = {
        **params,
        "after": after,
        "size": SIZE,
        "total": "false",
    }

    print(
        f"\nDownloading batch {batch_number} "
        f"(after={after}, size={SIZE})..."
    )

    response = session.get(
        BASE_URL,
        params=request_params,
        timeout=180,
    )

    response.raise_for_status()

    data = response.json()
    records = data.get("results", [])

    if not records:
        print("No records returned. Download complete.")
        break

    print(f"Records returned: {len(records)}")


    # -----------------------------------------------------
    # Get OBIS internal ID of final record.
    # This becomes the cursor for the next request.
    # -----------------------------------------------------

    last_id = records[-1].get("id")

    if last_id is None:
        print("ERROR: Last record has no OBIS 'id'.")
        break

    last_id = str(last_id)


    # -----------------------------------------------------
    # Detect pagination failure
    # -----------------------------------------------------

    if last_id == after:
        print(
            "ERROR: OBIS returned the same cursor again. "
            "Stopping to prevent duplicate downloads."
        )
        break


    # -----------------------------------------------------
    # Remove duplicates locally as an additional safeguard
    # -----------------------------------------------------

    clean_records = []

    for record in records:

        occurrence_id = (
            record.get("occurrenceID")
            or record.get("id")
        )

        if occurrence_id is None:
            continue

        occurrence_id = str(occurrence_id)

        if occurrence_id not in downloaded_ids:

            clean_records.append(record)
            downloaded_ids.add(occurrence_id)


    print(
        f"New unique records: {len(clean_records)}"
    )


    if len(clean_records) == 0:
        print(
            "ERROR: No new records found in this batch. "
            "Stopping to prevent duplicates."
        )
        break


    # -----------------------------------------------------
    # Save batch
    # -----------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / f"obis_fish_{batch_number:03d}.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            clean_records,
            f,
            ensure_ascii=False,
        )


    print(
        f"Saved {len(clean_records)} records -> "
        f"{output_file.name}"
    )

    total_downloaded += len(clean_records)

    print(
        f"Total unique records downloaded: "
        f"{total_downloaded}"
    )


    # -----------------------------------------------------
    # Move cursor forward
    # -----------------------------------------------------

    after = last_id

    batch_number += 1

    time.sleep(1)


# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------

print("\n========================================")
print("Download finished")
print("========================================")

print(f"OBIS reported total: {total}")
print(f"Unique records downloaded: {total_downloaded}")
print(f"Unique occurrence IDs: {len(downloaded_ids)}")