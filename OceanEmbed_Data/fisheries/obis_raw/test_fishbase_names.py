import csv
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

INPUT = r"D:\oceanembed-ml\OceanEmbed_Data\fisheries\obis_raw\obis_species_master.csv"

TEST_SPECIES = [
    "Thunnus albacares",
    "Katsuwonus pelamis",
    "Siganus javus",
    "Terapon jarbua",
    "Anodontostoma chacunda",
    "Lutjanus lutjanus",
    "Epinephelus coioides",
    "Abudefduf vaigiensis",
]

BASE_URL = "https://www.fishbase.se"

HEADERS = {
    "User-Agent": "OceanEmbed Fisheries Research/1.0"
}


def get_india_names(scientific_name):
    """
    Search FishBase's India common-name pages for a scientific name.
    This is a TEST ONLY. It does not modify the project database.
    """

    url = (
        BASE_URL
        + "/country/CountryComNamesSearchList.php"
        + "?Country=356"
        + "&showAll=yes"
    )

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=30
        )
        response.raise_for_status()

    except Exception as e:
        return {
            "scientific_name": scientific_name,
            "status": "REQUEST_FAILED",
            "error": str(e),
            "matches": []
        }

    soup = BeautifulSoup(response.text, "html.parser")

    matches = []

    # Search table rows for the scientific name.
    for row in soup.find_all("tr"):
        cells = row.find_all(["td", "th"])

        if len(cells) < 3:
            continue

        values = [
            cell.get_text(" ", strip=True)
            for cell in cells
        ]

        if scientific_name.lower() in " ".join(values).lower():
            matches.append(values)

    return {
        "scientific_name": scientific_name,
        "status": "FOUND" if matches else "NO_MATCH_ON_PAGE",
        "error": "",
        "matches": matches
    }


def main():

    print("=" * 70)
    print("FISHBASE INDIA NAME RESOLUTION TEST")
    print("=" * 70)

    print()
    print("This test does NOT modify your CSV.")
    print()

    for species in TEST_SPECIES:

        print("-" * 70)
        print(f"Species: {species}")

        result = get_india_names(species)

        print(f"Status: {result['status']}")

        if result["error"]:
            print(f"Error: {result['error']}")

        if result["matches"]:
            for match in result["matches"]:
                print("  ", " | ".join(match))
        else:
            print("  No match found on returned page.")

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()