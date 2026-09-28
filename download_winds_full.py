import earthaccess
import os

auth = earthaccess.login(persist=True)
print("Logged in:", auth.authenticated)

os.makedirs("data/winds_full", exist_ok=True)

print("\nSearching CCMP winds for full region, 2020-2023...")
ccmp_results = earthaccess.search_data(
    short_name="CCMP_WINDS_10M6HR_L4_V3.1",
    temporal=("2020-01-01", "2023-12-31"),
    bounding_box=(45, 5, 105, 30),
)
print(f"Found {len(ccmp_results)} CCMP granules")

print("\nDownloading CCMP winds (this will take a while - 4 years, wider region)...")
earthaccess.download(ccmp_results, "data/winds_full")
print("CCMP winds downloaded.")