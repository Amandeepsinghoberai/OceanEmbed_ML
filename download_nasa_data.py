import earthaccess
import os

auth = earthaccess.login(persist=True)
print("Logged in:", auth.authenticated)

os.makedirs("data/currents", exist_ok=True)
os.makedirs("data/winds", exist_ok=True)

print("\nSearching OSCAR currents for full year 2023...")
oscar_results = earthaccess.search_data(
    short_name="OSCAR_L4_OC_FINAL_V2.0",
    temporal=("2023-01-01", "2023-12-31"),
    bounding_box=(80, 5, 100, 22),
)
print(f"Found {len(oscar_results)} OSCAR granules")

print("\nDownloading OSCAR currents...")
earthaccess.download(oscar_results, "data/currents")
print("OSCAR currents downloaded.")

print("\nSearching CCMP winds for full year 2023...")
ccmp_results = earthaccess.search_data(
    short_name="CCMP_WINDS_10M6HR_L4_V3.1",
    temporal=("2023-01-01", "2023-12-31"),
    bounding_box=(80, 5, 100, 22),
)
print(f"Found {len(ccmp_results)} CCMP granules")

print("\nDownloading CCMP winds...")
earthaccess.download(ccmp_results, "data/winds")
print("CCMP winds downloaded.")

print("\nAll done.")