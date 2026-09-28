import earthaccess
import os

auth = earthaccess.login(persist=True)
print("Logged in:", auth.authenticated)

os.makedirs("data/currents_2020_2023", exist_ok=True)

print("\nSearching OSCAR currents for 2020-2023...")
oscar_results = earthaccess.search_data(
    short_name="OSCAR_L4_OC_FINAL_V2.0",
    temporal=("2020-01-01", "2023-12-31"),
    bounding_box=(80, 5, 100, 22),
)
print(f"Found {len(oscar_results)} OSCAR granules")

print("\nDownloading OSCAR currents (this will take a while - 4 years of daily files)...")
earthaccess.download(oscar_results, "data/currents_2020_2023")
print("OSCAR currents downloaded.")