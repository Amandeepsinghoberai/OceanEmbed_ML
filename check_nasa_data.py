import earthaccess

# Log in - first run will prompt for your NASA Earthdata username and password
# and save them locally so you won't be asked again
auth = earthaccess.login(persist=True)
print("Logged in:", auth.authenticated)

# Search (not download) for OSCAR ocean currents in our region/dates
print("\n--- Searching OSCAR currents ---")
oscar_results = earthaccess.search_data(
    short_name="OSCAR_L4_OC_FINAL_V2.0",
    temporal=("2023-01-01", "2023-01-05"),
    bounding_box=(80, 5, 100, 22),
)
print(f"Found {len(oscar_results)} OSCAR granules for a 5-day test window")
if oscar_results:
    print(oscar_results[0])

# Search (not download) for CCMP winds in our region/dates
print("\n--- Searching CCMP winds ---")
ccmp_results = earthaccess.search_data(
    short_name="CCMP_WINDS_10M6HR_L4_V3.1",
    temporal=("2023-01-01", "2023-01-05"),
    bounding_box=(80, 5, 100, 22),
)
print(f"Found {len(ccmp_results)} CCMP granules for a 5-day test window")
if ccmp_results:
    print(ccmp_results[0])