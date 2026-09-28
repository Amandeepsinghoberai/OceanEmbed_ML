import xarray as xr
import os

profiles_dir = "data/argo/profiles"
first_file = os.listdir(profiles_dir)[0]
print(f"Inspecting: {first_file}\n")

argo = xr.open_dataset(os.path.join(profiles_dir, first_file))

print("All variable names in this file:")
print(list(argo.data_vars.keys()))

print("\n--- Key values ---")
print("Latitude:", argo["LATITUDE"].values)
print("Longitude:", argo["LONGITUDE"].values)
print("Time (JULD):", argo["JULD"].values)

print("\n--- Pressure (depth proxy) ---")
print("Shape:", argo["PRES"].shape)
print("Values:", argo["PRES"].values)

print("\n--- Temperature ---")
print("Shape:", argo["TEMP"].shape)
print("Values:", argo["TEMP"].values)