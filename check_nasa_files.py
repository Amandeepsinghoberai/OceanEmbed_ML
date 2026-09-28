import xarray as xr
import os

currents_dir = "data/currents"
winds_dir = "data/winds"

first_currents = os.listdir(currents_dir)[0]
first_winds = os.listdir(winds_dir)[0]

print(f"--- OSCAR currents: {first_currents} ---")
currents = xr.open_dataset(os.path.join(currents_dir, first_currents))
print(currents)

print(f"\n--- CCMP winds: {first_winds} ---")
winds = xr.open_dataset(os.path.join(winds_dir, first_winds))
print(winds)