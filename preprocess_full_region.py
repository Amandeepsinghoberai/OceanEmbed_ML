import xarray as xr
import numpy as np
import dask

sst = xr.open_dataset("data/sst/sst_full_2020_2023.nc")
ssh = xr.open_dataset("data/ssh/ssh_full_2020_2023.nc")

# Load all 4 GLORYS years together as one combined dataset, chunked to
# stay memory-safe - smaller chunks than before, since this region is
# roughly 4-5x larger in area than our original Bay of Bengal box
print("Loading all 4 years of GLORYS (this may take a few minutes)...")
glorys = xr.open_mfdataset(
    [
        "data/glorys/glorys_full_2020.nc",
        "data/glorys/glorys_full_2021.nc",
        "data/glorys/glorys_full_2022.nc",
        "data/glorys/glorys_full_2023.nc",
    ],
    combine="by_coords",
    chunks={"time": 10},
)

print("SST shape:", sst["analysed_sst"].shape)
print("SSH shape:", ssh["sla"].shape)
print("GLORYS combined shape:", glorys["thetao"].shape)

# Official SIH region, 0.25 degree grid
target_lat = np.arange(5, 30.01, 0.25)
target_lon = np.arange(45, 105.01, 0.25)

sst_regrid = sst.interp(latitude=target_lat, longitude=target_lon)
ssh_regrid = ssh.interp(latitude=target_lat, longitude=target_lon)
glorys_regrid = glorys.interp(latitude=target_lat, longitude=target_lon)

sst_c = sst_regrid["analysed_sst"]
if float(sst_c.isel(time=0).mean()) > 100:
    sst_c = sst_c - 273.15
    print("Converted SST from Kelvin to Celsius")

min_depth = float(glorys_regrid["depth"].min())
sih_depths = [min_depth, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
glorys_15depth = glorys_regrid["thetao"].interp(depth=sih_depths)
glorys_15depth = glorys_15depth.assign_coords(
    depth=[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
)

print("\nSST (C) regridded shape:", sst_c.shape)
print("SSH regridded shape:", ssh_regrid["sla"].shape)
print("GLORYS 15-depth regridded shape:", glorys_15depth.shape)

# Sanity check two points: one in the Bay of Bengal, one in the Arabian Sea -
# confirming both regions are genuinely present in this wider dataset
sample_date = "2023-01-01"
print(f"\nBay of Bengal sample ({sample_date}, ~13N 88E):")
print("SST:", float(sst_c.sel(time=sample_date, latitude=13, longitude=88, method="nearest")))

print(f"\nArabian Sea sample ({sample_date}, ~15N 65E):")
print("SST:", float(sst_c.sel(time=sample_date, latitude=15, longitude=65, method="nearest")))

sst_c.to_netcdf("data/processed_sst_full.nc")
ssh_regrid["sla"].to_netcdf("data/processed_ssh_full.nc")

print("\nWriting GLORYS to disk in chunks (this is the slow part - be patient)...")
with dask.config.set(scheduler="synchronous"):
    glorys_15depth.to_netcdf("data/processed_glorys_15depth_full.nc")

print("\nSaved all full-region processed files to data/")
