import xarray as xr
import numpy as np
import dask

sst = xr.open_dataset("data/sst/sst_bob_2020_2023.nc")
ssh = xr.open_dataset("data/ssh/ssh_bob_2020_2023.nc")

# Smaller chunks (20 days instead of 60) to reduce peak memory per chunk
glorys = xr.open_dataset("data/glorys/glorys_temp_bob_2020_2023.nc", chunks={"time": 20})

target_lat = np.arange(5, 22.01, 0.25)
target_lon = np.arange(80, 100.01, 0.25)

sst_regrid = sst.interp(latitude=target_lat, longitude=target_lon)
ssh_regrid = ssh.interp(latitude=target_lat, longitude=target_lon)
glorys_regrid = glorys.interp(latitude=target_lat, longitude=target_lon)

sst_c = sst_regrid["analysed_sst"]
if float(sst_c.mean()) > 100:
    sst_c = sst_c - 273.15
    print("Converted SST from Kelvin to Celsius")

min_depth = float(glorys_regrid["depth"].min())
sih_depths = [min_depth, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
glorys_15depth = glorys_regrid["thetao"].interp(depth=sih_depths)
glorys_15depth = glorys_15depth.assign_coords(
    depth=[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
)

print("SST (C) regridded shape:", sst_c.shape)
print("SSH regridded shape:", ssh_regrid["sla"].shape)
print("GLORYS 15-depth regridded shape:", glorys_15depth.shape)

sample_date = "2020-01-01"
print(f"\nSample surface state ({sample_date}, ~13N 90E):")
print("SST:", float(sst_c.sel(time=sample_date, latitude=13, longitude=90, method="nearest")))
print("SSH:", float(ssh_regrid["sla"].sel(time=sample_date, latitude=13, longitude=90, method="nearest")))

sst_c.to_netcdf("data/processed_sst_4yr.nc")
ssh_regrid["sla"].to_netcdf("data/processed_ssh_4yr.nc")

# Force Dask to process one chunk at a time (synchronous), instead of several
# chunks in parallel, which was multiplying memory usage and crashing the write
print("\nWriting 4-year GLORYS to disk, one chunk at a time (slow but memory-safe)...")
with dask.config.set(scheduler="synchronous"):
    glorys_15depth.to_netcdf("data/processed_glorys_15depth_4yr.nc")

print("\nSaved 4-year processed files to data/")