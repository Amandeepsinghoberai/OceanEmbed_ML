import xarray as xr

sst = xr.open_dataset("data/sst/sst_bob_jan_mar_2023.nc")
ssh = xr.open_dataset("data/ssh/ssh_bob_jan_mar_2023.nc")
glorys = xr.open_dataset("data/glorys/glorys_temp_bob_jan_mar_2023.nc")

print("--- SST ---")
print(sst)
print("\n--- SSH ---")
print(ssh)
print("\n--- GLORYS ---")
print(glorys)