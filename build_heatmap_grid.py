import copernicusmarine
import xarray as xr
import numpy as np
import json
from datetime import datetime, timedelta, timezone

today = datetime.now(timezone.utc).date()
target_date = today - timedelta(days=2)  # same safe lag as our point predictions

print(f"Fetching regional SST grid for {target_date}...")

copernicusmarine.subset(
    dataset_id="METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2",
    variables=["analysed_sst"],
    minimum_longitude=45, maximum_longitude=100,
    minimum_latitude=5, maximum_latitude=30,
    start_datetime=str(target_date), end_datetime=str(target_date),
    output_directory="data/heatmap_cache", output_filename="regional_sst_grid.nc",
)

ds = xr.open_dataset("data/heatmap_cache/regional_sst_grid.nc")
sst = ds["analysed_sst"].isel(time=0)
if float(sst.mean()) > 100:
    sst = sst - 273.15

# Downsample slightly for a lighter file - full 0.05 degree resolution is
# more detail than a heatmap actually needs to look good
sst_coarse = sst.coarsen(latitude=5, longitude=5, boundary="trim").mean()

grid_data = {
    "date": str(target_date),
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "lat_min": float(sst_coarse.latitude.min()), "lat_max": float(sst_coarse.latitude.max()),
    "lon_min": float(sst_coarse.longitude.min()), "lon_max": float(sst_coarse.longitude.max()),
    "lat_step": float(sst_coarse.latitude[1] - sst_coarse.latitude[0]),
    "lon_step": float(sst_coarse.longitude[1] - sst_coarse.longitude[0]),
    "values": [[None if np.isnan(v) else round(float(v), 2) for v in row] for row in sst_coarse.values],
}

with open("data/heatmap_cache/regional_sst_grid.json", "w") as f:
    json.dump(grid_data, f)

print(f"Saved grid: {sst_coarse.shape[0]}x{sst_coarse.shape[1]} points")
print(f"File size check needed before handing to frontend")