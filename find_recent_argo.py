import pandas as pd
import xarray as xr
import numpy as np
import requests
import os

depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

df = pd.read_csv("data/ar_index_global_prof_latest.txt", comment="#")
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d%H%M%S", errors="coerce")

os.makedirs("data/recent_argo_cache", exist_ok=True)


def find_nearest_recent_argo(target_lat, target_lon, max_days=30, max_distance_deg=2.0):
    mask = (
        (df["date"] >= pd.Timestamp.now() - pd.Timedelta(days=max_days)) &
        (df["latitude"] >= target_lat - max_distance_deg) & (df["latitude"] <= target_lat + max_distance_deg) &
        (df["longitude"] >= target_lon - max_distance_deg) & (df["longitude"] <= target_lon + max_distance_deg)
    )
    candidates = df[mask].copy()

    if len(candidates) == 0:
        return None

    candidates["distance"] = np.sqrt(
        (candidates["latitude"] - target_lat) ** 2 + (candidates["longitude"] - target_lon) ** 2
    )
    candidates = candidates.sort_values("distance")

    # Try the closest few candidates, in case the very nearest file has an issue -
    # same resilience approach we used for the live satellite fetches
    for _, best in candidates.head(5).iterrows():
        try:
            file_url = f"https://data-argo.ifremer.fr/dac/{best['file']}"
            local_path = f"data/recent_argo_cache/{best['file'].split('/')[-1]}"

            r = requests.get(file_url, timeout=30)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)

            argo = xr.open_dataset(local_path)
            pres = argo["PRES"].values[0]
            temp = argo["TEMP"].values[0]
            temp_qc = argo["TEMP_QC"].values[0]

            valid = ~np.isnan(pres) & ~np.isnan(temp)
            qc_str = [q.decode() if isinstance(q, bytes) else str(q) for q in temp_qc]
            good_qc = np.array([q in ("1", "2") for q in qc_str])
            keep = valid & good_qc

            if keep.sum() < 2:
                continue

            # Sort by pressure explicitly before interpolating - real Argo
            # files aren't guaranteed to be stored shallow-to-deep, and
            # interpolation requires sorted input to give correct results
            sort_idx = np.argsort(pres[keep])
            pres_sorted = pres[keep][sort_idx]
            temp_sorted = temp[keep][sort_idx]

            # Only trust this as a "surface" reading if the shallowest real
            # measurement is genuinely close to the surface - otherwise this
            # float simply doesn't sample near-surface water, and we should
            # skip it rather than silently extrapolate a wrong value
            if pres_sorted[0] > 15:
                continue

            surface_temp = float(np.interp(0, pres_sorted, temp_sorted))
            days_ago = (pd.Timestamp.now() - best["date"]).days

            return {
                "surface_temp_c": round(surface_temp, 2),
                "date": str(best["date"].date()),
                "days_ago": days_ago,
                "lat": round(float(best["latitude"]), 2),
                "lon": round(float(best["longitude"]), 2),
                "distance_km": round(float(best["distance"]) * 111, 0),
            }
        except Exception:
            continue  # this specific file had an issue, try the next-nearest one

    return None


if __name__ == "__main__":
    print("Testing with the Bay of Bengal live-test point (15.0, 88.0)...")
    result = find_nearest_recent_argo(15.0, 88.0)
    print(result)

    print("\nTesting with the Arabian Sea live-test point (17.0, 58.0)...")
    result2 = find_nearest_recent_argo(17.0, 58.0)
    print(result2)