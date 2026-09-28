import pandas as pd

df = pd.read_csv("data/argo_full/argo_index_full_2020_2023.csv")
df["date"] = pd.to_datetime(df["date"])

# Arabian Sea only
df = df[df["longitude"] < 77.0]

print(f"Total Arabian Sea profiles available: {len(df)}\n")

for month in [1, 3, 6, 9, 11, 12]:
    subset = df[df["date"].dt.month == month]
    if len(subset) > 0:
        row = subset.iloc[len(subset) // 2]
        print(f"Month {month:>2}: lat={row['latitude']:.2f}, lon={row['longitude']:.2f}, date={row['date'].date()}")
    else:
        print(f"Month {month:>2}: no profiles found")