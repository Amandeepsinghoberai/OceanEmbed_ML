import pandas as pd

df = pd.read_csv("data/ar_index_global_prof_latest.txt", comment="#")
print("Sample 'file' column values, exactly as they appear in the index:")
print(df["file"].head(10).to_list())