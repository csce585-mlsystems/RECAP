# scripts/add_chip_paths.py
import os
import pandas as pd

IN = "info/index.csv"
OUT = "info/index_with_chips.csv"
CHIP_DIR = "data/chips"

df = pd.read_csv(IN)
chip_paths = []
missing = 0
for i, row in df.iterrows():
    b = str(row["building_id"])
    candidate = os.path.join(CHIP_DIR, f"{b}.npy")
    if os.path.exists(candidate):
        chip_paths.append(candidate)
    else:
        chip_paths.append("")  # leave blank to fall back later
        missing += 1

df["chip_path"] = chip_paths
df.to_csv(OUT, index=False)
print(f"Wrote {OUT}. Missing chips: {missing} / {len(df)}")
