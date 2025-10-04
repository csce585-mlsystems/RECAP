import pandas as pd
df = pd.read_csv("info/predictions.csv")   # or predictions_with_confidence.parquet
print("TOTAL rows:", len(df))
print("Columns:", df.columns.tolist())

# label_conf diagnostics
if "label_conf" in df.columns:
    print("label_conf dtype:", df["label_conf"].dtype)
    print("label_conf min/max (raw):", df["label_conf"].min(), df["label_conf"].max())
    # show a few unique sample values
    print("sample values:", df["label_conf"].dropna().unique()[:10])
else:
    print("No label_conf column found")

# event & per-event counts
if "event" in df.columns:
    print(df["event"].value_counts().head(10))

# check NaNs and strings
print("label_conf nulls:", df["label_conf"].isna().sum() if "label_conf" in df.columns else "NA")
print("Any non-numeric label_conf?", pd.to_numeric(df["label_conf"], errors="coerce").isna().sum() if "label_conf" in df.columns else "NA")

# If polygon_wkt exists: check parsing readiness (quick)
if "polygon_wkt" in df.columns:
    sample = df["polygon_wkt"].dropna().iloc[:3].tolist()
    print("polygon_wkt samples:", sample)
