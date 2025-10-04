# scripts/add_centroids.py
import pandas as pd
from shapely import wkt
from shapely.geometry import Polygon

df = pd.read_csv("info/predictions.csv")   # or index_with_chips.csv
cx = []
cy = []
for i, row in df.iterrows():
    try:
        poly = wkt.loads(row["polygon_wkt"])
        pt = poly.centroid
        cx.append(pt.x)
        cy.append(pt.y)
    except Exception:
        cx.append(None)
        cy.append(None)
df["centroid_x"] = cx
df["centroid_y"] = cy
df.to_parquet("info/predictions_with_centroids.parquet", index=False)  # parquet is faster/smaller
print("Wrote predictions_with_centroids.parquet")
