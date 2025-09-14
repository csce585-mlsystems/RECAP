import os
import json
import pandas as pd
from shapely import wkt

# Map damage labels to integers
DAMAGE_MAP = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3
}

def parse_labels(labels_dir, images_dir, split_name):
    """Parse one split (tier1 or tier3) into a dataframe of buildings."""
    rows = []

    for label_file in os.listdir(labels_dir):
        if not (label_file.endswith(".json") or label_file.endswith(".geojson")):
            continue

        with open(os.path.join(labels_dir, label_file), "r") as f:
            data = json.load(f)

        if "features" not in data or "xy" not in data["features"]:
            print(f"⚠️ Skipping {label_file} (unexpected format, keys={list(data.keys())})")
            continue

        for feature in data["features"]["xy"]:
            building_id = feature.get("properties", {}).get("uid", None)
            damage = feature.get("properties", {}).get("subtype", None)

            if damage not in DAMAGE_MAP:
                continue

            # Get polygon from WKT instead of geometry
            wkt_str = feature.get("wkt", None)
            if wkt_str is None:
                print(f"⚠️ Skipping building {building_id} in {label_file} (no wkt key)")
                continue

            polygon = wkt.loads(wkt_str).wkt

            base_name = label_file.replace("_labels.json", ".png").replace("_labels.geojson", ".png")
            pre_path = os.path.join(images_dir, "pre", base_name)
            post_path = os.path.join(images_dir, "post", base_name)

            rows.append({
                "split": split_name,
                "building_id": building_id,
                "label_id": DAMAGE_MAP[damage],
                "label_name": damage,
                "pre_path": pre_path,
                "post_path": post_path,
                "polygon_wkt": polygon
            })

    return pd.DataFrame(rows)


def build_index(xbd_root, out_file="info/index.csv"):
    """Walk through xBD dataset (tier1 + tier3) and build a CSV index."""
    all_rows = []

    for split in ["tier1", "tier3"]:
        split_root = os.path.join(xbd_root, split)
        labels_dir = os.path.join(split_root, "labels")
        images_dir = os.path.join(split_root, "images")

        if not os.path.isdir(labels_dir):
            print(f"⚠️ Skipping missing split: {split}")
            continue

        print(f"📂 Parsing split: {split}")
        df_split = parse_labels(labels_dir, images_dir, split)
        print(f"   Found {len(df_split)} buildings in {split}")
        if not df_split.empty:
            all_rows.append(df_split)

    if not all_rows:
        raise RuntimeError("❌ No labeled data found. Check dataset path and file format.")

    df = pd.concat(all_rows, ignore_index=True)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"✅ Index saved to {out_file} with {len(df)} rows.")


if __name__ == "__main__":
    # 👇 Change this to your actual dataset path
    XBD_ROOT = r"C:\Users\yatin\Desktop\Data\xview2_geotiff\geotiffs"
    build_index(XBD_ROOT, out_file="info/index.csv")
