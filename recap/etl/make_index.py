import os
import json
import pandas as pd
from shapely import wkt

# Damage mapping
DAMAGE_MAP = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3
}


def parse_labels(images_dir, labels_dir, split_name):
    """
    Parse JSON label files from xView2 Challenge format.
    Each row = one building footprint with pre/post image paths + label.
    Uses pixel coordinates from features["xy"].
    """
    rows = []

    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".json")]
    print(f"📂 Found {len(label_files)} label files in {labels_dir}")

    for label_file in label_files:
        with open(os.path.join(labels_dir, label_file), "r") as f:
            data = json.load(f)

        if "features" not in data or "xy" not in data["features"]:
            print(f"⚠️ Skipping {label_file}, no features['xy']")
            continue

        features = data["features"]["xy"]

        # Strip ".json"
        prefix = label_file.replace(".json", "")

        # Remove disaster suffixes if they exist
        if prefix.endswith("_pre_disaster"):
            prefix = prefix[:-13]
        elif prefix.endswith("_post_disaster"):
            prefix = prefix[:-14]

        for feature in features:
            props = feature.get("properties", {})
            building_id = props.get("uid")
            damage = props.get("subtype")

            if damage not in DAMAGE_MAP:
                continue

            wkt_str = feature.get("wkt")
            if not wkt_str:
                continue

            polygon = wkt.loads(wkt_str).wkt

            pre_file = f"{prefix}_pre_disaster.png"
            post_file = f"{prefix}_post_disaster.png"

            pre_path = os.path.join(images_dir, pre_file)
            post_path = os.path.join(images_dir, post_file)

            # Only add if files exist
            if os.path.exists(pre_path) and os.path.exists(post_path):
                rows.append({
                    "split": split_name,
                    "building_id": building_id,
                    "label_id": DAMAGE_MAP[damage],
                    "label_name": damage,
                    "pre_path": pre_path,
                    "post_path": post_path,
                    "polygon_wkt": polygon
                })

    print(f"✅ Parsed {len(rows)} buildings from {labels_dir}")
    return pd.DataFrame(rows)


def build_train_index(root_dir, out_file="info/index.csv"):
    images_dir = os.path.join(root_dir, "train", "images")
    labels_dir = os.path.join(root_dir, "train", "labels")

    print("📂 Parsing training set...")
    df = parse_labels(images_dir, labels_dir, "train")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"✅ Training index saved to {out_file} with {len(df)} rows.")


def build_test_index(root_dir, out_file="info/test_index.csv"):
    """
    Build a test set index with pre/post paths only (no labels).
    """
    images_dir = os.path.join(root_dir, "test", "images")

    print("📂 Building test set index...")
    rows = []
    files = [f for f in os.listdir(images_dir) if f.endswith(".png")]

    for f in files:
        if "pre_disaster" not in f:
            continue

        prefix = f.replace("_pre_disaster.png", "")
        pre_path = os.path.join(images_dir, f)
        post_file = f"{prefix}_post_disaster.png"
        post_path = os.path.join(images_dir, post_file)

        if os.path.exists(post_path):
            rows.append({
                "split": "test",
                "building_id": prefix,
                "pre_path": pre_path,
                "post_path": post_path
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"✅ Test index saved to {out_file} with {len(df)} rows.")


if __name__ == "__main__":
    ROOT_DIR = r"C:\Users\yatin\Desktop\Data"

    build_train_index(ROOT_DIR, out_file="info/index.csv")
    build_test_index(ROOT_DIR, out_file="info/test_index.csv")
