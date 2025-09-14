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
    rows = []

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".json"):
            continue

        with open(os.path.join(labels_dir, label_file), "r") as f:
            data = json.load(f)

        if "features" not in data or "xy" not in data["features"]:
            print(f"⚠️ Skipping {label_file} (unexpected format)")
            continue

        # Each feature = one building
        for feature in data["features"]["xy"]:
            building_id = feature.get("properties", {}).get("uid")
            damage = feature.get("properties", {}).get("subtype")

            if damage not in DAMAGE_MAP:
                continue

            # WKT polygon string
            wkt_str = feature.get("wkt")
            if wkt_str is None:
                continue

            polygon = wkt.loads(wkt_str).wkt

            # Base name matches pre/post image filenames
            base_name = label_file.replace("_labels.json", ".png")
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


def build_train_index(root_dir, out_file="info/index.csv"):
    images_dir = os.path.join(root_dir, "train", "images")
    labels_dir = os.path.join(root_dir, "train", "labels")

    print("📂 Parsing training set...")
    df = parse_labels(images_dir, labels_dir, "train")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"✅ Training index saved to {out_file} with {len(df)} rows.")


def build_test_index(root_dir, out_file="info/test_index.csv"):
    images_dir = os.path.join(root_dir, "test", "images")

    rows = []
    files = [f for f in os.listdir(images_dir) if f.endswith(".png")]

    # Only look at pre-disaster, then find the matching post-disaster
    for f in files:
        if "pre" not in f:
            continue

        pre_path = os.path.join(images_dir, f)
        post_name = f.replace("pre_disaster", "post_disaster")
        post_path = os.path.join(images_dir, post_name)

        if os.path.exists(post_path):
            rows.append({
                "split": "test",
                "building_id": f.replace(".png", ""),
                "pre_path": pre_path,
                "post_path": post_path
            })
        else:
            print(f"⚠️ No matching post file for {f}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"✅ Test index saved to {out_file} with {len(df)} rows.")



if __name__ == "__main__":
    ROOT_DIR = r"C:\Users\yatin\Desktop\Data"
    build_train_index(ROOT_DIR, out_file="info/index.csv")
    build_test_index(ROOT_DIR, out_file="info/test_index.csv")
