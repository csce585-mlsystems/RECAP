import argparse
import json
from pathlib import Path
import pandas as pd
from shapely import wkt

DAMAGE_MAP = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}

def to_posix_rel(p: Path) -> str:
    try:
        rel = p.resolve().relative_to(Path.cwd().resolve())
        return rel.as_posix()
    except Exception:
        return p.resolve().as_posix()

def parse_labels(images_dir: Path, labels_dir: Path, split_name: str):
    rows = []
    for jf in sorted(labels_dir.glob("*.json")):
        data = json.loads(jf.read_text(encoding="utf-8"))
        features = data.get("features", {}).get("xy", [])
        prefix = jf.stem
        if prefix.endswith("_pre_disaster"):
            prefix = prefix[:-13]
        elif prefix.endswith("_post_disaster"):
            prefix = prefix[:-14]
        for feat in features:
            props = feat.get("properties", {}) or {}
            building_id = props.get("uid") or prefix
            damage = props.get("subtype")
            if damage not in DAMAGE_MAP: 
                continue
            wkt_str = feat.get("wkt")
            if not wkt_str: 
                continue
            try:
                polygon = wkt.loads(wkt_str).wkt
            except Exception:
                polygon = wkt_str
            pre = images_dir / f"{prefix}_pre_disaster.png"
            post = images_dir / f"{prefix}_post_disaster.png"
            if pre.exists() and post.exists():
                rows.append({
                    "split": split_name,
                    "event": prefix.split("_")[0],
                    "building_id": building_id,
                    "label_id": DAMAGE_MAP[damage],
                    "label_name": damage,
                    "pre_path": to_posix_rel(pre),
                    "post_path": to_posix_rel(post),
                    "polygon_wkt": polygon
                })
    return pd.DataFrame(rows)

def build_train_index(dataset_root: Path, out_file: Path):
    images = dataset_root / "train" / "images"
    labels = dataset_root / "train" / "labels"
    df = parse_labels(images, labels, "train")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"✅ Wrote {len(df)} rows to {out_file}")

def build_test_index(dataset_root: Path, out_file: Path):
    images = dataset_root / "test" / "images"
    rows = []
    for f in images.glob("*_pre_disaster.png"):
        prefix = f.name.replace("_pre_disaster.png", "")
        post = images / f"{prefix}_post_disaster.png"
        if post.exists():
            rows.append({
                "split": "test",
                "event": prefix.split("_")[0],
                "building_id": prefix,
                "pre_path": to_posix_rel(f),
                "post_path": to_posix_rel(post)
            })
    df = pd.DataFrame(rows)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"✅ Wrote {len(df)} rows to {out_file}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="data/xBD Dataset")
    args = p.parse_args()
    dataset_root = Path(args.root)
    build_train_index(dataset_root, Path("info/index.csv"))
    build_test_index(dataset_root, Path("info/test_index.csv"))

if __name__ == "__main__":
    main()
