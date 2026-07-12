"""
app_streamlit_map.py

Streamlit demo for the polygon-siamese damage model on xBD.

Features:
- Scans xBD `labels/` to build a list of locations based on lng/lat polygons.
- Shows all tiles as pins on a world map (train + test splits).
- Lets you pick a tile (via selector) and then:
  * shows pre- and post-disaster images,
  * runs the polygon_siamese model on that tile,
  * shows GT damage overlay and predicted damage overlay,
  * prints per-tile accuracy + class counts.

Run from repo root with:
    streamlit run src/app_streamlit_map.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from PIL import Image

import torch
from torchvision import transforms as T
from shapely import wkt as shapely_wkt

from common import (
    PROJECT_ROOT,
    get_device,
    IDX2LABEL,
    LABEL2IDX,
    to_numpy_image,
    save_overlay_polygon,
    normalize_for_backbone,
)
from model_polygon_siamese import (
    SiameseTileBackbone,
    DamageHead,
    rasterize_polygon_mask,
    mask_pool_features,
)

# Use deck.gl icon atlas (same as your example)
ICON_URL = "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas.png"

DATA_ROOT = PROJECT_ROOT / "data" / "xBD Dataset"
MODEL_PATH = PROJECT_ROOT / "models" / "polygon_siamese_best.pt"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "streamlit_overlays"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

RESIZE = 512  # must match training / demo



# Utilities to index tiles and locations from xBD labels

def compute_tile_center_from_label(label_json_path: Path) -> Tuple[float, float] | None:
    """
    Given a label JSON with features["lng_lat"], compute an approximate
    tile center (lon, lat) by averaging building polygon centroids.
    """
    with label_json_path.open("r") as f:
        meta = json.load(f)

    feats_ll = meta["features"].get("lng_lat", [])
    if not feats_ll:
        return None

    lons = []
    lats = []
    for feat in feats_ll:
        try:
            geom = shapely_wkt.loads(feat["wkt"])
            c = geom.centroid
            lons.append(float(c.x))
            lats.append(float(c.y))
        except Exception:
            continue

    if not lons:
        return None

    lon = float(np.mean(lons))
    lat = float(np.mean(lats))
    return lon, lat


def build_tile_index() -> pd.DataFrame:
    """
    Scan train/test labels and build a DataFrame with one row per post tile:

    Columns:
        tile_id, split, disaster, lon, lat,
        pre_img, post_img, label_json
    """
    rows = []

    for split in ["train", "test"]:
        labels_dir = DATA_ROOT / split / "labels"
        images_dir = DATA_ROOT / split / "images"

        if not labels_dir.exists():
            continue

        for label_path in labels_dir.glob("*.json"):
            with label_path.open("r") as f:
                meta = json.load(f)

            md = meta.get("metadata", {})
            disaster = md.get("disaster", "unknown")
            img_name = md.get("img_name", "")

            # We expect img_name to end with "_post_disaster.png"
            if "_post_disaster" not in img_name:
                # If something weird, skip
                continue

            post_img_path = images_dir / img_name
            pre_img_name = img_name.replace("_post_disaster", "_pre_disaster")
            pre_img_path = images_dir / pre_img_name

            if not post_img_path.exists() or not pre_img_path.exists():
                # skip if images are missing
                continue

            center = compute_tile_center_from_label(label_path)
            if center is None:
                continue
            lon, lat = center

            tile_id = img_name.replace(".png", "")

            rows.append(
                {
                    "tile_id": tile_id,
                    "split": split,
                    "disaster": disaster,
                    "lon": lon,
                    "lat": lat,
                    "pre_img": str(pre_img_path),
                    "post_img": str(post_img_path),
                    "label_json": str(label_path),
                }
            )

    df = pd.DataFrame(rows)
    return df



# Model loading (cached in Streamlit session)
 
@st.cache_resource
def load_model():
    device = get_device(prefer_gpu=True)
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    backbone = SiameseTileBackbone(imagenet_weights=False).to(device)
    head = DamageHead(in_dim=backbone.out_channels * 4, n_classes=4).to(device)

    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])
    backbone.eval()
    head.eval()

    return device, backbone, head



# Inference helpers

def load_polygons_and_labels(label_json_path: Path):
    """
    Load polygons in pixel coordinates + labels (as ints) from xBD label JSON.
    Uses features["xy"] and LABEL2IDX mapping.
    """
    with label_json_path.open("r") as f:
        meta = json.load(f)

    polys_xy = []
    labels_idx = []

    feats_xy = meta["features"].get("xy", [])
    for feat in feats_xy:
        props = feat.get("properties", {})
        if props.get("feature_type") != "building":
            continue
        subtype = props.get("subtype", "")
        if subtype not in LABEL2IDX:
            continue
        lab_idx = LABEL2IDX[subtype]

        geom = shapely_wkt.loads(feat["wkt"])
        x, y = geom.exterior.coords.xy
        poly = np.stack([np.array(x), np.array(y)], axis=1)

        polys_xy.append(poly)
        labels_idx.append(lab_idx)

    metadata = meta.get("metadata", {})
    orig_w = int(metadata.get("width", metadata.get("original_width", 1024)))
    orig_h = int(metadata.get("height", metadata.get("original_height", 1024)))

    return polys_xy, labels_idx, orig_w, orig_h


def build_pair_features(v_pre: torch.Tensor, v_post: torch.Tensor) -> torch.Tensor:
    """
    Build the SAME 4x feature vector used in training:

        diff     = v_post - v_pre
        abs_diff = |diff|

        feat = [v_pre,
                v_post,
                abs_diff,
                diff]  -> shape = (4*C,)

    Matches train_polygon_siamese.make_change_features.
    """
    diff = v_post - v_pre
    abs_diff = torch.abs(diff)
    return torch.cat([v_pre, v_post, abs_diff, diff], dim=0)


def run_model_on_tile(
    device,
    backbone: SiameseTileBackbone,
    head: DamageHead,
    pre_img_path: Path,
    post_img_path: Path,
    label_json_path: Path,
):
    """
    Runs the polygon_siamese model on a single tile.

    Returns:
        post_rgb (np.uint8 HxWx3),
        polys_xy (list of np.ndarray in ORIGINAL coords),
        y_true (np.array ints),
        y_pred (np.array ints),
        pred_overlay_path (Path),
        gt_overlay_path (Path),
        tile_acc (float)
    """
    # --- load polygons and labels (original image coords) ---
    polys_xy, labels_idx, orig_w, orig_h = load_polygons_and_labels(label_json_path)
    if len(polys_xy) == 0:
        return None

    y_true = np.array(labels_idx, dtype=np.int64)

    # --- image transforms (match training) ---
    transform = T.Compose(
        [
            T.Resize((RESIZE, RESIZE)),
            T.ToTensor(),
        ]
    )

    pre_img = Image.open(pre_img_path).convert("RGB")
    post_img = Image.open(post_img_path).convert("RGB")

    pre_t = transform(pre_img).unsqueeze(0).to(device)  # (1,3,H,W)
    post_t = transform(post_img).unsqueeze(0).to(device)

    with torch.no_grad():
        F_pre = backbone(normalize_for_backbone(pre_t))[0]  # (C,Hf,Wf)
        F_post = backbone(normalize_for_backbone(post_t))[0]

    C, Hf, Wf = F_pre.shape

    feats_all = []
    valid_polys_orig = []
    gt_labels_filtered = []

    # build 4x feature vector for each polygon
    for poly_xy, lab_idx in zip(polys_xy, labels_idx):
        mask = rasterize_polygon_mask(poly_xy, Hf, Wf, orig_w, orig_h, device)
        if mask.sum() < 1.0:
            continue

        v_pre = mask_pool_features(F_pre, mask)
        v_post = mask_pool_features(F_post, mask)

        feat = build_pair_features(v_pre, v_post)  # (4C,)

        feats_all.append(feat)
        valid_polys_orig.append(poly_xy)
        gt_labels_filtered.append(lab_idx)

    if len(feats_all) == 0:
        return None

    X = torch.stack(feats_all, dim=0)
    y_true_f = np.array(gt_labels_filtered, dtype=np.int64)

    with torch.no_grad():
        logits = head(X)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

    y_pred = preds.cpu().numpy().astype(np.int64)

    # --- overlays: use the exact tensor used by the model (but drop batch dim) ---
    # post_t: (1,3,H,W) -> post_t[0]: (3,H,W)
    post_rgb = to_numpy_image(post_t[0])  # resized post image: (H,W,3)

    H_img, W_img, _ = post_rgb.shape
    ow = float(orig_w)
    oh = float(orig_h)
    scale_x = W_img / ow
    scale_y = H_img / oh

    # Scale polygons from original image coords -> resized image coords
    scaled_polys = []
    for poly in valid_polys_orig:
        poly_scaled = poly.astype(np.float32).copy()
        poly_scaled[:, 0] *= scale_x
        poly_scaled[:, 1] *= scale_y
        scaled_polys.append(poly_scaled)

    tile_id = label_json_path.stem
    pred_labels_str = [IDX2LABEL[int(i)] for i in y_pred]
    gt_labels_str = [IDX2LABEL[int(i)] for i in y_true_f]

    pred_overlay_path = ARTIFACT_DIR / f"{tile_id}_pred.png"
    gt_overlay_path = ARTIFACT_DIR / f"{tile_id}_gt.png"

    # Use scaled polygons for overlays (aligned with resized image)
    save_overlay_polygon(post_rgb, scaled_polys, pred_labels_str, pred_overlay_path)
    save_overlay_polygon(post_rgb, scaled_polys, gt_labels_str, gt_overlay_path)

    tile_acc = float((y_true_f == y_pred).mean())

    return post_rgb, valid_polys_orig, y_true_f, y_pred, pred_overlay_path, gt_overlay_path, tile_acc




# Streamlit App

def main():
    st.set_page_config(page_title="xBD Damage Map Demo", layout="wide")

    st.title("🌍 xBD Disaster Map — Pre/Post Tiles + Damage Model Demo")

    st.markdown(
        """
This app:
- Reads the **xBD label JSONs** and computes a pin (lon/lat) for each post-disaster tile.
- Shows **all tiles** as markers on a world map (train + test splits).
- Lets you select a tile → shows pre/post images and **runs your polygon-siamese model**
  to generate:
  - Ground-truth damage overlay
  - Predicted damage overlay
  - Per-tile accuracy and class counts
        """
    )

    # Load tile index (cached)
    @st.cache_resource
    def _load_tile_df():
        df = build_tile_index()
        return df

    df = _load_tile_df()

    if df.empty:
        st.error("No tiles found. Check that xBD data is under data/xBD Dataset/")
        return

    st.sidebar.header("Filters")

    split_opt = st.sidebar.selectbox("Split", ["all", "train", "test"], index=2)
    disaster_opt = st.sidebar.selectbox(
        "Disaster filter", ["all"] + sorted(df["disaster"].unique())
    )

    df_filt = df.copy()
    if split_opt != "all":
        df_filt = df_filt[df_filt["split"] == split_opt]
    if disaster_opt != "all":
        df_filt = df_filt[df_filt["disaster"] == disaster_opt]

    if df_filt.empty:
        st.warning("No tiles match the selected filters.")
        return

    # Selector for active tile
    df_filt = df_filt.sort_values(["disaster", "split", "tile_id"])
    options = [
        f"{row['tile_id']} | {row['disaster']} | {row['split']}"
        for _, row in df_filt.iterrows()
    ]
    default_idx = 0

    selected_label = st.selectbox("Select a tile", options, index=default_idx)
    sel_row = df_filt.iloc[options.index(selected_label)]

    # Build map data
    df_map = df_filt[["tile_id", "lon", "lat", "disaster", "split"]].copy()
    df_map["icon_data"] = df_map.apply(
        lambda r: {
            "url": ICON_URL,
            "width": 128,
            "height": 128,
            "anchorY": 128,
        },
        axis=1,
    )

    view_state = pdk.ViewState(
        longitude=sel_row["lon"],
        latitude=sel_row["lat"],
        zoom=10,
        pitch=45,
        bearing=0,
    )

    pins_layer = pdk.Layer(
        "IconLayer",
        data=df_map,
        get_icon="icon_data",
        get_size=3,
        size_scale=12,
        get_position="[lon, lat]",
        pickable=True,
    )

    deck = pdk.Deck(
        layers=[pins_layer],
        initial_view_state=view_state,
        tooltip={"text": "{tile_id}\n{disaster}\n{split}"},
        # map_style="mapbox://styles/mapbox/light-v10",
    )

    st.pydeck_chart(deck)


    # Show images + run model for selected tile
    
    st.subheader("Selected Tile Details")

    col_info, col_imgs = st.columns([1, 3])

    with col_info:
        st.markdown(f"**Tile ID:** `{sel_row['tile_id']}`")
        st.markdown(f"**Disaster:** `{sel_row['disaster']}`")
        st.markdown(f"**Split:** `{sel_row['split']}`")
        st.markdown(
            "**Coordinates:** "
            f"{sel_row['lat']:.5f}, {sel_row['lon']:.5f}"
        )

    pre_img_path = Path(sel_row["pre_img"])
    post_img_path = Path(sel_row["post_img"])
    label_json_path = Path(sel_row["label_json"])

    with col_imgs:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Pre-disaster image**")
            st.image(str(pre_img_path), use_column_width=True)
        with c2:
            st.markdown("**Post-disaster image**")
            st.image(str(post_img_path), use_column_width=True)

    st.markdown("---")
    st.subheader("Model Inference: Ground Truth vs Prediction")

    device, backbone, head = load_model()

    with st.spinner("Running polygon-siamese model on this tile..."):
        result = run_model_on_tile(
            device,
            backbone,
            head,
            pre_img_path,
            post_img_path,
            label_json_path,
        )

    if result is None:
        st.warning("No valid building polygons found for this tile.")
        return

    (
        post_rgb,
        polys_xy,
        y_true,
        y_pred,
        pred_overlay_path,
        gt_overlay_path,
        tile_acc,
    ) = result

    col_gt, col_pred = st.columns(2)
    with col_gt:
        st.markdown("**Ground-truth damage overlay**")
        st.image(str(gt_overlay_path), use_column_width=True)
    with col_pred:
        st.markdown("**Predicted damage overlay**")
        st.image(str(pred_overlay_path), use_column_width=True)

    st.markdown(f"**Tile accuracy:** `{tile_acc:.3f}`")

    # Per-class counts for this tile
    st.markdown("**Per-class counts (this tile):**")
    counts_true = {i: int((y_true == i).sum()) for i in range(4)}
    counts_pred = {i: int((y_pred == i).sum()) for i in range(4)}

    rows = []
    for i in range(4):
        label = IDX2LABEL[i]
        rows.append(
            {
                "class": label,
                "true_count": counts_true[i],
                "pred_count": counts_pred[i],
            }
        )
    df_counts = pd.DataFrame(rows)
    st.dataframe(df_counts, hide_index=True)


if __name__ == "__main__":
    main()
