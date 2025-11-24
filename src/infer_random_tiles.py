"""
infer_random_tiles.py
Purpose:
  Use the trained polygon-aware Siamese model to run on random Test tiles,
  and save:
    - color overlays of post images with predicted building damage
    - a CSV of per-building predictions (tile_id, uid, label, confidence)
"""

from pathlib import Path
import json
import random
import csv
from typing import List

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from .common import PROJECT_ROOT, get_device, set_seed, IDX2LABEL, save_overlay_polygon, to_numpy_image
from .dataset_tiles import TileBuildingDataset
from .model_polygon_siamese import (
    SiameseTileBackbone,
    DamageHead,
    rasterize_polygon_mask,
    mask_pool_features,
)


CONFIG = {
    "DATA_ROOT": str(PROJECT_ROOT / "data" / "xBD Dataset"),
    "SPLIT": "test",
    "RESIZE": 512,
    "N_TILES": 10,   # how many random tiles to run
    "SEED": 123,
    "MODEL_PATH": str(PROJECT_ROOT / "models" / "polygon_siamese_best.pt"),
    "OUT_OVERLAY_DIR": str(PROJECT_ROOT / "artifacts" / "overlays"),
    "OUT_CSV_PATH": str(PROJECT_ROOT / "artifacts" / "csv" / "random_test_predictions.csv"),
}


def main():
    cfg = CONFIG
    set_seed(cfg["SEED"])
    device = get_device(prefer_gpu=True)
    print("Device:", device)

    # build test dataset and subsample N_TILES tiles
    full_ds = TileBuildingDataset(cfg["DATA_ROOT"], split=cfg["SPLIT"], resize=cfg["RESIZE"], limit_tiles=None)
    n_tiles = len(full_ds)
    idxs = list(range(n_tiles))
    random.shuffle(idxs)
    idxs = idxs[: cfg["N_TILES"]]
    sub_paths = [full_ds.tile_paths[i] for i in idxs]

    ds = TileBuildingDataset(cfg["DATA_ROOT"], split=cfg["SPLIT"], resize=cfg["RESIZE"], tile_paths=sub_paths)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=lambda b: b[0])

    # load model
    ckpt_path = Path(cfg["MODEL_PATH"])
    assert ckpt_path.exists(), f"Model checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device)

    backbone = SiameseTileBackbone(imagenet_weights=False).to(device)
    head = DamageHead(in_dim=backbone.out_channels, n_classes=4).to(device)
    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])
    backbone.eval()
    head.eval()

    out_overlay_dir = Path(cfg["OUT_OVERLAY_DIR"])
    out_overlay_dir.mkdir(parents=True, exist_ok=True)
    out_csv_path = Path(cfg["OUT_CSV_PATH"])
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    csv_rows = []

    for pre_t, post_t, polys_xy, labels_t, tile_id, orig_w, orig_h in tqdm(loader, desc="infer"):
        pre_t = pre_t.to(device)
        post_t = post_t.to(device)

        # encode
        F_pre = backbone(pre_t)[0]   # (C,H',W')
        F_post = backbone(post_t)[0]
        C, Hf, Wf = F_pre.shape

        feats_diff = []
        poly_valid = []
        for poly_xy in polys_xy:
            mask = rasterize_polygon_mask(poly_xy, Hf, Wf, orig_w, orig_h, device)
            if mask.sum() < 1.0:
                continue
            v_pre = mask_pool_features(F_pre, mask)
            v_post = mask_pool_features(F_post, mask)
            feats_diff.append(torch.abs(v_post - v_pre))
            poly_valid.append(poly_xy)

        if len(feats_diff) == 0:
            continue

        X = torch.stack(feats_diff, dim=0)   # (#buildings,C)
        with torch.no_grad():
            logits = head(X)
            probs = torch.softmax(logits, dim=1)  # (#buildings,4)
            confs, idxs_pred = probs.max(dim=1)

        post_rgb = to_numpy_image(post_t[0])
        pred_labels = [IDX2LABEL[i.item()] for i in idxs_pred]
        save_path = out_overlay_dir / f"{tile_id}_overlay.png"
        save_overlay_polygon(post_rgb, poly_valid, pred_labels, save_path)

        # for CSV, we need building IDs; get them from JSON
        # reload the label json quickly:
        lbl_path = Path(cfg["DATA_ROOT"]) / cfg["SPLIT"] / "labels" / f"{tile_id}.json"
        with open(lbl_path, "r") as f:
            meta = json.load(f)
        feats = meta["features"]["xy"]
        uids = []
        for feat in feats:
            props = feat["properties"]
            if props.get("feature_type") != "building":
                continue
            subtype = props.get("subtype", "")
            if subtype not in IDX2LABEL.values():
                continue
            uids.append(props.get("uid", ""))

        # align uids to valid polys (assume same order after filtering)
        uids = uids[: len(poly_valid)]

        for uid, lab, conf in zip(uids, pred_labels, confs.detach().cpu().numpy()):
            csv_rows.append(
                {
                    "tile_id": tile_id,
                    "uid": uid,
                    "pred_label": lab,
                    "confidence": float(conf),
                }
            )

    # write CSV
    with out_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tile_id", "uid", "pred_label", "confidence"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Saved overlays to: {out_overlay_dir}")
    print(f"Saved CSV to: {out_csv_path}")


if __name__ == "__main__":
    main()
