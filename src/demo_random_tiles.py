"""
demo_random_tiles.py
Purpose:
  Demo script for the polygon-aware Siamese damage model.

  - Randomly selects N tiles from the xBD "test" split.
  - For each tile:
      * Loads pre and post images.
      * Uses the trained polygon_siamese model to predict damage
        for each building polygon.
      * Creates:
          - A predicted overlay (post image + predicted damage colors).
          - A ground-truth overlay (post image + GT damage colors).
  - Computes metrics (accuracy, macro-F1, per-class stats) over
    all buildings in those tiles and prints a summary.

Outputs:
  artifacts/demo_overlays/
      <tile_id>_pred.png   (predicted damage, colored polygons)
      <tile_id>_gt.png     (ground-truth damage, colored polygons)

  Prints metrics to stdout for your README / presentation.
"""

from pathlib import Path
import json
import random
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

import torch
from torch.utils.data import DataLoader

from .common import (
    PROJECT_ROOT,
    get_device,
    set_seed,
    IDX2LABEL,
    LABEL2IDX,
    save_overlay_polygon,
    to_numpy_image,
)
from .dataset_tiles import TileBuildingDataset
from .model_polygon_siamese import (
    SiameseTileBackbone,
    DamageHead,
    rasterize_polygon_mask,
    mask_pool_features,
)


CONFIG = {
    "DATA_ROOT": str(PROJECT_ROOT / "data" / "xBD Dataset"),
    "SPLIT": "test",  # matches your folder 'test'
    "RESIZE": 512,
    "N_TILES": 10,  # how many random tiles to demo
    "SEED": 123,
    "MODEL_PATH": str(PROJECT_ROOT / "models" / "polygon_siamese_best.pt"),
    "OUT_DIR": str(PROJECT_ROOT / "artifacts" / "demo_overlays"),
    "NUM_WORKERS": 0,
}


def load_label_polygons_and_labels(
    label_json_path: Path,
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Load polygons + labels + uids from an xBD label JSON.

    Returns:
      polys_xy  : list of np.ndarray (N_i, 2) pixel coords in ORIGINAL image space
      label_idx : list of int labels {0..3}
      uids      : list of string building IDs
    """
    with label_json_path.open("r") as f:
        meta = json.load(f)

    polys_xy: List[np.ndarray] = []
    labels: List[int] = []
    uids: List[str] = []

    feats = meta["features"]["xy"]
    from shapely import wkt as shapely_wkt

    for feat in feats:
        props = feat["properties"]
        if props.get("feature_type") != "building":
            continue
        subtype = props.get("subtype", "")
        if subtype not in LABEL2IDX:
            # skip weird/unlabeled types
            continue
        lab_idx = LABEL2IDX[subtype]
        uid = props.get("uid", "")

        geom = shapely_wkt.loads(feat["wkt"])
        x, y = geom.exterior.coords.xy
        poly = np.stack([np.array(x), np.array(y)], axis=1)  # (N,2)

        polys_xy.append(poly)
        labels.append(lab_idx)
        uids.append(uid)

    return polys_xy, labels, uids


def build_pair_features(v_pre: torch.Tensor, v_post: torch.Tensor) -> torch.Tensor:
    """
    Build the SAME kind of 4x feature vector used in training:

        diff     = v_post - v_pre
        abs_diff = |diff|

        feat = [v_pre,
                v_post,
                abs_diff,
                diff]   ->  shape = (4*C,)

    This matches train_polygon_siamese.make_change_features.
    """
    diff = v_post - v_pre
    abs_diff = torch.abs(diff)
    return torch.cat([v_pre, v_post, abs_diff, diff], dim=0)  # (4*C,)


def _to_float(val) -> float:
    """Helper to safely convert Tensor/int to float."""
    if isinstance(val, torch.Tensor):
        return float(val.item())
    return float(val)


def demo_split(cfg: Dict):
    """
    Run the demo on N_TILES randomly chosen tiles from cfg["SPLIT"].
    """
    device = get_device(prefer_gpu=True)
    print("Device:", device)

    # build full test dataset, then subsample N_TILES
    full_ds = TileBuildingDataset(
        root=cfg["DATA_ROOT"],
        split=cfg["SPLIT"],
        resize=cfg["RESIZE"],
        limit_tiles=None,
    )

    n_tiles = len(full_ds)
    print(f"Total tiles in split '{cfg['SPLIT']}': {n_tiles}")
    assert n_tiles > 0, f"No tiles found for split {cfg['SPLIT']}"

    idxs = list(range(n_tiles))
    random.shuffle(idxs)
    idxs = idxs[: cfg["N_TILES"]]

    tile_paths = [full_ds.tile_paths[i] for i in idxs]
    ds = TileBuildingDataset(
        root=cfg["DATA_ROOT"],
        split=cfg["SPLIT"],
        resize=cfg["RESIZE"],
        tile_paths=tile_paths,
    )

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["NUM_WORKERS"],
        collate_fn=lambda b: b[0],  # B=1
    )

    # load model
    ckpt_path = Path(cfg["MODEL_PATH"])
    assert ckpt_path.exists(), f"Model checkpoint not found: {ckpt_path}"

    # allow full checkpoint loading (PyTorch 2.6+)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    backbone = SiameseTileBackbone(imagenet_weights=False).to(device)
    # NOTE: head expects in_dim = 4 * backbone.out_channels to match checkpoint
    head = DamageHead(in_dim=backbone.out_channels * 4, n_classes=4).to(device)

    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])
    backbone.eval()
    head.eval()

    out_dir = Path(cfg["OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)

    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    print(f"\n[Demo] Running on {len(ds)} random tiles from split '{cfg['SPLIT']}'...\n")

    for (
        pre_t,
        post_t,
        polys_xy_ds,
        labels_t_ds,
        tile_id,
        orig_w,
        orig_h,
    ) in tqdm(loader, desc="demo"):

        # Convert orig_w, orig_h to float for scaling
        ow = _to_float(orig_w)
        oh = _to_float(orig_h)

        # Re-load polygons + labels from the JSON directly (to keep GT order + uids)
        label_json_path = (
            Path(cfg["DATA_ROOT"]) / cfg["SPLIT"] / "labels" / f"{tile_id}.json"
        )
        polys_xy, labels_idx, uids = load_label_polygons_and_labels(label_json_path)

        if len(polys_xy) == 0:
            print(f"[Demo] Tile {tile_id} has no valid building polygons, skipping.")
            continue

        pre_t = pre_t.to(device)
        post_t = post_t.to(device)

        with torch.no_grad():
            F_pre = backbone(pre_t)[0]  # (C,H',W')
            F_post = backbone(post_t)[0]
        C, Hf, Wf = F_pre.shape

        feats_all = []
        gt_labels_filtered: List[int] = []
        valid_polys_orig: List[np.ndarray] = []

        # For each building polygon, mask-pool and compute 4x feature vector
        for poly_xy, lab_idx in zip(polys_xy, labels_idx):
            mask = rasterize_polygon_mask(
                poly_xy,
                Hf,
                Wf,
                ow,
                oh,
                device,
            )
            if mask.sum() < 1.0:
                continue
            v_pre = mask_pool_features(F_pre, mask)  # (C,)
            v_post = mask_pool_features(F_post, mask)  # (C,)
            feat = build_pair_features(v_pre, v_post)  # (4*C,)
            feats_all.append(feat)
            gt_labels_filtered.append(int(lab_idx))
            valid_polys_orig.append(poly_xy)

        if len(feats_all) == 0:
            print(f"[Demo] No valid masked buildings for tile {tile_id}, skipping.")
            continue

        X = torch.stack(feats_all, dim=0)  # (#buildings, 4*C)
        y_true = np.array(gt_labels_filtered, dtype=np.int64)

        with torch.no_grad():
            logits = head(X)
            probs = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)

        y_pred = preds.cpu().numpy().astype(np.int64)

        # store for global metrics
        all_true.append(y_true)
        all_pred.append(y_pred)

        # ---- Make overlays ----
        # post_t is already resized to (RESIZE, RESIZE) from the dataset
        post_rgb = to_numpy_image(post_t)  # HxWx3, H=W=RESIZE

        H_img, W_img, _ = post_rgb.shape
        scale_x = W_img / ow
        scale_y = H_img / oh

        # Scale polygons from original image coords -> resized image coords
        scaled_polys: List[np.ndarray] = []
        for poly in valid_polys_orig:
            poly_scaled = poly.astype(np.float32).copy()
            poly_scaled[:, 0] *= scale_x
            poly_scaled[:, 1] *= scale_y
            scaled_polys.append(poly_scaled)

        # predicted overlay
        pred_labels_str = [IDX2LABEL[i] for i in y_pred]
        pred_overlay_path = out_dir / f"{tile_id}_pred.png"
        save_overlay_polygon(post_rgb, scaled_polys, pred_labels_str, pred_overlay_path)

        # ground-truth overlay
        gt_labels_str = [IDX2LABEL[i] for i in y_true]
        gt_overlay_path = out_dir / f"{tile_id}_gt.png"
        save_overlay_polygon(post_rgb, scaled_polys, gt_labels_str, gt_overlay_path)

        # quick per-tile summary
        tile_acc = (y_true == y_pred).mean()
        print(
            f"[Demo] Tile {tile_id}: buildings={len(y_true)}, "
            f"tile accuracy={tile_acc:.3f}"
        )

    # ---- Global metrics over all demo buildings ----
    if not all_true:
        print("[Demo] No buildings processed; check your data or config.")
        return

    y_true_all = np.concatenate(all_true, axis=0)
    y_pred_all = np.concatenate(all_pred, axis=0)

    acc = accuracy_score(y_true_all, y_pred_all)
    macro_f1 = f1_score(y_true_all, y_pred_all, average="macro")
    weighted_f1 = f1_score(y_true_all, y_pred_all, average="weighted")
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true_all,
        y_pred_all,
        labels=[0, 1, 2, 3],
        zero_division=0,
    )
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1, 2, 3])

    print("\n================ DEMO SUMMARY (random tiles) ================")
    print(f"Total buildings evaluated: {y_true_all.size}")
    print(f"Overall accuracy      : {acc:.4f}")
    print(f"Macro F1 score        : {macro_f1:.4f}")
    print(f"Weighted F1 score     : {weighted_f1:.4f}")
    print("\nPer-class metrics:")
    for i in range(4):
        lab = IDX2LABEL[i]
        print(
            f"  {lab:13s} | "
            f"P={prec[i]:.3f}, R={rec[i]:.3f}, F1={f1[i]:.3f}, support={support[i]}"
        )
    print("\nConfusion matrix (rows=true, cols=pred):")
    labels = [IDX2LABEL[i] for i in range(4)]
    header = " " * 15 + " ".join(f"{lab[:4]:>6s}" for lab in labels)
    print(header)
    for i in range(4):
        row_vals = " ".join(f"{cm[i, j]:6d}" for j in range(4))
        print(f"{labels[i]:15s} {row_vals}")
    print("===============================================================\n")
    print(f"Overlays saved to: {out_dir}")


def main():
    cfg = CONFIG
    set_seed(cfg["SEED"])
    demo_split(cfg)


if __name__ == "__main__":
    main()
