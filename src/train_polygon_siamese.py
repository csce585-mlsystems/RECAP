"""
train_polygon_siamese.py
Purpose:
  Train the polygon-aware tile-level Siamese model on xBD.

High-level idea:
  - Each training sample is one post-disaster tile + its matching pre-disaster tile.
  - For that tile we also get:
      - list of building polygons (in xy pixel coordinates)
      - one damage label per building (0=no, 1=minor, 2=major, 3=destroyed)
  - Pipeline:
      1) Encode pre and post tiles with a shared ResNet-18 backbone.
      2) For each polygon, rasterize a mask on the feature map and average-pool features.
      3) Take |F_post - F_pre| as the "change" vector for that building.
      4) Classify into 4 damage classes with a small MLP head.
  - Loss:
      - Class-weighted cross-entropy (fixed weights to fight imbalance).
      - No ordinal/temperature scaling to keep it simpler.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .common import PROJECT_ROOT, get_device, set_seed, IDX2LABEL
from .dataset_tiles import TileBuildingDataset
from .model_polygon_siamese import (
    SiameseTileBackbone,
    DamageHead,
    rasterize_polygon_mask,
    mask_pool_features,
)

# ---------------- CONFIG (easy to edit) ----------------
CONFIG: Dict = {
    # data
    "DATA_ROOT": str(PROJECT_ROOT / "data" / "xBD Dataset"),
    "TRAIN_SPLIT": "train",   # we split this into train/val inside the script
    "RESIZE": 512,

    # training hyperparams
    "EPOCHS": 22, #15 for cpu and 22 for gpu             # for full run; start with 3–5 when debugging
    "LR": 1e-4, #1e-4 for gpu and 3e-4 for cpu
    "BATCH_TILES": 2, #1 for CPU and 2 for GPU        # one tile at a time (variable #buildings)
    "LIMIT_TILES_TRAIN": None,
    "LIMIT_TILES_VAL": None,

    # model
    "IMAGENET_BACKBONE": True,      # use resnet18 ImageNet weights
    "CLASS_WEIGHTS": [0.25, 1.5, 2.0, 1.75],  # [no, minor, major, destroyed]
    "ORD_LOSS_WEIGHT": 0.0,         # keep at 0.0 (no ordinal reg for now)

    # misc
    "SEED": 42,
    "NUM_WORKERS": 0,   # 0 = safer cross-platform; increase if you want speed
    "SAVE_BEST_PATH": str(PROJECT_ROOT / "models" / "polygon_siamese_best.pt"),
}


# -------------------------------------------------------
# Dataset construction
# -------------------------------------------------------
def build_datasets(cfg: Dict) -> Tuple[TileBuildingDataset, TileBuildingDataset]:
    """
    Build Train/Val splits from the xBD 'train' split.

    We:
      - Load all tiles from DATA_ROOT / TRAIN_SPLIT.
      - Randomly shuffle with a fixed SEED.
      - Use 80% tiles as train, 20% as val.
    """
    root = cfg["DATA_ROOT"]
    split = cfg["TRAIN_SPLIT"]
    resize = cfg["RESIZE"]

    # Full dataset of tiles (this will scan all *_post_disaster.png under images/)
    full_ds = TileBuildingDataset(root, split=split, resize=resize, limit_tiles=None)
    n = len(full_ds)
    idxs = np.arange(n)
    rng = np.random.default_rng(cfg["SEED"])
    rng.shuffle(idxs)

    n_train = int(0.8 * n)
    train_idxs = idxs[:n_train]
    val_idxs = idxs[n_train:]

    # Re-use the same tile_ids but split
    tile_ids = full_ds.tile_ids
    train_ids = [tile_ids[i] for i in train_idxs]
    val_ids = [tile_ids[i] for i in val_idxs]

    train_ds = TileBuildingDataset(
        root,
        split=split,
        resize=resize,
        tile_ids=train_ids,
        limit_tiles=cfg["LIMIT_TILES_TRAIN"],
    )
    val_ds = TileBuildingDataset(
        root,
        split=split,
        resize=resize,
        tile_ids=val_ids,
        limit_tiles=cfg["LIMIT_TILES_VAL"],
    )
    return train_ds, val_ds


# -------------------------------------------------------
# Class counting & weighting
# -------------------------------------------------------
def compute_class_counts(ds: TileBuildingDataset) -> np.ndarray:
    """
    Count how many buildings per class in the dataset, for reporting
    and to sanity-check imbalance.
    """
    counts = np.zeros(4, dtype=np.int64)
    for _, _, _, labels, _, _, _ in tqdm(ds, desc="class-counts"):
        if labels.numel() == 0:
            continue
        for y in labels.numpy():
            counts[y] += 1
    return counts


# -------------------------------------------------------
# Training / eval loops
# -------------------------------------------------------
def train_one_epoch(
    backbone: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    device: torch.device,
    ord_loss_weight: float,
) -> Dict[str, float]:
    """
    One training epoch:
      - loop over tiles
      - encode pre/post
      - mask-pool per building
      - class-balanced CE + (optional) ordinal regularizer
    """
    backbone.train()
    head.train()
    ce_weight = class_weights.to(device)

    all_y = []
    all_pred = []
    total_loss = 0.0
    n_buildings = 0

    for pre_t, post_t, polys, labels_t, _, orig_w, orig_h in tqdm(
        loader, desc="train", leave=False
    ):
        # pre_t/post_t come from Dataset as (1, 3, H, W)
        pre_t = pre_t.to(device)
        post_t = post_t.to(device)
        labels_t = labels_t.to(device)  # shape (#buildings,)

        if labels_t.numel() == 0:
            continue

        # orig_w, orig_h might be ints or tensors; normalize to floats
        if isinstance(orig_w, torch.Tensor):
            ow = float(orig_w.item())
        else:
            ow = float(orig_w)
        if isinstance(orig_h, torch.Tensor):
            oh = float(orig_h.item())
        else:
            oh = float(orig_h)

        optimizer.zero_grad()

        # encode tiles: backbone returns (B, C, Hf, Wf); B=1 so index [0]
        F_pre = backbone(pre_t)[0]   # (C,Hf,Wf)
        F_post = backbone(post_t)[0]

        C, Hf, Wf = F_pre.shape

        feats = []
        ys = []

        # Build per-building features
        for poly_xy, y in zip(polys, labels_t):
            mask = rasterize_polygon_mask(poly_xy, Hf, Wf, ow, oh, device)
            if mask.sum() < 1.0:
                continue
            v_pre = mask_pool_features(F_pre, mask)
            v_post = mask_pool_features(F_post, mask)
            feats.append(torch.abs(v_post - v_pre))
            ys.append(int(y.item()))

        if len(feats) == 0:
            continue

        X = torch.stack(feats, dim=0)  # (#buildings, C)
        y_true = torch.tensor(ys, device=device, dtype=torch.long)

        # ---- undersample "no-damage" within each tile to fight imbalance ----
        idx_all = torch.arange(y_true.numel(), device=device)
        idx_0 = idx_all[y_true == 0]
        idx_pos = idx_all[y_true != 0]

        if idx_pos.numel() > 0 and idx_0.numel() > 0:
            max_0 = int(2.0 * idx_pos.numel())  # at most 2x positives
            if idx_0.numel() > max_0:
                perm = torch.randperm(idx_0.numel(), device=device)[:max_0]
                idx_0_keep = idx_0[perm]
            else:
                idx_0_keep = idx_0
            keep_idx = torch.cat([idx_0_keep, idx_pos], dim=0)
            X = X[keep_idx]
            y_true = y_true[keep_idx]
        # --------------------------------------------------------------------

        logits = head(X)  # (#buildings, 4)

        # class-weighted CE
        ce = F.cross_entropy(logits, y_true, weight=ce_weight)

        # optional ordinal loss (we keep this OFF by default)
        loss = ce
        if ord_loss_weight > 0.0:
            probs = torch.softmax(logits, dim=1)
            idxs = torch.arange(4, device=device).float()
            expected = (probs * idxs.unsqueeze(0)).sum(dim=1)
            mse = F.mse_loss(expected, y_true.float())
            loss = ce + ord_loss_weight * mse

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y_true.numel()
        n_buildings += y_true.numel()

        preds = logits.argmax(dim=1)
        all_y.append(y_true.detach().cpu().numpy())
        all_pred.append(preds.detach().cpu().numpy())

    if n_buildings == 0:
        return {"loss": 0.0, "macro_f1": 0.0}

    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)
    macro_f1 = f1_score(all_y, all_pred, average="macro")

    return {
        "loss": total_loss / n_buildings,
        "macro_f1": macro_f1,
    }


@torch.no_grad()
def eval_one_epoch(
    backbone: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Validation loop:
      - same as train, but no gradients
      - returns macro-F1 and full classification_report string
    """
    backbone.eval()
    head.eval()

    all_y = []
    all_pred = []

    for pre_t, post_t, polys, labels_t, _, orig_w, orig_h in tqdm(
        loader, desc="val", leave=False
    ):
        pre_t = pre_t.to(device)
        post_t = post_t.to(device)
        labels_t = labels_t.to(device)

        if labels_t.numel() == 0:
            continue

        if isinstance(orig_w, torch.Tensor):
            ow = float(orig_w.item())
        else:
            ow = float(orig_w)
        if isinstance(orig_h, torch.Tensor):
            oh = float(orig_h.item())
        else:
            oh = float(orig_h)

        F_pre = backbone(pre_t)[0]
        F_post = backbone(post_t)[0]

        C, Hf, Wf = F_pre.shape
        feats = []
        ys = []

        for poly_xy, y in zip(polys, labels_t):
            mask = rasterize_polygon_mask(poly_xy, Hf, Wf, ow, oh, device)
            if mask.sum() < 1.0:
                continue
            v_pre = mask_pool_features(F_pre, mask)
            v_post = mask_pool_features(F_post, mask)
            feats.append(torch.abs(v_post - v_pre))
            ys.append(int(y.item()))

        if len(feats) == 0:
            continue

        X = torch.stack(feats, dim=0)
        y_true = torch.tensor(ys, device=device, dtype=torch.long)

        logits = head(X)
        preds = logits.argmax(dim=1)

        all_y.append(y_true.detach().cpu().numpy())
        all_pred.append(preds.detach().cpu().numpy())

    if not all_y:
        return {"macro_f1": 0.0, "report": ""}

    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)
    macro_f1 = f1_score(all_y, all_pred, average="macro")
    report = classification_report(
        all_y,
        all_pred,
        target_names=[IDX2LABEL[i] for i in range(4)],
        digits=3,
    )

    return {"macro_f1": macro_f1, "report": report}


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    cfg = CONFIG
    set_seed(cfg["SEED"])
    device = get_device(prefer_gpu=True)
    print("Device:", device)

    # Build datasets and loaders
    train_ds, val_ds = build_datasets(cfg)
    print(f"Train tiles: {len(train_ds)}, Val tiles: {len(val_ds)}")

    train_counts = compute_class_counts(train_ds)
    print("Class counts (train):", train_counts)

    cb_weights = torch.tensor(cfg["CLASS_WEIGHTS"], dtype=torch.float32)
    print("Manual class weights:", cb_weights.numpy())

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["BATCH_TILES"],
        shuffle=True,
        num_workers=cfg["NUM_WORKERS"],
        collate_fn=lambda batch: batch[0],  # variable-length buildings → keep single sample
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["NUM_WORKERS"],
        collate_fn=lambda batch: batch[0],
    )

    # Build model
    backbone = SiameseTileBackbone(imagenet_weights=cfg["IMAGENET_BACKBONE"]).to(device)
    head = DamageHead(in_dim=backbone.out_channels, n_classes=4).to(device)

    params = list(backbone.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg["LR"], weight_decay=1e-4)

    best_val_f1 = 0.0
    save_path = Path(cfg["SAVE_BEST_PATH"])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["EPOCHS"] + 1):
        print(f"\nEpoch {epoch}/{cfg['EPOCHS']}")
        tr_stats = train_one_epoch(
            backbone,
            head,
            train_loader,
            optimizer,
            cb_weights,
            device,
            cfg["ORD_LOSS_WEIGHT"],
        )
        print(f"Train loss: {tr_stats['loss']:.4f}, macro-F1: {tr_stats['macro_f1']:.4f}")

        val_stats = eval_one_epoch(backbone, head, val_loader, device)
        print(f"Val macro-F1: {val_stats['macro_f1']:.4f}")
        print(val_stats["report"])

        if val_stats["macro_f1"] > best_val_f1:
            best_val_f1 = val_stats["macro_f1"]
            torch.save(
                {
                    "backbone": backbone.state_dict(),
                    "head": head.state_dict(),
                    "config": cfg,
                },
                save_path,
            )
            print(f"[*] Saved new best model to {save_path} (macro-F1={best_val_f1:.4f})")


if __name__ == "__main__":
    main()
