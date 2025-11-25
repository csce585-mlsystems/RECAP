"""
train_polygon_siamese.py

Purpose:
  Train the polygon-aware Siamese model on xBD using:

    - ResNet-34 backbone with multi-scale features (implemented in SiameseTileBackbone).
    - Per-building dataset (BuildingChangeDataset) with class-balanced sampling.
    - Rich change representation per building:
        feat = concat(v_pre, v_post, |diff|, diff).
    - Deeper MLP head with dropout (DamageHead).
    - Cosine learning rate schedule.
"""

from pathlib import Path
from typing import Dict

import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .common import PROJECT_ROOT, get_device, set_seed, IDX2LABEL
from .dataset_tiles import BuildingChangeDataset
from .model_polygon_siamese import (
    SiameseTileBackbone,
    DamageHead,
    rasterize_polygon_mask,
    mask_pool_features,
)

# ---------------- CONFIG ----------------
CONFIG: Dict = {
    # data
    "DATA_ROOT": str(PROJECT_ROOT / "data" / "xBD Dataset"),
    "TRAIN_SPLIT": "train",
    "RESIZE": 512,

    # base training hyperparams
    "EPOCHS": 30,
    "LR": 2e-4,
    "WEIGHT_DECAY": 1e-4,

    # CPU vs GPU specific settings
    # (we pick at runtime based on device.type)
    "SAMPLES_PER_EPOCH_GPU": 80_000,   # number of buildings per epoch on GPU
    "SAMPLES_PER_EPOCH_CPU": 30_000,   # fewer for CPU so it finishes
    "BATCH_BUILDINGS_GPU": 64,         # per-building batch size on GPU
    "BATCH_BUILDINGS_CPU": 8,          # per-building batch size on CPU
    "NUM_WORKERS_GPU": 4,
    "NUM_WORKERS_CPU": 0,              # safer on Mac / CPU

    # model
    "IMAGENET_BACKBONE": True,
    "CLASS_WEIGHTS": [0.5, 1.2, 1.5, 1.5],  # [no, minor, major, destroyed]
    "ORD_LOSS_WEIGHT": 0.0,                 # keep off for now

    # misc
    "SEED": 42,
    "SAVE_BEST_PATH": str(PROJECT_ROOT / "models" / "polygon_siamese_best.pt"),
}


# -------------------------------------------------------
# Dataset + sampler helpers
# -------------------------------------------------------
def build_train_dataset(cfg: Dict) -> BuildingChangeDataset:
    ds = BuildingChangeDataset(
        root=cfg["DATA_ROOT"],
        split=cfg["TRAIN_SPLIT"],
        resize=cfg["RESIZE"],
        limit_tiles=None,          # all tiles by default
        use_augmentation=True,
    )
    return ds


def compute_class_counts(labels_tensor: torch.Tensor) -> np.ndarray:
    counts = np.zeros(4, dtype=np.int64)
    for y in labels_tensor:
        counts[int(y.item())] += 1
    return counts


def build_sampler(
    labels_tensor: torch.Tensor,
    num_samples_per_epoch: int,
):
    """
    Build a WeightedRandomSampler to approximately balance classes at building level.

    We:
      - compute inverse-frequency class weights
      - assign each building a weight based on its class
      - sample with replacement for num_samples_per_epoch samples
    """
    labels_np = labels_tensor.numpy()
    class_counts = np.bincount(labels_np, minlength=4)

    # inverse frequency weights
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_weights)

    sample_weights = class_weights[labels_np]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

    num_samples = min(num_samples_per_epoch, len(labels_tensor))

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True,
    )
    return sampler, class_counts, class_weights


# -------------------------------------------------------
# Feature construction
# -------------------------------------------------------
def make_change_features(
    v_pre: torch.Tensor,   # (C,)
    v_post: torch.Tensor,  # (C,)
) -> torch.Tensor:
    """
    Build rich change feature:
        diff = v_post - v_pre
        abs_diff = |diff|
        feat = concat(v_pre, v_post, abs_diff, diff) ∈ R^(4C)
    """
    diff = v_post - v_pre
    abs_diff = torch.abs(diff)
    feat = torch.cat([v_pre, v_post, abs_diff, diff], dim=0)
    return feat


# -------------------------------------------------------
# Collate function for per-building dataset
# -------------------------------------------------------
def building_collate(batch):
    """
    Custom collate for BuildingChangeDataset.

    batch: list of
      (pre, post, poly_xy, label, tile_id, orig_w, orig_h)

    Returns:
      pre_batch      : (B,3,H,W) tensor
      post_batch     : (B,3,H,W) tensor
      polys_batch    : list of np.ndarray (one per building)
      labels_batch   : (B,) tensor
      tile_ids       : list[str]
      orig_w_batch   : list[int]
      orig_h_batch   : list[int]
    """
    pre_list, post_list, poly_list, label_list, tile_ids, orig_w_list, orig_h_list = zip(
        *batch
    )
    pre_batch = torch.stack(pre_list, dim=0)
    post_batch = torch.stack(post_list, dim=0)
    labels_batch = torch.stack(label_list, dim=0)
    polys_batch = list(poly_list)
    tile_ids = list(tile_ids)
    orig_w_batch = list(orig_w_list)
    orig_h_batch = list(orig_h_list)
    return pre_batch, post_batch, polys_batch, labels_batch, tile_ids, orig_w_batch, orig_h_batch


# -------------------------------------------------------
# Training / validation loops
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
    backbone.train()
    head.train()
    ce_weight = class_weights.to(device)

    all_y = []
    all_pred = []
    total_loss = 0.0
    n_buildings = 0

    for (
        pre_batch,
        post_batch,
        polys_batch,
        labels_batch,
        tile_ids,
        orig_w_batch,
        orig_h_batch,
    ) in tqdm(loader, desc="train", leave=False):

        pre_batch = pre_batch.to(device)   # (B,3,H,W)
        post_batch = post_batch.to(device) # (B,3,H,W)
        labels_batch = labels_batch.to(device)  # (B,)

        B = pre_batch.shape[0]
        optimizer.zero_grad()

        # Encode tiles once per batch
        F_pre_batch = backbone(pre_batch)   # (B,C,Hf,Wf)
        F_post_batch = backbone(post_batch) # (B,C,Hf,Wf)

        feats = []
        ys = []

        for b in range(B):
            y = labels_batch[b]
            poly_xy = polys_batch[b]

            if isinstance(orig_w_batch[b], torch.Tensor):
                ow = float(orig_w_batch[b].item())
            else:
                ow = float(orig_w_batch[b])
            if isinstance(orig_h_batch[b], torch.Tensor):
                oh = float(orig_h_batch[b].item())
            else:
                oh = float(orig_h_batch[b])

            F_pre = F_pre_batch[b]   # (C,Hf,Wf)
            F_post = F_post_batch[b] # (C,Hf,Wf)

            C, Hf, Wf = F_pre.shape
            mask = rasterize_polygon_mask(poly_xy, Hf, Wf, ow, oh, device)
            if mask.sum() < 1.0:
                continue

            v_pre = mask_pool_features(F_pre, mask)
            v_post = mask_pool_features(F_post, mask)
            feat = make_change_features(v_pre, v_post)  # (4C,)

            feats.append(feat)
            ys.append(int(y.item()))

        if len(feats) == 0:
            continue

        X = torch.stack(feats, dim=0)  # (N_buildings_in_batch, 4C)
        y_true = torch.tensor(ys, device=device, dtype=torch.long)

        logits = head(X)  # (N,4)

        # class-weighted CE
        ce = F.cross_entropy(logits, y_true, weight=ce_weight)
        loss = ce

        # optional ordinal loss (kept off by default)
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

    return {"loss": total_loss / n_buildings, "macro_f1": macro_f1}


@torch.no_grad()
def eval_one_epoch(
    backbone: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    backbone.eval()
    head.eval()

    all_y = []
    all_pred = []

    for (
        pre_batch,
        post_batch,
        polys_batch,
        labels_batch,
        tile_ids,
        orig_w_batch,
        orig_h_batch,
    ) in tqdm(loader, desc="val", leave=False):

        pre_batch = pre_batch.to(device)
        post_batch = post_batch.to(device)
        labels_batch = labels_batch.to(device)

        B = pre_batch.shape[0]

        F_pre_batch = backbone(pre_batch)
        F_post_batch = backbone(post_batch)

        feats = []
        ys = []

        for b in range(B):
            y = labels_batch[b]
            poly_xy = polys_batch[b]

            if isinstance(orig_w_batch[b], torch.Tensor):
                ow = float(orig_w_batch[b].item())
            else:
                ow = float(orig_w_batch[b])
            if isinstance(orig_h_batch[b], torch.Tensor):
                oh = float(orig_h_batch[b].item())
            else:
                oh = float(orig_h_batch[b])

            F_pre = F_pre_batch[b]
            F_post = F_post_batch[b]

            C, Hf, Wf = F_pre.shape
            mask = rasterize_polygon_mask(poly_xy, Hf, Wf, ow, oh, device)
            if mask.sum() < 1.0:
                continue

            v_pre = mask_pool_features(F_pre, mask)
            v_post = mask_pool_features(F_post, mask)
            feat = make_change_features(v_pre, v_post)

            feats.append(feat)
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

    is_cpu = (device.type == "cpu")

    # Pick CPU/GPU-aware settings
    samples_per_epoch = (
        cfg["SAMPLES_PER_EPOCH_CPU"] if is_cpu else cfg["SAMPLES_PER_EPOCH_GPU"]
    )
    batch_buildings = (
        cfg["BATCH_BUILDINGS_CPU"] if is_cpu else cfg["BATCH_BUILDINGS_GPU"]
    )
    num_workers = cfg["NUM_WORKERS_CPU"] if is_cpu else cfg["NUM_WORKERS_GPU"]

    print(f"Using batch size (buildings): {batch_buildings}")
    print(f"Samples per epoch: {samples_per_epoch}")
    print(f"Num workers: {num_workers}")

    # Dataset + sampler
    train_ds = build_train_dataset(cfg)
    print(f"Train buildings: {len(train_ds)}")

    sampler, class_counts, class_weights_sampler = build_sampler(
        train_ds.labels_tensor,
        samples_per_epoch,
    )
    print("Building-level class counts:", class_counts)
    print("Sampler class weights:", class_weights_sampler)

    class_weights_loss = torch.tensor(cfg["CLASS_WEIGHTS"], dtype=torch.float32)
    print("Loss class weights:", class_weights_loss.numpy())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_buildings,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=building_collate,
    )

    # Validation dataset – same split, no augmentation, no sampler
    val_ds = BuildingChangeDataset(
        root=cfg["DATA_ROOT"],
        split=cfg["TRAIN_SPLIT"],
        resize=cfg["RESIZE"],
        limit_tiles=None,
        use_augmentation=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_buildings,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=building_collate,
    )

    # Model
    backbone = SiameseTileBackbone(
        imagenet_weights=cfg["IMAGENET_BACKBONE"],
    ).to(device)
    head_in_dim = 4 * backbone.out_channels
    head = DamageHead(in_dim=head_in_dim, n_classes=4).to(device)

    params = list(backbone.parameters()) + list(head.parameters())

    # AdamW with foreach=False → fewer DirectML fallbacks
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg["LR"],
        weight_decay=cfg["WEIGHT_DECAY"],
        foreach=False,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["EPOCHS"], eta_min=1e-6
    )

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
            class_weights_loss,
            device,
            cfg["ORD_LOSS_WEIGHT"],
        )
        print(
            f"Train loss: {tr_stats['loss']:.4f}, "
            f"macro-F1: {tr_stats['macro_f1']:.4f}"
        )

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
            print(
                f"[*] Saved new best model to {save_path} "
                f"(macro-F1={best_val_f1:.4f})"
            )

        scheduler.step()


if __name__ == "__main__":
    main()
