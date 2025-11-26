from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from .common import PROJECT_ROOT, get_device, set_seed
from .building_seg_model import BuildingSegModel
from .dataset_building_seg import BuildingSegDataset


CONFIG: Dict = {
    "DATA_ROOT": str(PROJECT_ROOT / "data" / "xBD Dataset"),
    "RESIZE": 512,
    "TRAIN_SPLIT": "train",
    "EPOCHS": 30,
    "LR": 1e-4,
    "BATCH_SIZE": 2,  # keep small for Mac CPU
    "NUM_WORKERS": 0,  # 0 is safest on macOS
    "SAVE_BEST_PATH": str(PROJECT_ROOT / "models" / "building_seg_best.pt"),
    "SEED": 42,
}


def split_train_val_ids(
    root: str,
    split: str,
    val_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[list, list]:
    """
    Random 80/20 split of tile IDs for segmentation training/validation.
    """
    tmp_ds = BuildingSegDataset(root=root, split=split, resize=CONFIG["RESIZE"])
    all_ids = tmp_ds.tile_ids
    n = len(all_ids)
    idxs = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idxs)
    n_val = int(val_frac * n)
    val_idxs = idxs[:n_val]
    train_idxs = idxs[n_val:]
    train_ids = [all_ids[i] for i in train_idxs]
    val_ids = [all_ids[i] for i in val_idxs]
    return train_ids, val_ids


def compute_pos_weight(ds: BuildingSegDataset, max_tiles: int = 200) -> float:
    """
    Estimate pos_weight for BCEWithLogitsLoss:
    pos_weight ≈ (#background pixels) / (#building pixels)
    """
    total_pos = 0.0
    total_neg = 0.0
    for i in range(len(ds)):
        if i >= max_tiles:
            break
        _, mask, _ = ds[i]
        pos = mask.sum().item()
        neg = mask.numel() - pos
        total_pos += pos
        total_neg += neg
    if total_pos == 0:
        return 1.0
    return float(total_neg / total_pos)


def compute_seg_metrics(
    logits: torch.Tensor, masks: torch.Tensor, eps: float = 1e-6
) -> Tuple[float, float, float]:
    """
    Compute pixel accuracy, IoU, and Dice for a batch.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    correct = (preds == masks).float().mean().item()

    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - intersection
    iou = ((intersection + eps) / (union + eps)).mean().item()

    dice = (
        (2 * intersection + eps)
        / (preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + eps)
    ).mean().item()

    return correct, iou, dice


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    total_iou = 0.0
    total_dice = 0.0
    n_samples = 0

    for imgs, masks, _ in tqdm(loader, desc="train", leave=False):
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(imgs)  # (B,1,H,W)

        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        acc, iou, dice = compute_seg_metrics(logits.detach(), masks)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_acc += acc * bs
        total_iou += iou * bs
        total_dice += dice * bs
        n_samples += bs

    if n_samples == 0:
        return {"loss": 0.0, "acc": 0.0, "iou": 0.0, "dice": 0.0}

    return {
        "loss": total_loss / n_samples,
        "acc": total_acc / n_samples,
        "iou": total_iou / n_samples,
        "dice": total_dice / n_samples,
    }


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_iou = 0.0
    total_dice = 0.0
    n_samples = 0

    for imgs, masks, _ in tqdm(loader, desc="val", leave=False):
        imgs = imgs.to(device)
        masks = masks.to(device)

        logits = model(imgs)

        loss = criterion(logits, masks)

        acc, iou, dice = compute_seg_metrics(logits, masks)
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_acc += acc * bs
        total_iou += iou * bs
        total_dice += dice * bs
        n_samples += bs

    if n_samples == 0:
        return {"loss": 0.0, "acc": 0.0, "iou": 0.0, "dice": 0.0}

    return {
        "loss": total_loss / n_samples,
        "acc": total_acc / n_samples,
        "iou": total_iou / n_samples,
        "dice": total_dice / n_samples,
    }


def main():
    cfg = CONFIG
    set_seed(cfg["SEED"])
    device = get_device(prefer_gpu=False)  # force CPU on Mac
    print("Device:", device)

    # ---- Split tiles into train / val ----
    train_ids, val_ids = split_train_val_ids(
        root=cfg["DATA_ROOT"],
        split=cfg["TRAIN_SPLIT"],
        val_frac=0.2,
        seed=cfg["SEED"],
    )
    print(f"Train tiles: {len(train_ids)}, Val tiles: {len(val_ids)}")

    train_ds = BuildingSegDataset(
        root=cfg["DATA_ROOT"],
        split=cfg["TRAIN_SPLIT"],
        resize=cfg["RESIZE"],
        tile_ids=train_ids,
        use_augmentation=True,
    )
    val_ds = BuildingSegDataset(
        root=cfg["DATA_ROOT"],
        split=cfg["TRAIN_SPLIT"],
        resize=cfg["RESIZE"],
        tile_ids=val_ids,
        use_augmentation=False,
    )

    # Estimate pos_weight from a subset
    pos_weight_value = compute_pos_weight(train_ds, max_tiles=200)
    print(f"Estimated pos_weight (buildings): {pos_weight_value:.3f}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=True,
        num_workers=cfg["NUM_WORKERS"],
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=False,
        num_workers=cfg["NUM_WORKERS"],
        pin_memory=False,
    )

    # Model + loss + optimizer
    model = BuildingSegModel(pretrained=True).to(device)

    pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["LR"], weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["EPOCHS"], eta_min=1e-6
    )

    best_val_iou = 0.0
    save_path = Path(cfg["SAVE_BEST_PATH"])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["EPOCHS"] + 1):
        print(f"\nEpoch {epoch}/{cfg['EPOCHS']}")

        tr_stats = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        print(
            f"Train loss: {tr_stats['loss']:.4f}, "
            f"acc: {tr_stats['acc']:.4f}, "
            f"IoU: {tr_stats['iou']:.4f}, "
            f"Dice: {tr_stats['dice']:.4f}"
        )

        val_stats = eval_one_epoch(
            model, val_loader, criterion, device
        )
        print(
            f"Val loss: {val_stats['loss']:.4f}, "
            f"acc: {val_stats['acc']:.4f}, "
            f"IoU: {val_stats['iou']:.4f}, "
            f"Dice: {val_stats['dice']:.4f}"
        )

        if val_stats["iou"] > best_val_iou:
            best_val_iou = val_stats["iou"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "best_val_iou": best_val_iou,
                    "epoch": epoch,
                },
                save_path,
            )
            print(
                f"[*] Saved new best model to {save_path} "
                f"(val IoU={best_val_iou:.4f})"
            )

        scheduler.step()


if __name__ == "__main__":
    main()
