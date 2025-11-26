from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .common import PROJECT_ROOT, get_device
from .building_seg_model import BuildingSegModel
from .dataset_building_seg import BuildingSegDataset


CONFIG: Dict = {
    "DATA_ROOT": str(PROJECT_ROOT / "data" / "xBD Dataset"),
    "RESIZE": 512,
    "BATCH_SIZE": 2,
    "NUM_WORKERS": 0,
    "CKPT_PATH": str(PROJECT_ROOT / "models" / "building_seg_best.pt"),
}


def compute_seg_metrics(
    logits: torch.Tensor, masks: torch.Tensor, eps: float = 1e-6
) -> Tuple[float, float, float]:
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


@torch.no_grad()
def eval_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    total_acc = 0.0
    total_iou = 0.0
    total_dice = 0.0
    n_samples = 0

    for imgs, masks, _ in tqdm(loader, desc="eval", leave=False):
        imgs = imgs.to(device)
        masks = masks.to(device)

        logits = model(imgs)

        acc, iou, dice = compute_seg_metrics(logits, masks)
        bs = imgs.size(0)
        total_acc += acc * bs
        total_iou += iou * bs
        total_dice += dice * bs
        n_samples += bs

    if n_samples == 0:
        return {"acc": 0.0, "iou": 0.0, "dice": 0.0}

    return {
        "acc": total_acc / n_samples,
        "iou": total_iou / n_samples,
        "dice": total_dice / n_samples,
    }


def main():
    cfg = CONFIG
    device = get_device(prefer_gpu=False)
    print("Device:", device)

    ckpt_path = Path(cfg["CKPT_PATH"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Build model with random init, then load trained weights
    model = BuildingSegModel(pretrained=False).to(device)

    # Important: weights_only=False to avoid PyTorch 2.6 safe-unpickling error
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"[Eval] Loaded checkpoint from: {ckpt_path}")

    for split in ["train", "test"]:
        print(f"\n[Eval] Split = {split}")

        ds = BuildingSegDataset(
            root=cfg["DATA_ROOT"],
            split=split,
            resize=cfg["RESIZE"],
            use_augmentation=False,
        )
        if len(ds) == 0:
            print(f"[Eval] No tiles found for split '{split}', skipping.")
            continue

        loader = DataLoader(
            ds,
            batch_size=cfg["BATCH_SIZE"],
            shuffle=False,
            num_workers=cfg["NUM_WORKERS"],
            pin_memory=False,
        )

        stats = eval_split(model, loader, device)
        print(
            f"[Eval:{split}] "
            f"acc: {stats['acc']:.4f}, "
            f"IoU: {stats['iou']:.4f}, "
            f"Dice: {stats['dice']:.4f}"
        )


if __name__ == "__main__":
    main()
