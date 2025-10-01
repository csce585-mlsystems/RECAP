import os
import glob
import csv
import time
from collections import Counter
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

from recap.datasets.xview2_dataset import XView2Dataset

# --------------------------
# Config - tweak these
# --------------------------
INDEX_FILE = "info/index.csv"
CHIP_SIZE = 224

BATCH_SIZE = 32
NUM_WORKERS = 4
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

EPOCHS = 30
LR = 1e-4
VAL_SPLIT = 0.2
NUM_CLASSES = 4

USE_SAMPLER = True     # set True to enable WeightedRandomSampler (oversample minorities)
USE_FOCAL = False      # set True to use FocalLoss instead of CrossEntropy
BACKBONE = "resnet18"  # "resnet18" or "resnet34"

CHECKPOINT_DIR = "recap/models"
LOG_CSV = "info/training_log.csv"
PRED_CSV_DIR = "info/predictions_by_epoch"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("info", exist_ok=True)
os.makedirs(PRED_CSV_DIR, exist_ok=True)

# --------------------------
# Device detection (DirectML-aware)
# --------------------------
def get_device():
    try:
        import torch_directml
        dml = torch_directml.device()
        device = torch.device(dml)
        print("Using DirectML device:", dml)
        return device
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        return device

# --------------------------
# Focal loss (optional)
# --------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean() if self.reduction == 'mean' else loss.sum()

# --------------------------
# Helper: load latest checkpoint
# --------------------------
def latest_checkpoint(path=CHECKPOINT_DIR):
    files = glob.glob(os.path.join(path, "checkpoint_epoch*.pth"))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

# --------------------------
# Main
# --------------------------
def main():
    device = get_device()

    # transforms (tensor-friendly). You can expand if needed.
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomErasing(p=0.25, scale=(0.02,0.2)),
    ])
    val_transform = transforms.Compose([])

    print("Loading dataset index:", INDEX_FILE)
    df = pd.read_csv(INDEX_FILE)

    # dataset
    dataset = XView2Dataset(INDEX_FILE, transform=train_transform, chip_size=CHIP_SIZE)

    # split
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    # ensure underlying dataset transforms set
    train_set.dataset.transform = train_transform
    val_set.dataset.transform = val_transform

    # --------------------------
    # Build WeightedRandomSampler for the train subset (if requested)
    # Important: sampler must align with the Subset indices (train_set.indices)
    # --------------------------
    sampler = None
    if USE_SAMPLER:
        print("Building WeightedRandomSampler (oversampling minority classes) for the train subset...")

        # get train indices that map back to original dataset / df
        # torch.utils.data.Subset has attribute 'indices' (a list of original indices)
        if hasattr(train_set, "indices"):
            train_indices = train_set.indices
        else:
            # fallback: try to detect via attribute names used by different torch versions
            train_indices = getattr(train_set, "subset_indices", None) or getattr(train_set, "dataset_indices", None)
            if train_indices is None:
                # last-resort: assume first train_size rows (shouldn't happen in normal random_split)
                train_indices = list(range(train_size))
                print("Warning: couldn't find train_set.indices; falling back to first N rows")

        # compute class weights over full df (simple inverse frequency)
        label_counts = Counter(df["label_id"].values)
        total = sum(label_counts.values())
        class_weights = {cls: total/count for cls, count in label_counts.items()}

        # build sample_weights aligned with train_indices order
        sample_weights_train = []
        for orig_idx in train_indices:
            lbl = int(df.iloc[orig_idx]["label_id"])
            sample_weights_train.append(class_weights[lbl])

        # create sampler on train subset
        sampler = WeightedRandomSampler(weights=sample_weights_train,
                                        num_samples=len(sample_weights_train),
                                        replacement=True)
        print(f"Sampler built. Train subset size: {len(train_indices)}. Sampler samples: {len(sample_weights_train)}")

    # dataloaders
    if sampler:
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler,
                                  num_workers=NUM_WORKERS, pin_memory=(device.type!="cpu"),
                                  persistent_workers=PERSISTENT_WORKERS, prefetch_factor=PREFETCH_FACTOR)
    else:
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=(device.type!="cpu"),
                                  persistent_workers=PERSISTENT_WORKERS, prefetch_factor=PREFETCH_FACTOR)

    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=max(2, NUM_WORKERS//2), pin_memory=(device.type!="cpu"),
                            persistent_workers=PERSISTENT_WORKERS, prefetch_factor=max(1, PREFETCH_FACTOR//2))

    # model
    if BACKBONE == "resnet34":
        model = models.resnet34(weights=None)
    else:
        model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)
    print("Model device:", next(model.parameters()).device)

    # optimizer + scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # class weights for criterion (kept on device)
    label_counts = Counter(df["label_id"].values)
    total = sum(label_counts.values())
    weights = [total / label_counts[i] for i in range(NUM_CLASSES)]
    weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
    print("Class counts:", label_counts)
    print("Class weights:", weights_tensor)

    # criterion
    if USE_FOCAL:
        criterion = FocalLoss(gamma=2.0, weight=weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # resume if checkpoint exists
    ckpt = latest_checkpoint()
    start_epoch = 0
    if ckpt:
        print("Found checkpoint, loading:", ckpt)
        sd = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(sd["model_state"])
        optimizer.load_state_dict(sd["opt_state"])
        if "sched_state" in sd and sd["sched_state"] is not None:
            scheduler.load_state_dict(sd["sched_state"])
        start_epoch = sd.get("epoch", 0)
        print("Resuming from epoch", start_epoch)

    # logging CSV (overwrite each run)
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_acc", "val_macro_f1"])

    # training loop
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        total = 0
        correct = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            pbar.set_postfix({"batch_loss": loss.item()})

        train_acc = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = train_loss / total if total > 0 else 0.0
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # validation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(imgs)
                _, preds = outputs.max(1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        # compute metrics (use sklearn if available)
        try:
            from sklearn.metrics import classification_report, confusion_matrix, f1_score
            val_acc = 100.0 * sum([yt==yp for yt,yp in zip(y_true,y_pred)]) / len(y_true) if len(y_true)>0 else 0.0
            val_macro_f1 = f1_score(y_true, y_pred, average="macro") if len(y_true)>0 else 0.0
            print(classification_report(y_true, y_pred, target_names=["no","minor","major","destroyed"]))
            print(confusion_matrix(y_true, y_pred))
        except Exception as e:
            print("sklearn not available or metric failed:", e)
            val_acc = 0.0
            val_macro_f1 = 0.0

        # save predictions for analysis
        try:
            # map val_set indices back to original df rows and save
            if hasattr(val_set, "indices"):
                val_orig_indices = val_set.indices
                val_df = pd.read_csv(INDEX_FILE).iloc[val_orig_indices].reset_index(drop=True)
            else:
                # fallback
                val_df = pd.read_csv(INDEX_FILE).iloc[train_size:train_size+len(y_true)].reset_index(drop=True)
            preds_df = val_df.copy()
            preds_df["y_true"] = y_true
            preds_df["y_pred"] = y_pred
            preds_df.to_csv(os.path.join(PRED_CSV_DIR, f"preds_epoch{epoch+1}.csv"), index=False)
        except Exception:
            pass

        # log
        with open(LOG_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_loss, train_acc, val_acc, val_macro_f1])

        # checkpoint
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch{epoch+1}.pth")
        torch.save({
            "epoch": epoch+1,
            "model_state": model.state_dict(),
            "opt_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict() if scheduler is not None else None
        }, ckpt_path)
        print("Saved checkpoint:", ckpt_path)

        # step scheduler
        scheduler.step()

    # final save (also saved as last checkpoint)
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "baseline_resnet_final.pth"))
    print("✅ Training complete. Model and log saved.")

if __name__ == "__main__":
    # safe multiprocess on Windows
    try:
        import multiprocessing
        multiprocessing.freeze_support()
    except Exception:
        pass
    main()
