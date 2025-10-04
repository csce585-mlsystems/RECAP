# predict_all.py  (place under recap/etl/)
"""
Safe inference script for RECAP (Windows-friendly).

Usage (from project root):
    python -m recap.etl.predict_all

Or run the file directly (ensure repo root is on PYTHONPATH).
"""
import os
import sys
import math
from pathlib import Path
from tqdm import tqdm

# ensure repo root on sys.path when running file directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import models
from torch.utils.data import DataLoader

# local dataset
from recap.datasets.xview2_dataset import XView2Dataset

# --------------------------
# Config - tweak these
# --------------------------
INDEX_FILE = "info/index_with_chips.csv"   # expected to exist (or info/index.csv)
MODEL_PATH = "recap/models/checkpoint_epoch25.pth"   # or whichever checkpoint you want
OUT_CSV = "info/predictions.csv"
OUT_PQ = "info/predictions_with_confidence.parquet"

BATCH_SIZE = 32
NUM_WORKERS = 6     # set to 0 for debugging; increase for faster IO (but watch CPU)
CHIP_SIZE = 224
NUM_CLASSES = 4

# --------------------------
# Device detection (DirectML-aware)
# --------------------------
def get_device():
    """Return a torch.device for DirectML (if installed), else cuda/cpu."""
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
# Safe checkpoint loader
# --------------------------
def load_checkpoint_safe(model, path, device):
    """
    Load checkpoint to CPU first (safe for unusual devices like DirectML),
    then move model to device. Supports both raw state_dict or checkpoint dicts
    containing "model_state".
    """
    print("Loading checkpoint (CPU first):", path)
    data = torch.load(path, map_location="cpu")  # load to CPU first
    if isinstance(data, dict) and "model_state" in data:
        state = data["model_state"]
    else:
        state = data

    # Try direct load, otherwise strip 'module.' prefixes
    try:
        model.load_state_dict(state)
    except RuntimeError:
        new_state = {}
        for k, v in state.items():
            new_k = k.replace("module.", "") if k.startswith("module.") else k
            new_state[new_k] = v
        model.load_state_dict(new_state)

    model = model.to(device)
    model.eval()
    return model

# --------------------------
# Inference loop
# --------------------------
def run_inference(index_file=INDEX_FILE, model_path=MODEL_PATH, out_csv=OUT_CSV,
                  out_parquet=OUT_PQ, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):

    device = get_device()

    # Build model (6-channel ResNet18)
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    # load checkpoint safely and move model to device
    model = load_checkpoint_safe(model, model_path, device)

    # dataset & loader
    print("Loading index file:", index_file)
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Index file not found: {index_file}")

    dataset = XView2Dataset(index_file, chip_size=CHIP_SIZE, transform=None)  # dataset loads precomputed chips or paths depending on implementation
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type!="cpu"))

    # run inference
    all_preds = []
    all_confs = []

    print(f"Running inference: {len(dataset)} samples, batch_size={batch_size}, num_workers={num_workers}")
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Running inference", unit="batch"):
            imgs = imgs.to(device, non_blocking=True)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_confs.extend(confs.cpu().numpy().tolist())

    # sanity length checks
    if len(all_preds) != len(dataset):
        print("Warning: prediction count does not match dataset length.")
        print("len(preds)=", len(all_preds), "len(dataset)=", len(dataset))

    # Save predictions: map back to original index file ordering
    try:
        # try to get DataFrame from dataset if available
        if hasattr(dataset, "df"):
            df = dataset.df.reset_index(drop=True).copy()
        else:
            df = pd.read_csv(index_file)
    except Exception:
        df = pd.read_csv(index_file)

    # ensure matching length -- if mismatch, try to trim or pad
    n = len(df)
    preds_arr = np.array(all_preds, dtype=int)
    confs_arr = np.array(all_confs, dtype=float)

    if len(preds_arr) < n:
        # pad with -1 / 0.0 (shouldn't normally happen)
        pad_len = n - len(preds_arr)
        preds_arr = np.concatenate([preds_arr, np.full(pad_len, -1, dtype=int)])
        confs_arr = np.concatenate([confs_arr, np.zeros(pad_len, dtype=float)])
    elif len(preds_arr) > n:
        preds_arr = preds_arr[:n]
        confs_arr = confs_arr[:n]

    df["label_pred"] = preds_arr
    df["label_conf"] = confs_arr

    # map numeric preds to names if label_name column exists or use id->name map
    id2name = {0: "no-damage", 1: "minor-damage", 2: "major-damage", 3: "destroyed"}
    # if label_name column exists (ground truth), keep it; map preds to label_name_pred
    df["label_name_pred"] = df["label_pred"].map(id2name)

    # compute centroids if polygon_wkt present but centroid_x/centroid_y missing
    if "centroid_x" not in df.columns or "centroid_y" not in df.columns:
        if "polygon_wkt" in df.columns:
            try:
                from shapely import wkt
                print("Computing centroid_x / centroid_y from polygon_wkt (this may take a bit)...")
                df["centroid_x"] = df["polygon_wkt"].apply(lambda x: float(wkt.loads(x).centroid.x) if pd.notna(x) else np.nan)
                df["centroid_y"] = df["polygon_wkt"].apply(lambda x: float(wkt.loads(x).centroid.y) if pd.notna(x) else np.nan)
            except Exception as e:
                print("Centroid computation failed:", e)
                df["centroid_x"] = np.nan
                df["centroid_y"] = np.nan
        else:
            df["centroid_x"] = np.nan
            df["centroid_y"] = np.nan

    # persist CSV and Parquet
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    try:
        df.to_parquet(out_parquet, index=False, compression="snappy")
    except Exception as e:
        print("Parquet write failed (fine to ignore):", e)

    print(f"✅ Predictions saved to {out_csv} (and parquet at {out_parquet} if supported).")
    return out_csv

# --------------------------
# Entrypoint (Windows-safe)
# --------------------------
def main():
    run_inference()

if __name__ == "__main__":
    # Important on Windows when using DataLoader with workers
    try:
        import multiprocessing
        multiprocessing.freeze_support()
    except Exception:
        pass
    main()
