# recap/datasets/xview2_dataset.py (replace __getitem__ with this)
import os
import torch
import pandas as pd
import cv2
import numpy as np
from shapely import wkt
from shapely.geometry import Polygon
from torch.utils.data import Dataset

CHIPS_DIR = "data/chips"   # precomputed chips folder

class XView2Dataset(Dataset):
    def __init__(self, index_file, transform=None, chip_size=224):
        self.df = pd.read_csv(index_file)
        self.transform = transform
        self.chip_size = chip_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        building_id = str(row["building_id"])
        chip_path = os.path.join(CHIPS_DIR, f"{building_id}.npy")

        chip = None
        if os.path.exists(chip_path):
            try:
                chip = np.load(chip_path)  # HxWx6 uint8
            except Exception:
                chip = None

        if chip is None:
            # fallback to original PNG crop logic
            pre_img = cv2.imread(row["pre_path"], cv2.IMREAD_COLOR)
            post_img = cv2.imread(row["post_path"], cv2.IMREAD_COLOR)
            if pre_img is None or post_img is None:
                raise FileNotFoundError(f"Missing image for building {building_id}")
            h, w, _ = pre_img.shape
            poly = wkt.loads(row["polygon_wkt"])
            cx, cy = map(int, poly.centroid.coords[0]) if isinstance(poly, Polygon) else (w//2, h//2)
            half = self.chip_size // 2
            x1, x2 = max(0, cx - half), min(w, cx + half)
            y1, y2 = max(0, cy - half), min(h, cy + half)
            pre_chip = pre_img[y1:y2, x1:x2]
            post_chip = post_img[y1:y2, x1:x2]
            pre_chip = cv2.resize(pre_chip, (self.chip_size, self.chip_size))
            post_chip = cv2.resize(post_chip, (self.chip_size, self.chip_size))
            chip = np.concatenate([pre_chip, post_chip], axis=2)

        # Normalise and convert to tensor [C,H,W]
        combined = torch.from_numpy(chip).permute(2, 0, 1).float() / 255.0

        if self.transform:
            try:
                combined = self.transform(combined)
            except Exception:
                # If transforms expect PIL, they may fail; handle accordingly later if needed.
                pass

        label = torch.tensor(row["label_id"]).long()
        return combined, label
