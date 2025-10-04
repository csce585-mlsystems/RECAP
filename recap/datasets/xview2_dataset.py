# recap/datasets/xview2_dataset.py
import os
import torch
import pandas as pd
import cv2
import numpy as np
from shapely import wkt
from shapely.geometry import Polygon
from torch.utils.data import Dataset

class XView2Dataset(Dataset):
    """
    Dataset that prefers combined chips saved as .npy under data/chips/{building_id}.npy
    Combined chip shape expected: H x W x 6 (pre channels then post channels).
    Falls back to cropping pre_path/post_path if chip not present.
    Returns: (tensor[6,H,W], label)
    """
    def __init__(self, index_file, transform=None, chip_size=224):
        self.df = pd.read_csv(index_file)
        self.transform = transform
        self.chip_size = int(chip_size)

    def __len__(self):
        return len(self.df)

    def _load_npy_chip(self, path):
        try:
            arr = np.load(path)
            # ensure H,W,6
            if arr.ndim == 3 and arr.shape[2] == 6:
                return arr.astype(np.uint8)
            # sometimes saved as (6,H,W)
            if arr.ndim == 3 and arr.shape[0] == 6:
                arr = np.transpose(arr, (1,2,0))
                return arr.astype(np.uint8)
        except Exception:
            return None
        return None

    def _crop_from_full(self, img, cx, cy):
        h, w = img.shape[:2]
        half = self.chip_size // 2
        x1, x2 = max(0, cx - half), min(w, cx + half)
        y1, y2 = max(0, cy - half), min(h, cy + half)
        chip = img[y1:y2, x1:x2]
        if chip is None or chip.size == 0:
            chip = cv2.resize(img, (self.chip_size, self.chip_size))
        else:
            chip = cv2.resize(chip, (self.chip_size, self.chip_size))
        return chip

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1) Try combined chip
        chip = None
        if "chip_path" in row and isinstance(row["chip_path"], str) and row["chip_path"].strip():
            p = row["chip_path"]
            if os.path.exists(p):
                chip = self._load_npy_chip(p)

        # 2) If chip missing, load pre/post full images and crop
        if chip is None:
            pre_img = None
            post_img = None
            if "pre_path" in row and isinstance(row["pre_path"], str) and row["pre_path"].strip():
                pre_img = cv2.imread(row["pre_path"], cv2.IMREAD_COLOR)
            if "post_path" in row and isinstance(row["post_path"], str) and row["post_path"].strip():
                post_img = cv2.imread(row["post_path"], cv2.IMREAD_COLOR)

            # compute centroid fallback
            cx = cy = None
            if "polygon_wkt" in row and isinstance(row["polygon_wkt"], str) and row["polygon_wkt"].strip():
                try:
                    poly = wkt.loads(row["polygon_wkt"])
                    if isinstance(poly, Polygon):
                        cen = poly.centroid
                        cx, cy = int(cen.x), int(cen.y)
                except Exception:
                    cx = cy = None

            # fallback to image center
            if cx is None or cy is None:
                if pre_img is not None:
                    h,w = pre_img.shape[:2]
                    cx, cy = w//2, h//2
                elif post_img is not None:
                    h,w = post_img.shape[:2]
                    cx, cy = w//2, h//2
                else:
                    raise FileNotFoundError(f"No chip and no pre/post images for idx {idx}")

            if pre_img is None:
                raise FileNotFoundError(f"Missing pre image for idx {idx}")
            if post_img is None:
                raise FileNotFoundError(f"Missing post image for idx {idx}")

            pre_chip = self._crop_from_full(pre_img, cx, cy)
            post_chip = self._crop_from_full(post_img, cx, cy)
            chip = np.concatenate([pre_chip, post_chip], axis=2)  # H,W,6

        # convert to tensor C,H,W float32 [0,1]
        arr = torch.from_numpy(chip).permute(2,0,1).float().contiguous() / 255.0

        if self.transform:
            arr = self.transform(arr)

        label = torch.tensor(int(row["label_id"]), dtype=torch.long)
        return arr, label
