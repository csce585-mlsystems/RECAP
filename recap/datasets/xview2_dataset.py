import torch
import pandas as pd
import cv2
import numpy as np
from shapely import wkt
from shapely.geometry import Polygon
from torch.utils.data import Dataset
from torchvision import transforms

class XView2Dataset(Dataset):
    """
    PyTorch Dataset for xView2 Challenge (building-level damage classification).
    Loads pre/post PNGs directly and crops around building centroid.
    """

    def __init__(self, index_file, transform=None, chip_size=224):
        self.df = pd.read_csv(index_file)
        self.transform = transform
        self.chip_size = chip_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load pre/post images
        pre_img = cv2.imread(row["pre_path"], cv2.IMREAD_COLOR)
        post_img = cv2.imread(row["post_path"], cv2.IMREAD_COLOR)

        # Fallback if image is missing
        if pre_img is None or post_img is None:
            raise FileNotFoundError(f"Missing image for building {row['building_id']}")

        h, w, _ = pre_img.shape

        # Get centroid of building polygon
        poly = wkt.loads(row["polygon_wkt"])
        if not isinstance(poly, Polygon):
            raise ValueError(f"Invalid geometry for building {row['building_id']}")
        cx, cy = map(int, poly.centroid.coords[0])

        # Crop function
        def crop(img):
            half = self.chip_size // 2
            x1, x2 = max(0, cx - half), min(w, cx + half)
            y1, y2 = max(0, cy - half), min(h, cy + half)
            chip = img[y1:y2, x1:x2]
            chip_resized = cv2.resize(chip, (self.chip_size, self.chip_size))
            return chip_resized

        pre_chip = crop(pre_img)
        post_chip = crop(post_img)

        # Stack pre+post → shape (224, 224, 6)
        combined = np.concatenate([pre_chip, post_chip], axis=2)

        # Convert to tensor [C,H,W]
        combined = torch.from_numpy(combined).permute(2, 0, 1).float()

        # Normalize to [0,1]
        combined = combined / 255.0

        if self.transform:
            combined = self.transform(combined)

        label = torch.tensor(row["label_id"]).long()

        return combined, label
