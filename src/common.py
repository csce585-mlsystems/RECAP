"""
common.py
Purpose:
  Shared helpers: project root detection, device selection, seeding,
  color mapping, simple overlay utilities, and a small logger.
"""

from pathlib import Path
import random
import numpy as np
from PIL import Image

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Map damage class index -> text + color
IDX2LABEL = {
    0: "no-damage",
    1: "minor-damage",
    2: "major-damage",
    3: "destroyed",
}

LABEL2IDX = {v: k for k, v in IDX2LABEL.items()}

# Simple RGBA colors for overlays (r,g,b,a)
LABEL2COLOR = {
    "no-damage":     (128, 128, 128, 90),
    "minor-damage":  (255, 255, 0,   90),
    "major-damage":  (255, 140, 0,   90),
    "destroyed":     (255, 0,   0,   90),
}


def to_numpy_image(t: torch.Tensor) -> np.ndarray:
    """
    t: (3,H,W) in [0,1]
    -> uint8 (H,W,3)
    """
    arr = (t.detach().cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def save_overlay_polygon(
    base_rgb: np.ndarray,
    polygons_xy,
    labels,
    save_path: Path,
):
    """
    Draw semi-transparent colored polygons on top of the RGB base image.
    polygons_xy: list of np.ndarray of shape (N_i,2) in pixel coords (x,y)
    labels: list of str damage label ("no-damage", etc.)
    """
    H, W, _ = base_rgb.shape
    base_img = Image.fromarray(base_rgb)
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = Image.core.draw(overlay.im)  # faster than ImageDraw.Draw for many polys

    for poly, lab in zip(polygons_xy, labels):
        color = LABEL2COLOR.get(lab, (0, 255, 255, 90))
        # PIL draw expects a flat list of coordinates: [x0,y0,x1,y1,...]
        coords = [float(x) for xy in poly for x in xy]
        draw.polygon(coords, fill=color)

    blended = Image.alpha_composite(base_img.convert("RGBA"), overlay)
    blended.convert("RGB").save(save_path)
