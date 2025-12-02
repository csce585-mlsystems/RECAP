"""
common.py
Purpose:
  Shared helpers: project root detection, device selection, seeding,
  color mapping, simple overlay utilities, and a small logger.
"""

import random
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image, ImageDraw

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


#def get_device(prefer_gpu: bool = True) -> torch.device: # Default get device with someone with no GPU or has a Nvidia GPU
 #   if prefer_gpu and torch.cuda.is_available():
  #      return torch.device("cuda")
   # return torch.device("cpu")

def get_device(prefer_gpu: bool = True) -> torch.device: # For Yatin as he has a AMD GPU and not Nvidia
    if prefer_gpu:
        try:
            import torch_directml
            return torch_directml.device()
        except ImportError:
            pass
        if torch.cuda.is_available():
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


def to_numpy_image(t: "torch.Tensor") -> np.ndarray:
    """
    Convert a CHW or HWC float tensor in [0,1] to uint8 HxWx3 numpy array.
    """
    import torch

    if isinstance(t, torch.Tensor):
        arr = t.detach().cpu().numpy()
    else:
        arr = np.asarray(t)

    # if CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)
    elif arr.ndim != 3:
        raise ValueError(f"Expected 3D tensor/image, got shape {arr.shape}")

    # if single-channel, repeat to RGB
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def save_overlay_polygon(
    base_rgb: np.ndarray,
    polys_xy: List[np.ndarray],
    labels: List[str],
    out_path: Path,
) -> None:
    """
    Draw damage polygons on top of a base RGB image and save to disk.

    Args:
        base_rgb : HxWx3 uint8 numpy array (e.g., from to_numpy_image(post_t))
        polys_xy : list of (N_i, 2) arrays in pixel coords
        labels   : list of damage labels ('no-damage', 'minor-damage', etc.)
        out_path : where to save the PNG
    """
    assert len(polys_xy) == len(labels), "polygons and labels must have same length"

    # ensure uint8 RGB
    if base_rgb.dtype != np.uint8:
        base_rgb = base_rgb.astype(np.uint8)
    if base_rgb.ndim != 3 or base_rgb.shape[2] != 3:
        raise ValueError(f"Expected base_rgb as HxWx3, got shape {base_rgb.shape}")

    # base image + transparent overlay
    base_img = Image.fromarray(base_rgb, mode="RGB")
    overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    # simple color map for the 4 damage classes
    color_map = {
        "no-damage": (0, 255, 0, 80),        # green, transparent
        "minor-damage": (255, 255, 0, 80),   # yellow
        "major-damage": (255, 165, 0, 80),   # orange
        "destroyed": (255, 0, 0, 80),        # red
    }

    for poly, lab in zip(polys_xy, labels):
        if poly is None or len(poly) == 0:
            continue
        coords = [(float(x), float(y)) for x, y in poly]
        color = color_map.get(lab, (255, 255, 255, 80))  # default: white
        # filled polygon + white outline for visibility
        draw.polygon(coords, fill=color, outline=(255, 255, 255, 128))

    # alpha-composite overlay on top of base image
    out_img = Image.alpha_composite(base_img.convert("RGBA"), overlay)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)