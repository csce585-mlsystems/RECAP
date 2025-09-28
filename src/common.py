"""
common.py
Purpose:
  Shared helpers & constants. All paths resolve from the project root, so you can
  move scripts inside `src/` (or elsewhere) without breaking anything.
"""

import os, random, json, platform
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch

# ----- project root ----- #
# project root = parent of this file's parent (i.e., repo/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ----- label palette ----- #
PALETTE = {
    0: (0, 0, 0),        # background
    1: (0, 200, 0),      # no damage
    2: (255, 215, 0),    # minor
    3: (255, 140, 0),    # major
    4: (220, 20, 60),    # destroyed
}
CLASS_NAMES = {0: "bg", 1: "no", 2: "minor", 3: "major", 4: "destroyed"}
N_CLASSES = 5

# ----- utils ----- #
def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def set_seed(s=123):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ts_filename() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def ts_human() -> str:
    return datetime.now().strftime("%m/%d/%Y %H:%M")

def save_metrics(dict_obj: Dict, out_root: str, run_name: str) -> Tuple[str, str]:
    ensure_dirs(out_root)
    j = Path(out_root) / f"{run_name}.json"
    t = Path(out_root) / f"{run_name}.txt"
    with open(j, "w") as f: json.dump(dict_obj, f, indent=2)
    with open(t, "w") as f:
        for k, v in dict_obj.items():
            if isinstance(v, (dict, list)): continue
            f.write(f"{k}: {v}\n")
    return str(j), str(t)

def append_run_csv(csv_path: str, row: Dict):
    p = Path(csv_path); p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists()
    import csv as _csv
    with open(p, "a", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)

def colorize_mask(mask_np: np.ndarray) -> np.ndarray:
    h, w = mask_np.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for k, rgb in PALETTE.items():
        out[mask_np == k] = rgb
    return out

def blend_rgba(img_rgb: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return (alpha * mask_rgb + (1 - alpha) * img_rgb).astype(np.uint8)

def default_num_workers() -> int:
    # Windows is safer with 0; elsewhere use ~half the cores
    if platform.system().lower().startswith("win"):
        return 0
    return max(2, (os.cpu_count() or 4) // 2)
