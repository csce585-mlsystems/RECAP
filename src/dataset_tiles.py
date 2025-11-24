"""
dataset_tiles.py
Purpose:
  Wrap the xBD "train"/"test" folders into a PyTorch Dataset that:

    - Finds *_post_disaster.png tiles under:
        <root>/<split>/images/
    - For each post tile, finds:
        - matching pre tile by name (replace "post_disaster" → "pre_disaster")
        - JSON label file under <root>/<split>/labels/<tile_id>.json
    - Parses building polygons from the JSON ("features" → "xy").
    - Maps damage subtype ("no-damage", "minor-damage", etc.) to label indices 0..3.
    - Returns:
        pre_tensor:  (1,3,H,W)
        post_tensor: (1,3,H,W)
        polys_xy:    list[np.ndarray] of shape (N_i,2) in original-pixel coords
        labels:      torch.LongTensor (#buildings,)
        tile_id:     str
        orig_w:      int (original image width, e.g., 1024)
        orig_h:      int (original image height)

  We also apply simple photometric augmentation (brightness/contrast + noise)
  on the training split to improve robustness, but keep geometry unchanged.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import json
import random

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from .common import LABEL2IDX


def _parse_xy_polygon_wkt(wkt_str: str) -> np.ndarray:
    """
    Parse a WKT POLYGON string in image xy coordinates into
    an array of shape (N, 2) with columns [x, y].

    Example WKT:
      "POLYGON ((x1 y1, x2 y2, ..., xN yN))"
    """
    wkt_str = wkt_str.strip()
    # remove "POLYGON ((" prefix and "))" suffix
    if wkt_str.upper().startswith("POLYGON"):
        inner = wkt_str[wkt_str.find("((") + 2 : wkt_str.rfind("))")]
    else:
        inner = wkt_str
    pts_str = inner.split(",")
    coords = []
    for pt in pts_str:
        parts = pt.strip().split()
        if len(parts) != 2:
            continue
        x, y = float(parts[0]), float(parts[1])
        coords.append([x, y])
    if not coords:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array(coords, dtype=np.float32)


class TileBuildingDataset(Dataset):
    """
    TileBuildingDataset

    Works directly on the xBD directory layout:
      root/
        train/
          images/
          labels/
          targets/
        test/
          images/
          labels/
          targets/

    It treats each *_post_disaster.png as one tile and finds:
      - the matching *_pre_disaster.png
      - the corresponding JSON label file.

    __getitem__ returns:
      pre_tensor  : torch.FloatTensor (1,3,H,W)
      post_tensor : torch.FloatTensor (1,3,H,W)
      polys_xy    : list[np.ndarray] of original xy coords
      labels      : torch.LongTensor (#buildings,)
      tile_id     : str (basename without .png/.json)
      orig_w      : int
      orig_h      : int
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        resize: int = 512,
        tile_ids: Optional[List[str]] = None,
        limit_tiles: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        # normalize split name: "train", "Train", "TRAIN" → "train"
        split_lower = split.lower()
        if split_lower.startswith("train"):
            self.split = "train"
        else:
            self.split = "test"

        self.images_dir = self.root / self.split / "images"
        self.labels_dir = self.root / self.split / "labels"
        self.resize = resize

        # Discover all *_post_disaster.png as tiles if tile_ids not provided
        if tile_ids is None:
            all_post = sorted(self.images_dir.glob("*_post_disaster.png"))
            self.tile_ids: List[str] = [p.stem for p in all_post]
        else:
            # tile_ids are base names like "guatemala-volcano_00000003_post_disaster"
            self.tile_ids = list(tile_ids)

        if limit_tiles is not None:
            self.tile_ids = self.tile_ids[: limit_tiles]

    def __len__(self) -> int:
        return len(self.tile_ids)

    def _load_pair_images(self, tile_id: str) -> Tuple[Image.Image, Image.Image]:
        """
        Given a post-disaster tile_id like:
          'guatemala-volcano_00000003_post_disaster'
        we:
          - load post from images_dir / (tile_id + ".png")
          - build pre_id by replacing 'post_disaster' with 'pre_disaster'
        """
        post_path = self.images_dir / f"{tile_id}.png"
        if not post_path.exists():
            raise FileNotFoundError(f"Post image not found: {post_path}")

        if "post_disaster" not in tile_id:
            raise ValueError(f"tile_id does not look like a post tile: {tile_id}")

        pre_id = tile_id.replace("post_disaster", "pre_disaster")
        pre_path = self.images_dir / f"{pre_id}.png"
        if not pre_path.exists():
            raise FileNotFoundError(f"Pre image not found for {tile_id}: {pre_path}")

        post_img = Image.open(post_path).convert("RGB")
        pre_img = Image.open(pre_path).convert("RGB")
        return pre_img, post_img

    def _load_polygons_and_labels(
        self, tile_id: str
    ) -> Tuple[List[np.ndarray], torch.Tensor, int, int]:
        """
        Read JSON label file and extract:
          - polys_xy: list of np.ndarray (N_i, 2)
          - labels: torch.LongTensor (#buildings,)
          - orig_w, orig_h: original image width/height (int)
        """
        json_path = self.labels_dir / f"{tile_id}.json"
        if not json_path.exists():
            # No labels: return empty
            return [], torch.zeros(0, dtype=torch.long), 1024, 1024

        with json_path.open("r") as f:
            meta = json.load(f)

        metadata = meta.get("metadata", {})
        orig_w = int(
            metadata.get(
                "width",
                metadata.get("original_width", metadata.get("width", 1024)),
            )
        )
        orig_h = int(
            metadata.get(
                "height",
                metadata.get("original_height", metadata.get("height", 1024)),
            )
        )

        feats = meta.get("features", {}).get("xy", [])
        polys_xy: List[np.ndarray] = []
        labels_list: List[int] = []

        for feat in feats:
            props = feat.get("properties", {})
            if props.get("feature_type") != "building":
                continue
            subtype = props.get("subtype", "")
            if subtype not in LABEL2IDX:
                # ignore unclassified/other
                continue
            cls_idx = LABEL2IDX[subtype]
            poly_wkt = feat.get("wkt", "")
            coords = _parse_xy_polygon_wkt(poly_wkt)
            if coords.shape[0] == 0:
                continue
            polys_xy.append(coords)
            labels_list.append(cls_idx)

        if not polys_xy:
            labels_tensor = torch.zeros(0, dtype=torch.long)
        else:
            labels_tensor = torch.tensor(labels_list, dtype=torch.long)

        return polys_xy, labels_tensor, orig_w, orig_h

    def __getitem__(self, idx: int):
        tile_id = self.tile_ids[idx]

        # 1) Load pre/post images
        pre_img, post_img = self._load_pair_images(tile_id)

        # 2) Optional resize (same for both)
        if self.resize is not None:
            pre_img = pre_img.resize((self.resize, self.resize), Image.BILINEAR)
            post_img = post_img.resize((self.resize, self.resize), Image.BILINEAR)

        # 3) Convert to tensors in [0,1], shape (3,H,W)
        pre = TF.to_tensor(pre_img)
        post = TF.to_tensor(post_img)

        # 4) Simple photometric augmentation on training split
        if self.split == "train":
            # brightness + contrast jitter
            if random.random() < 0.8:
                b = 0.8 + 0.4 * random.random()  # [0.8, 1.2]
                c = 0.8 + 0.4 * random.random()
                pre = TF.adjust_brightness(pre, b)
                post = TF.adjust_brightness(post, b)
                pre = TF.adjust_contrast(pre, c)
                post = TF.adjust_contrast(post, c)
            # small Gaussian noise
            if random.random() < 0.5:
                noise = 0.01 * torch.randn_like(pre)
                pre = torch.clamp(pre + noise, 0.0, 1.0)
                noise = 0.01 * torch.randn_like(post)
                post = torch.clamp(post + noise, 0.0, 1.0)

        # 5) Add batch-dim here so backbone sees (B=1,3,H,W)
        pre = pre.unsqueeze(0)   # (1,3,H,W)
        post = post.unsqueeze(0) # (1,3,H,W)

        # 6) Load polygons, labels and original size
        polys_xy, labels, orig_w, orig_h = self._load_polygons_and_labels(tile_id)

        return pre, post, polys_xy, labels, tile_id, orig_w, orig_h
