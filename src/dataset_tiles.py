"""
dataset_tiles.py
Purpose:
  Dataset for xBD / xView2 tiles that:
    - Reads pre and post images from:
          <DATA_ROOT>/<split>/images/
    - Reads building polygons + damage labels from:
          <DATA_ROOT>/<split>/labels/*.json
    - Returns:
        * pre_t:  Tensor (3,H,W)  – pre-disaster RGB
        * post_t: Tensor (3,H,W)  – post-disaster RGB
        * polys_xy: list of np.ndarray (N_i, 2) – building polygons in pixel coords
        * labels_t: LongTensor (#buildings,) – damage class indices {0..3}
        * tile_id: str – base name of the tile (without extension)
        * orig_w, orig_h: int – original image width/height from JSON

  It also exposes:
    self.tile_paths  – list[Path] of *_post_disaster.png
  so other scripts can subsample tiles (e.g., demo_random_tiles).
"""

from pathlib import Path
from typing import Optional, List, Tuple, Dict

import json
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from shapely import wkt as shapely_wkt

from .common import LABEL2IDX


def _parse_xy_polygon_wkt(wkt_str: str) -> np.ndarray:
    """
    Parse a WKT POLYGON string in image xy coordinates into
    an array of shape (N, 2) with columns [x, y].
    """
    wkt_str = wkt_str.strip()
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
    Tile-level dataset for xBD / xView2.

    Expected directory structure (for each split = 'train' or 'test'):

        <root>/
          train/
            images/*.png      # *_pre_disaster.png, *_post_disaster.png
            labels/*.json     # *_post_disaster.json
            targets/*.png     # (not used here)
          test/
            images/*.png
            labels/*.json
            targets/*.png

    We index tiles by their *post-disaster* image:
        <disaster>_<id>_post_disaster.png

    Matching files:
        pre image  : replace "_post_" with "_pre_"
        label JSON : same stem + ".json"
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        tile_paths: Optional[List[Path]] = None,
        resize: int = 512,
        limit_tiles: Optional[int] = None,
    ):
        """
        Args:
            root: path to "xBD Dataset" directory
            split: 'train' or 'test'
            tile_paths: optional list of specific *_post_disaster.png Paths
                        (if None, we glob all of them)
            resize: resize size for images (square)
            limit_tiles: optional cap on number of tiles (for debugging)
        """
        self.root = Path(root)
        self.split = split
        self.resize = resize

        split_dir = self.root / split          # e.g. data/xBD Dataset/train
        img_dir = split_dir / "images"         # you have 'images'
        label_dir = split_dir / "labels"       # and 'labels'
        # targets_dir = split_dir / "targets"  # not used here

        self.img_dir = img_dir
        self.label_dir = label_dir

        # --- collect tile paths (POST images) ---
        if tile_paths is None:
            # find all post-disaster images
            self.tile_paths: List[Path] = sorted(img_dir.glob("*_post_disaster.png"))
        else:
            self.tile_paths = tile_paths

        # optionally limit number of tiles
        if limit_tiles is not None:
            self.tile_paths = self.tile_paths[:limit_tiles]

        # simple transforms: resize + to tensor
        self.tfm = transforms.Compose([
            transforms.Resize(
                (resize, resize),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.tile_paths)

    def _load_images(
        self, post_path: Path
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Load pre and post images as tensors and return original width/height.

        Returns:
            pre_t, post_t: 3xH'xW' tensors
            orig_w, orig_h: original image size from disk (before resize)
        """
        tile_id = post_path.stem  # e.g. "guatemala-volcano_00000003_post_disaster"
        pre_name = tile_id.replace("_post_", "_pre_") + ".png"
        pre_path = self.img_dir / pre_name

        # open in RGB
        post_img = Image.open(post_path).convert("RGB")
        orig_w, orig_h = post_img.size

        if not pre_path.exists():
            raise FileNotFoundError(f"Missing pre image: {pre_path}")
        pre_img = Image.open(pre_path).convert("RGB")

        # apply same resize+tensor transform
        pre_t = self.tfm(pre_img)   # 3xH'xW'
        post_t = self.tfm(post_img)

        return pre_t, post_t, orig_w, orig_h

    def _load_polygons_and_labels(
        self, tile_id: str
    ) -> Tuple[List[np.ndarray], torch.LongTensor]:
        """
        Load building polygons + damage labels from the label JSON.

        Returns:
            polys_xy: list of np.ndarray (N_i, 2) for each building
            labels_t: LongTensor (#buildings,) with indices in {0..3}
        """
        lbl_path = self.label_dir / f"{tile_id}.json"
        if not lbl_path.exists():
            # Some tiles may be unlabeled; return empty
            return [], torch.zeros(0, dtype=torch.long)

        with lbl_path.open("r") as f:
            meta = json.load(f)

        polys_xy: List[np.ndarray] = []
        labels: List[int] = []

        feats = meta["features"]["xy"]
        for feat in feats:
            props = feat["properties"]
            if props.get("feature_type") != "building":
                continue
            subtype = props.get("subtype", "")
            if subtype not in LABEL2IDX:
                # skip any non-standard subtype
                continue
            lab_idx = LABEL2IDX[subtype]

            geom = shapely_wkt.loads(feat["wkt"])
            x, y = geom.exterior.coords.xy
            poly = np.stack([np.asarray(x), np.asarray(y)], axis=1)  # (N,2)

            polys_xy.append(poly)
            labels.append(lab_idx)

        if len(labels) == 0:
            return [], torch.zeros(0, dtype=torch.long)

        labels_t = torch.as_tensor(labels, dtype=torch.long)
        return polys_xy, labels_t

    def __getitem__(self, idx: int):
        """
        Returns:
            pre_t      : Tensor (3,H,W) – pre-disaster image
            post_t     : Tensor (3,H,W) – post-disaster image
            polys_xy   : list[np.ndarray] – each (N_i,2) polygon in original pixel coords
            labels_t   : LongTensor (#buildings,) – damage class indices
            tile_id    : str – stem of the post tile file
            orig_w     : int – original width
            orig_h     : int – original height
        """
        post_path = self.tile_paths[idx]
        tile_id = post_path.stem

        # images
        pre_t, post_t, orig_w, orig_h = self._load_images(post_path)

        # polygons + labels
        polys_xy, labels_t = self._load_polygons_and_labels(tile_id)

        return pre_t, post_t, polys_xy, labels_t, tile_id, orig_w, orig_h


class BuildingChangeDataset(Dataset):
    """
    Per-building dataset for training.

    Each sample corresponds to ONE building:
      pre : (3,H,W) tensor
      post: (3,H,W) tensor
      poly_xy: np.ndarray (N,2)
      label: torch.LongTensor scalar
      tile_id: str
      orig_w, orig_h: int

    This makes it easy to use a WeightedRandomSampler for class balancing.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        resize: int = 512,
        limit_tiles: Optional[int] = None,
        use_augmentation: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        split_lower = split.lower()
        if split_lower.startswith("train"):
            self.split = "train"
        else:
            self.split = "test"

        self.images_dir = self.root / self.split / "images"
        self.labels_dir = self.root / self.split / "labels"
        self.resize = resize
        self.use_augmentation = use_augmentation

        # Discover tiles
        all_post = sorted(self.images_dir.glob("*_post_disaster.png"))
        tile_ids = [p.stem for p in all_post]
        if limit_tiles is not None:
            tile_ids = tile_ids[:limit_tiles]

        # Build per-building index
        self.items: List[Dict] = []
        self.labels_list: List[int] = []

        for tile_id in tile_ids:
            json_path = self.labels_dir / f"{tile_id}.json"
            if not json_path.exists():
                continue

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
            for feat in feats:
                props = feat.get("properties", {})
                if props.get("feature_type") != "building":
                    continue
                subtype = props.get("subtype", "")
                if subtype not in LABEL2IDX:
                    continue
                cls_idx = LABEL2IDX[subtype]
                poly_wkt = feat.get("wkt", "")
                coords = _parse_xy_polygon_wkt(poly_wkt)
                if coords.shape[0] == 0:
                    continue

                self.items.append(
                    {
                        "tile_id": tile_id,
                        "poly_xy": coords,
                        "label": cls_idx,
                        "orig_w": orig_w,
                        "orig_h": orig_h,
                    }
                )
                self.labels_list.append(cls_idx)

        self.labels_tensor = torch.tensor(self.labels_list, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.items)

    def _load_pair_images(self, tile_id: str) -> Tuple[Image.Image, Image.Image]:
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

    def __getitem__(self, idx: int):
        item = self.items[idx]
        tile_id = item["tile_id"]
        poly_xy = item["poly_xy"]
        label = item["label"]
        orig_w = item["orig_w"]
        orig_h = item["orig_h"]

        pre_img, post_img = self._load_pair_images(tile_id)

        if self.resize is not None:
            pre_img = pre_img.resize((self.resize, self.resize), Image.BILINEAR)
            post_img = post_img.resize((self.resize, self.resize), Image.BILINEAR)

        pre = TF.to_tensor(pre_img)
        post = TF.to_tensor(post_img)

        # Photometric aug (no geometric changes -> polygons remain valid)
        if self.split == "train" and self.use_augmentation:
            if random.random() < 0.8:
                b = 0.8 + 0.4 * random.random()
                c = 0.8 + 0.4 * random.random()
                pre = TF.adjust_brightness(pre, b)
                post = TF.adjust_brightness(post, b)
                pre = TF.adjust_contrast(pre, c)
                post = TF.adjust_contrast(post, c)
            if random.random() < 0.5:
                noise = 0.01 * torch.randn_like(pre)
                pre = torch.clamp(pre + noise, 0.0, 1.0)
                noise = 0.01 * torch.randn_like(post)
                post = torch.clamp(post + noise, 0.0, 1.0)

        return (
            pre,                                # (3,H,W)
            post,                               # (3,H,W)
            poly_xy,                            # np.ndarray (N,2)
            torch.tensor(label, dtype=torch.long),
            tile_id,
            orig_w,
            orig_h,
        )
