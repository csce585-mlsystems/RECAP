from pathlib import Path
from typing import List, Optional, Tuple

import json
import random

import numpy as np
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class BuildingSegDataset(Dataset):
    """
    xBD post-disaster building segmentation dataset.

    For each tile:
      - loads *_post_disaster.png from split/images
      - loads matching .json from split/labels
      - rasterizes all building polygons into a binary mask
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        resize: int = 512,
        tile_ids: Optional[List[str]] = None,
        limit_tiles: Optional[int] = None,
        use_augmentation: bool = False,
    ) -> None:
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

        # Discover post-disaster tiles
        if tile_ids is None:
            all_post = sorted(self.images_dir.glob("*_post_disaster.png"))
            self.tile_ids: List[str] = [p.stem for p in all_post]
        else:
            self.tile_ids = list(tile_ids)

        if limit_tiles is not None:
            self.tile_ids = self.tile_ids[:limit_tiles]

    def __len__(self) -> int:
        return len(self.tile_ids)

    # ---- Helpers ----

    def _parse_xy_polygon_wkt(self, wkt_str: str) -> np.ndarray:
        """
        Parse a WKT POLYGON string in xy-image coordinates into (N,2) array.
        Example:
          "POLYGON ((x1 y1, x2 y2, ..., xN yN))"
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

    def _load_image(self, tile_id: str) -> Image.Image:
        path = self.images_dir / f"{tile_id}.png"
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        img = Image.open(path).convert("RGB")
        if self.resize is not None:
            img = img.resize((self.resize, self.resize), Image.BILINEAR)
        return img

    def _load_mask(self, tile_id: str, size: Tuple[int, int]) -> Image.Image:
        """
        Rasterize all building polygons in the label JSON into a binary mask.
        1 = building, 0 = background
        """
        W, H = size
        json_path = self.labels_dir / f"{tile_id}.json"
        mask = Image.new("L", (W, H), 0)  # start with all zeros

        if not json_path.exists():
            # No labels? return empty mask.
            return mask

        with json_path.open("r") as f:
            meta = json.load(f)

        metadata = meta.get("metadata", {})
        orig_w = int(
            metadata.get(
                "width",
                metadata.get("original_width", metadata.get("width", W)),
            )
        )
        orig_h = int(
            metadata.get(
                "height",
                metadata.get("original_height", metadata.get("height", H)),
            )
        )

        feats = meta.get("features", {}).get("xy", [])
        draw = ImageDraw.Draw(mask)

        for feat in feats:
            props = feat.get("properties", {})
            if props.get("feature_type") != "building":
                continue

            poly_wkt = feat.get("wkt", "")
            coords = self._parse_xy_polygon_wkt(poly_wkt)
            if coords.shape[0] == 0:
                continue

            # Scale polygon from original size → resized tile
            scale_x = W / float(orig_w)
            scale_y = H / float(orig_h)
            poly_scaled = np.stack(
                [coords[:, 0] * scale_x, coords[:, 1] * scale_y],
                axis=1,
            )
            # Fill polygon as 1
            draw.polygon(list(map(tuple, poly_scaled)), outline=1, fill=1)

        return mask

    def _apply_augmentation(self, img: Image.Image, mask: Image.Image):
        # Simple geometric augs; must be applied to both
        if random.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        if random.random() < 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        return img, mask

    def __getitem__(self, idx: int):
        tile_id = self.tile_ids[idx]

        # Load image and mask
        img = self._load_image(tile_id)
        W, H = img.size
        mask = self._load_mask(tile_id, (W, H))

        # Augmentation
        if self.use_augmentation:
            img, mask = self._apply_augmentation(img, mask)

        # To tensor + normalize with ImageNet stats
        img_t = TF.to_tensor(img)  # (3,H,W), 0-1
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std

        # Mask to {0,1} float tensor
        mask_np = (np.array(mask, dtype="float32") > 0.5).astype("float32")
        mask_t = torch.from_numpy(mask_np).unsqueeze(0)  # (1,H,W)

        return img_t, mask_t, tile_id
