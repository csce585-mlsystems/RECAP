"""
model_polygon_siamese.py

Purpose:
  - Define the siamese tile backbone (ResNet-34) and the per-building damage head.
  - Provide utilities for rasterizing polygon masks and pooling features.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class SiameseTileBackbone(nn.Module):
    """
    Shared ResNet-34 encoder used for both pre- and post-disaster tiles.

    Inputs we support:
      - (3, H, W)
      - (B, 3, H, W)
      - (1, 1, 3, H, W) or (B, 1, 3, H, W) from weird collate cases

    Output:
      - feature map: (B, C, H', W') where C = self.out_channels (512)
    """

    def __init__(
        self,
        imagenet_weights: bool = True,
        weight_path: str | None = None,
        local_weight_path: str | None = None,
    ) -> None:
        super().__init__()

        # unify the two possible arg names
        if local_weight_path is None and weight_path is not None:
            local_weight_path = weight_path

        if imagenet_weights:
            # Try local .pth first (for offline use), fall back to torchvision weights.
            if local_weight_path is not None and Path(local_weight_path).exists():
                base = models.resnet34(weights=None)
                state = torch.load(local_weight_path, map_location="cpu")
                base.load_state_dict(state)
                print(
                    f"[PolygonSiamese] Loaded local ResNet-34 weights from: "
                    f"{local_weight_path}"
                )
            else:
                # Will try to download if not cached
                base = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
                print(
                    "[PolygonSiamese] Using ResNet-34 with ImageNet pretrained "
                    "weights (torchvision)."
                )
        else:
            base = models.resnet34(weights=None)
            print(
                "[PolygonSiamese] Using ResNet-34 with random init "
                "(no pretrained weights)."
            )

        # Drop avgpool + fc, keep conv -> layer4
        self.encoder = nn.Sequential(*list(base.children())[:-2])
        self.out_channels = 512

    # ---------- shape normalizer so conv1 never explodes ----------
    def _normalize_shape(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make sure x ends up as (B, 3, H, W).

        Handles:
          - (3, H, W)          -> (1, 3, H, W)
          - (1, 3, H, W)       -> (1, 3, H, W)
          - (B, 3, H, W)       -> (B, 3, H, W)
          - (1, 1, 3, H, W)    -> (1, 3, H, W)
          - (B, 1, 3, H, W)    -> (B, 3, H, W)
        """
        if x.dim() == 5:
            # assume (B, 1, 3, H, W) or (1, 1, 3, H, W)
            if x.size(1) == 1:
                # drop that dummy middle dim
                x = x.squeeze(1)  # (B, 3, H, W)
            else:
                # odd case, fallback to drop first dim
                x = x.squeeze(0)

        if x.dim() == 3:
            # (3, H, W) -> (1, 3, H, W)
            x = x.unsqueeze(0)

        # now we expect (B, 3, H, W)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize_shape(x)
        return self.encoder(x)  # (B, 512, H', W')


class DamageHead(nn.Module):
    """
    Deeper MLP that maps per-building feature vectors to 4 damage classes.

    This is backward compatible with the training script calling:
        head = DamageHead(in_dim=backbone.out_channels, n_classes=4)

    but internally we use a small 3-layer MLP with dropout.
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # 3-layer MLP: in_dim -> hidden -> hidden -> n_classes
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (#buildings, in_dim)
        return self.mlp(x)


def rasterize_polygon_mask(
    poly_xy: np.ndarray,
    Hf: int,
    Wf: int,
    orig_w: int,
    orig_h: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert a polygon (in original image pixel coords) into a binary mask at
    feature-map resolution.

    poly_xy : (N, 2) array in original coords (x, y)
    Returns:
      mask: (Hf, Wf) float tensor in {0,1}
    """
    if poly_xy.size == 0:
        return torch.zeros((Hf, Wf), dtype=torch.float32, device=device)

    poly = poly_xy.astype(float)

    # scale from original size -> feature map size
    scale_x = Wf / float(orig_w)
    scale_y = Hf / float(orig_h)
    poly_scaled = np.stack(
        [poly[:, 0] * scale_x, poly[:, 1] * scale_y],
        axis=1,
    )

    # Draw into a PIL image
    mask_img = Image.new("L", (Wf, Hf), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.polygon(list(map(tuple, poly_scaled)), outline=1, fill=1)
    mask_np = np.array(mask_img, dtype="float32")

    mask = torch.from_numpy(mask_np).to(device)  # (Hf, Wf)
    return mask


def mask_pool_features(
    feat_map: torch.Tensor,  # (C, Hf, Wf)
    mask: torch.Tensor,      # (Hf, Wf)
) -> torch.Tensor:
    """
    Average-pool features inside a binary mask.

    Returns:
      v: (C,) pooled feature vector.
    """
    C, Hf, Wf = feat_map.shape
    m = mask.view(1, Hf, Wf)  # (1, Hf, Wf)
    denom = m.sum() + 1e-6
    v = (feat_map * m).sum(dim=(1, 2)) / denom
    return v
