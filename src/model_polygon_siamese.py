"""
model_polygon_siamese.py

Purpose:
  - Define the siamese tile backbone (ResNet-18) and the per-building damage head.
  - Provide utilities for rasterizing polygon masks and pooling features.
"""

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class SiameseTileBackbone(nn.Module):
    """
    Shared ResNet-18 encoder used for both pre- and post-disaster tiles.

    Input:
      x: (3,H,W) or (B,3,H,W) or (1,1,3,H,W) in some buggy cases.
    Output:
      feature map: (B,C,H',W')
    """

    def __init__(self, imagenet_weights: bool = True):
        super().__init__()
        if imagenet_weights:
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            print("[PolygonSiamese] Using ResNet-18 with ImageNet pretrained weights.")
        else:
            base = models.resnet18(weights=None)
            print("[PolygonSiamese] Using ResNet-18 with random init (no pretrained weights).")

        # Remove avgpool + fc, keep conv → layer4
        layers = list(base.children())[:-2]
        self.encoder = nn.Sequential(*layers)
        self.out_channels = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fix shapes like (1,1,3,H,W) → (1,3,H,W)
        if x.dim() == 5:
            # Most common case from buggy collate: (1,1,3,H,W)
            # squeeze first dim → (1,3,H,W)
            x = x.squeeze(0)

        # If single image (3,H,W), add batch dim
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Now x must be (B,3,H,W)
        return self.encoder(x)


class DamageHead(nn.Module):
    """
    Simple MLP that maps per-building feature vectors to 4 damage classes.

    IMPORTANT:
      - Submodule is called 'net' so keys look like:
          'net.0.weight', 'net.0.bias', 'net.2.weight', 'net.2.bias'
      - Architecture matches the checkpoint that was used during training:
          Linear -> ReLU -> Linear
        (no Dropout, no extra layers)
    """

    def __init__(self, in_dim: int, n_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),  # net.0 (has weights/bias)
            nn.ReLU(inplace=True),   # net.1 (no weights)
            nn.Linear(256, n_classes),  # net.2 (has weights/bias)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def rasterize_polygon_mask(
    poly_xy,
    Hf: int,
    Wf: int,
    orig_w: int,
    orig_h: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert a polygon (in original image pixel coords) into a binary mask at feature-map resolution.

    Returns:
      mask: (Hf,Wf) float tensor in {0,1}
    """
    import numpy as np
    from PIL import Image, ImageDraw

    # poly_xy is an Nx2 array in original image coords (x,y)
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

    mask = torch.from_numpy(mask_np).to(device)  # (Hf,Wf)
    return mask


def mask_pool_features(
    feat_map: torch.Tensor,  # (C,Hf,Wf)
    mask: torch.Tensor,      # (Hf,Wf)
) -> torch.Tensor:
    """
    Average-pool features inside a binary mask.

    Returns:
      v: (C,) pooled feature vector.
    """
    C, Hf, Wf = feat_map.shape
    m = mask.view(1, Hf, Wf)  # (1,Hf,Wf)
    denom = m.sum() + 1e-6
    v = (feat_map * m).sum(dim=(1, 2)) / denom
    return v
