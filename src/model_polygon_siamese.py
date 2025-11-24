"""
model_polygon_siamese.py
Purpose:
  Siamese ResNet-18 backbone for pre/post tiles + MLP head for damage,
  plus mask-pooling utilities that turn polygons into building embeddings.
"""
from pathlib import Path
import torch

from .common import PROJECT_ROOT
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

WEIGHTS_DIR = PROJECT_ROOT / "weights"
LOCAL_RESNET18 = WEIGHTS_DIR / "resnet18-f37072fd.pth"
class SiameseTileBackbone(nn.Module):
    """
    Shared ResNet-18 encoder that returns a high-level feature map for a tile.
    We remove avgpool+fc to keep spatial features: (B, C, H', W').
    """
    def __init__(self, imagenet_weights: bool = True):
        super().__init__()

        if imagenet_weights and LOCAL_RESNET18.exists():
            print(f"[PolygonSiamese] Loaded local ResNet-18 weights from: {LOCAL_RESNET18}")
            base = models.resnet18(weights=None)  # do not download
            state = torch.load(LOCAL_RESNET18, map_location="cpu")
            base.load_state_dict(state)
        elif imagenet_weights:
            print("[PolygonSiamese] Using torchvision ResNet-18 weights (will attempt download).")
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            print("[PolygonSiamese] Using ResNet-18 with random init (no pretrained weights).")
            base = models.resnet18(weights=None)

        # keep everything up to layer4 as the encoder
        self.encoder = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )

        # number of channels in the final feature map
        self.out_channels = base.fc.in_features
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:  (B, 3, H, W)
        out:(B, C, H', W')
        """
        return self.encoder(x)


class DamageHead(nn.Module):
    """
    Simple MLP head that maps building-level feature difference to 4 damage classes.
    """
    def __init__(self, in_dim: int = 512, n_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def rasterize_polygon_mask(
    poly_xy: np.ndarray,
    feat_h: int,
    feat_w: int,
    orig_w: int,
    orig_h: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Given polygon in ORIGINAL pixel coords (x,y) and feature map size (feat_w, feat_h),
    create a float32 mask of shape (feat_h, feat_w) with 1 inside the polygon and 0 outside.
    """
    # map from original (0..orig_w-1, 0..orig_h-1) to feature map (0..feat_w-1, 0..feat_h-1)
    sx = feat_w / float(orig_w)
    sy = feat_h / float(orig_h)
    xs = poly_xy[:, 0] * sx
    ys = poly_xy[:, 1] * sy

    # clamp to valid range
    xs = np.clip(xs, 0, feat_w - 1)
    ys = np.clip(ys, 0, feat_h - 1)

    img = Image.new("L", (feat_w, feat_h), 0)
    draw = ImageDraw.Draw(img)
    coords = list(zip(xs.tolist(), ys.tolist()))
    draw.polygon(coords, outline=1, fill=1)
    mask_np = np.array(img, dtype=np.float32)  # 0 or 1
    mask_t = torch.from_numpy(mask_np).to(device)  # (H',W')
    return mask_t


def mask_pool_features(
    feat: torch.Tensor,  # (C,H',W')
    mask: torch.Tensor,  # (H',W')
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Mean-pool features inside mask.
    returns: (C,)
    """
    # ensure same device
    C, H, W = feat.shape
    mask = mask.view(1, H, W)
    # multiply and sum
    weighted = feat * mask
    s = weighted.view(C, -1).sum(dim=1)  # (C,)
    denom = mask.view(-1).sum() + eps
    return s / denom
