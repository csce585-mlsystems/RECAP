"""
model_polygon_siamese.py

Purpose:
  - Define the siamese tile backbone (ResNet-34, multi-scale features).
  - Define the per-building damage head.
  - Provide utilities for rasterizing polygon masks and pooling features.
"""

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from .common import PROJECT_ROOT


class SiameseTileBackbone(nn.Module):
    """
    Shared ResNet-34 encoder used for both pre- and post-disaster tiles.

    - Uses conv1 → bn1 → relu → maxpool → layer1/2/3/4.
    - Multi-scale features: take outputs from layer2, layer3, layer4,
      upsample them to layer2 spatial size and concatenate along channels.

    Input:
      x: (3,H,W) or (B,3,H,W) or (1,1,3,H,W)
    Output:
      feature map: (B, C_out, H', W')
        where C_out = 128 + 256 + 512 = 896 for ResNet-34
    """

    def __init__(self, imagenet_weights: bool = True):
        super().__init__()

        if imagenet_weights:
            # Path where you manually placed the weights
            weight_path = PROJECT_ROOT / "weights" / "resnet34-b627a593.pth"

            # Build a ResNet-34 *without* downloading any weights
            base = models.resnet34(weights=None)

            if weight_path.exists():
                state = torch.load(weight_path, map_location="cpu")
                # Official torchvision resnet34-b627a593.pth is just a plain state_dict
                base.load_state_dict(state)
                print(f"[PolygonSiamese] Loaded local ResNet-34 weights from: {weight_path}")
            else:
                print(
                    f"[PolygonSiamese] WARNING: local ResNet-34 weights not found at {weight_path}, "
                    "falling back to random init."
                )
        else:
            print("[PolygonSiamese] Using ResNet-34 with random init (no pretrained weights).")
            base = models.resnet34(weights=None)

        # Keep named blocks for multi-scale forward
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1  # C=64
        self.layer2 = base.layer2  # C=128
        self.layer3 = base.layer3  # C=256
        self.layer4 = base.layer4  # C=512

        # Multi-scale output channels
        self.out_channels = 128 + 256 + 512  # 896

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fix shapes like (1,1,3,H,W) → (1,3,H,W)
        if x.dim() == 5:
            x = x.squeeze(0)
        # If single image (3,H,W), add batch dim
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Standard ResNet stem + layers
        x = self.conv1(x)   # (B,64,H/2,W/2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # (B,64,H/4,W/4)

        x1 = self.layer1(x)  # (B,64, H/4,  W/4)
        x2 = self.layer2(x1) # (B,128,H/8,  W/8)
        x3 = self.layer3(x2) # (B,256,H/16, W/16)
        x4 = self.layer4(x3) # (B,512,H/32, W/32)

        # Use x2, x3, x4 → upsample to x2 spatial size and concat
        B, C2, H2, W2 = x2.shape
        x3_up = F.interpolate(x3, size=(H2, W2), mode="bilinear", align_corners=False)
        x4_up = F.interpolate(x4, size=(H2, W2), mode="bilinear", align_corners=False)

        feat = torch.cat([x2, x3_up, x4_up], dim=1)  # (B, 128+256+512, H2, W2)
        return feat


class DamageHead(nn.Module):
    """
    MLP head that maps per-building feature vectors to 4 damage classes.

    Rich change representation:
      v_pre, v_post, diff = v_post - v_pre, abs_diff = |diff|
      feat = concat(v_pre, v_post, abs_diff, diff) ∈ R^(4 * C_backbone)

    So in_dim = 4 * backbone.out_channels.

    Architecture:
      Linear(4C → 512) → ReLU → Dropout(0.5)
      Linear(512 → 256) → ReLU → Dropout(0.5)
      Linear(256 → 4)
    """

    def __init__(self, in_dim: int, n_classes: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


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

    poly = poly_xy.astype(float)  # Nx2 in original image coords

    scale_x = Wf / float(orig_w)
    scale_y = Hf / float(orig_h)
    poly_scaled = np.stack(
        [poly[:, 0] * scale_x, poly[:, 1] * scale_y],
        axis=1,
    )

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
    m = mask.view(1, Hf, Wf)
    denom = m.sum() + 1e-6
    v = (feat_map * m).sum(dim=(1, 2)) / denom
    return v
