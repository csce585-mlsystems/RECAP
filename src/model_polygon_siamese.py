from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from pathlib import Path


class SiameseTileBackbone(nn.Module):
    """
    Shared ResNet-34 encoder used for both pre- and post-disaster tiles.

    Accepts:
      x: (3,H,W) or (B,3,H,W) or (B,1,3,H,W) or (1,1,3,H,W)
    Returns:
      feat: (B, C, H', W')
    """

    def __init__(
        self,
        imagenet_weights: bool = True,
        local_weight_path: str | None = None,
    ):
        super().__init__()

        if imagenet_weights:
            if local_weight_path is not None and Path(local_weight_path).exists():
                # load ResNet-34 weights from local .pth
                base = models.resnet34(weights=None)
                print(
                    f"[PolygonSiamese] Loaded local ResNet-34 weights from: {local_weight_path}"
                )
                state = torch.load(local_weight_path, map_location="cpu")
                base.load_state_dict(state)
            else:
                # fallback: torchvision IMAGENET1K_V1 (will try to download)
                print(
                    "[PolygonSiamese] Using ResNet-34 with ImageNet pretrained weights (torchvision)."
                )
                base = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            print("[PolygonSiamese] Using ResNet-34 with random init.")
            base = models.resnet34(weights=None)

        # Keep stem + conv layers up to layer4 (drop avgpool, fc)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.out_channels = 512  # ResNet-34 final feature dim

    def _normalize_shape(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make sure x is (B,3,H,W) float32.

        Handles common bugs:
          - (3,H,W)  → add batch dim
          - (1,3,H,W) or (B,1,3,H,W) → squeeze the extra 1
          - (1,1,3,H,W) → squeeze both singleton dims
        """
        # If 5D, squeeze a singleton dimension
        if x.dim() == 5:
            # typical from DataLoader over (1,3,H,W): (B,1,3,H,W)
            if x.size(1) == 1 and x.size(2) == 3:
                x = x.squeeze(1)  # (B,3,H,W)
            # sometimes (1,1,3,H,W)
            elif x.size(0) == 1 and x.size(1) == 1 and x.size(2) == 3:
                x = x.squeeze(0).squeeze(0)  # (3,H,W)
            else:
                # last-resort: flatten to (B',3,H,W)
                B = x.size(0) * x.size(1)
                x = x.view(B, 3, x.size(-2), x.size(-1))

        # If single image (3,H,W), add batch dim
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Final sanity check
        assert x.dim() == 4, f"Expected 4D tensor (B,3,H,W), got shape {x.shape}"
        if x.size(1) != 3:
            raise ValueError(f"Expected 3 channels, got {x.size(1)} in shape {x.shape}")

        if x.dtype != torch.float32:
            x = x.float()

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,3,H,W) or similar
        returns: (B,512,H',W')
        """
        x = self._normalize_shape(x)

        x = self.conv1(x)     # (B,64,H/2,W/2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)   # (B,64,H/4,W/4)

        x = self.layer1(x)    # (B,64, ...)
        x = self.layer2(x)    # (B,128, ...)
        x = self.layer3(x)    # (B,256, ...)
        x = self.layer4(x)    # (B,512, ...)

        return x  # (B,512,H',W')
