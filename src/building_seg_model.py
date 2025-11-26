from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torchvision.models.segmentation import fcn_resnet50

from .common import PROJECT_ROOT


class BuildingSegModel(nn.Module):
    """
    Binary building segmentation using FCN-ResNet50.

    - If pretrained=True, loads COCO weights from a *local* .pth file:
        PROJECT_ROOT / "weights" / "fcn_resnet50_coco-1167a1af.pth"
    - Uses fully offline initialization (no downloads).
    - Replaces the classifier head with a 1-channel conv (building vs background).
    """

    def __init__(
        self,
        pretrained: bool = True,
        local_weight_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Always construct the model with NO remote weights:
        #   - weights=None → no FCN weights
        #   - weights_backbone=None → no ResNet-50 backbone weights
        #   - aux_loss=True so it matches the official COCO checkpoint architecture
        base = fcn_resnet50(
            weights=None,
            weights_backbone=None,
            num_classes=21,
            aux_loss=True,
        )

        if pretrained:
            if local_weight_path is None:
                local_weight_path = str(
                    PROJECT_ROOT / "weights" / "fcn_resnet50_coco-1167a1af.pth"
                )

            weight_path = Path(local_weight_path)
            if not weight_path.exists():
                raise FileNotFoundError(
                    f"[BuildingSegModel] Local FCN weights not found at:\n"
                    f"  {weight_path}\n"
                    f"Download 'fcn_resnet50_coco-1167a1af.pth' once on a machine "
                    f"with internet and place it there."
                )

            # Load the full FCN-ResNet50 state dict from your local file
            state = torch.load(weight_path, map_location="cpu")
            base.load_state_dict(state)
            print(
                f"[BuildingSegModel] Loaded local FCN-ResNet50 weights from: {weight_path}"
            )
        else:
            print("[BuildingSegModel] Using FCN-ResNet50 with random init.")

        # Replace classifier head → 1 logit channel (building vs background)
        in_ch = base.classifier[4].in_channels  # last conv in classifier head
        base.classifier[4] = nn.Conv2d(in_ch, 1, kernel_size=1)

        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, dict):  # torchvision segmentation models return dict
            out = out["out"]
        return out  # (B,1,H,W)
