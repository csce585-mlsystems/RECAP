"""
demo_infer_pair.py
Purpose:
  Run the trained two-stage pipeline on your own pre/post PNG pair. Exports
  per-instance overlays + CSV + GeoJSON to artifacts/app/demo_pair/.
  Edit CONFIG['PRE_PATH'] and CONFIG['POST_PATH'] before running.
"""

from pathlib import Path
import csv, json
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

import torch
import torch.nn.functional as F
from torchvision import models, transforms

from common import PROJECT_ROOT, ensure_dirs, colorize_mask, blend_rgba

CONFIG = {
    "PRE_PATH":  str(PROJECT_ROOT / "your_pre.png"),
    "POST_PATH": str(PROJECT_ROOT / "your_post.png"),
    "RESIZE": 512,
    "PATCH": 160,
    "STAGE1_CKPT": str(PROJECT_ROOT / "models" / "stage1_deeplabv3_resnet50_best.pt"),
    "STAGE2_CKPT": str(PROJECT_ROOT / "models" / "stage2_siamese_resnet18_best.pt"),
    "SAVE_ROOT": str(PROJECT_ROOT / "artifacts" / "app" / "demo_pair"),
    "DEVICE": "auto",
}

def load_rgb(path, resize):
    img = Image.open(path).convert("RGB")
    tfm = transforms.Resize((resize, resize), transforms.InterpolationMode.BILINEAR)
    t = tfm(transforms.ToTensor()(img))  # (3,H,W) [0,1]
    return t, np.array(tfm(img))

def build_stage1():
    m = models.segmentation.deeplabv3_resnet50(weights=None, weights_backbone=None)
    m.classifier[4] = torch.nn.Conv2d(256, 2, 1)
    return m

class SiameseResNet18Cls(torch.nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        m = models.resnet18(weights=None)
        m.fc = torch.nn.Identity()
        self.enc = m
        self.head = torch.nn.Sequential(torch.nn.Linear(512,256), torch.nn.ReLU(True), torch.nn.Linear(256,n_classes))
    def forward(self, x6):
        pre, post = x6[:, :3], x6[:, 3:]
        f_pre = self.enc(pre); f_post = self.enc(post)
        return self.head(torch.abs(f_pre - f_post))

def main(cfg=CONFIG):
    ensure_dirs(cfg["SAVE_ROOT"])
    device = torch.device("cuda" if (cfg["DEVICE"]!="cpu" and torch.cuda.is_available()) else "cpu")

    pre_t, pre_rgb = load_rgb(cfg["PRE_PATH"], cfg["RESIZE"])
    post_t, post_rgb = load_rgb(cfg["POST_PATH"], cfg["RESIZE"])

    stage1 = build_stage1().to(device)
    stage1.load_state_dict(torch.load(cfg["STAGE1_CKPT"], map_location=device)["model"]); stage1.eval()

    stage2 = SiameseResNet18Cls(n_classes=4).to(device)
    stage2.load_state_dict(torch.load(cfg["STAGE2_CKPT"], map_location=device)["model"]); stage2.eval()

    with torch.no_grad():
        out = stage1(post_t.unsqueeze(0).to(device))["out"]
        pred_b = out.argmax(1)[0].cpu().numpy().astype(np.uint8)  # 0/1

    lab, ninst = ndi.label(pred_b)
    H, W = pred_b.shape; r = cfg["PATCH"] // 2
    rows = []; features = []

    for k in range(1, ninst + 1):
        ys, xs = np.where(lab == k)
        if len(xs) == 0: continue
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        y0, y1 = max(0, cy - r), min(H, cy + r)
        x0, x1 = max(0, cx - r), min(W, cx + r)

        pre_c  = pre_t[:, y0:y1, x0:x1]
        post_c = post_t[:, y0:y1, x0:x1]
        pad_h = max(0, cfg["PATCH"] - (y1 - y0)); pad_w = max(0, cfg["PATCH"] - (x1 - x0))
        if pad_h > 0 or pad_w > 0:
            pre_c  = F.pad(pre_c,  (0, pad_w, 0, pad_h))
            post_c = F.pad(post_c, (0, pad_w, 0, pad_h))
        x6 = torch.cat([pre_c.unsqueeze(0), post_c.unsqueeze(0)], dim=1).to(device)

        with torch.no_grad():
            probs = torch.softmax(stage2(x6), dim=1)[0]
            cls = int(probs.argmax().item())  # 0..3
            conf = float(probs.max().item())
            label = [1, 2, 3, 4][cls]

        rows.append({"instance": k, "cx": int(cx), "cy": int(cy), "label": label, "confidence": conf})

        inst_mask = np.zeros((H, W), np.uint8); inst_mask[lab == k] = label
        overlay = blend_rgba(post_rgb, colorize_mask(inst_mask), 0.45)
        Image.fromarray(overlay).save(Path(cfg["SAVE_ROOT"]) / f"overlay_{k:03d}_label{label}_conf{conf:.2f}.png")

        features.append({"type": "Feature",
                         "properties": {"instance": k, "label": label, "conf": conf},
                         "geometry": {"type": "Point", "coordinates": [int(cx), int(cy)]}})

    with open(Path(cfg["SAVE_ROOT"]) / "demo_pair.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["instance", "cx", "cy", "label", "confidence"])
        w.writeheader(); [w.writerow(r) for r in rows]
    with open(Path(cfg["SAVE_ROOT"]) / "demo_pair.geojson", "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    print("Saved results to", cfg["SAVE_ROOT"])

if __name__ == "__main__":
    main()
