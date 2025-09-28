"""
export_for_app.py
Purpose:
  Run both stages on a subset of the Test split and export:
    - artifacts/app/predictions.csv
    - artifacts/app/predictions.geojson
    - artifacts/app/<tile>_<k>_overlay.png
  Uses trained checkpoints. All paths relative to project root.
"""

from pathlib import Path
import csv, json
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import ndimage as ndi

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from common import PROJECT_ROOT, ensure_dirs, colorize_mask, blend_rgba, default_num_workers

CONFIG = {
    "DATA_ROOT": str(PROJECT_ROOT / "data" / "xBD Dataset"),
    "RESIZE": 512,
    "PATCH": 160,
    "LIMIT_TILES": 400,
    "STAGE1_CKPT": str(PROJECT_ROOT / "models" / "stage1_deeplabv3_resnet50_best.pt"),
    "STAGE2_CKPT": str(PROJECT_ROOT / "models" / "stage2_siamese_resnet18_best.pt"),
    "SAVE_ROOT": str(PROJECT_ROOT / "artifacts" / "app"),
    "DEVICE": "auto",
}

try:
    from torchgeo.datasets import XView2 as TGXView2
    _HAS_TORCHGEO = True
except Exception:
    _HAS_TORCHGEO = False

class PostOnly(Dataset):
    def __init__(self, root, split="test", resize=512, limit=None):
        self.root = Path(root); self.split = split.lower(); self.resize = resize; self.limit = limit
        self.mode = None; self.base = None
        self.post_list = []; self.pre_list = []
        self.rimg = transforms.Resize((resize, resize), transforms.InterpolationMode.BILINEAR)

        if _HAS_TORCHGEO:
            try:
                self.base = TGXView2(root=self.root, split=self.split, transforms=None, checksum=False)
                self.mode = "torchgeo"
            except Exception:
                self.base = None; self.mode = None
        if self.mode is None:
            split_dir = "Test" if self.split == "test" else "Train"
            img_dir = self.root / split_dir / "image"
            if not img_dir.exists():
                raise FileNotFoundError(f"Missing {img_dir}")
            self.post_list = sorted(img_dir.glob("*_post_disaster.png"))
            self.pre_list  = [img_dir / p.name.replace("_post_", "_pre_") for p in self.post_list]
            self.mode = "custom"

    def __len__(self):
        n = len(self.base) if self.mode == "torchgeo" else len(self.post_list)
        return min(n, self.limit) if self.limit else n

    def __getitem__(self, i):
        if self.mode == "torchgeo":
            s = self.base[i]
            pre  = self.rimg((s["image"][0].float()/255.0))
            post = self.rimg((s["image"][1].float()/255.0))
            tile_id = f"{i:06d}"
            return pre, post, tile_id
        else:
            pre_p, post_p = self.pre_list[i], self.post_list[i]
            tfm = transforms.ToTensor()
            pre  = self.rimg(tfm(Image.open(pre_p).convert("RGB")))
            post = self.rimg(tfm(Image.open(post_p).convert("RGB")))
            return pre, post, post_p.stem

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
    ensure_dirs(cfg["SAVE_ROOT"], PROJECT_ROOT / "artifacts" / "stage1_masks", PROJECT_ROOT / "artifacts" / "stage1_instances")
    device = torch.device("cuda" if (cfg["DEVICE"]!="cpu" and torch.cuda.is_available()) else "cpu")

    # Load trained weights (from training scripts)
    stage1 = build_stage1().to(device)
    stage1.load_state_dict(torch.load(cfg["STAGE1_CKPT"], map_location=device)["model"]); stage1.eval()

    stage2 = SiameseResNet18Cls(n_classes=4).to(device)
    stage2.load_state_dict(torch.load(cfg["STAGE2_CKPT"], map_location=device)["model"]); stage2.eval()

    ds = PostOnly(cfg["DATA_ROOT"], split="test", resize=cfg["RESIZE"], limit=cfg["LIMIT_TILES"])
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=default_num_workers())

    rows=[]; features=[]
    for pre, post, tile_id in tqdm(dl, desc="export"):
        pre, post = pre.to(device), post.to(device)
        out = stage1(post)["out"]
        pred_b = out.argmax(1)[0].cpu().numpy().astype(np.uint8)  # 0/1
        Image.fromarray((pred_b*255).astype(np.uint8)).save(PROJECT_ROOT / "artifacts" / "stage1_masks" / f"{tile_id[0]}_bldg.png")

        lab, ninst = ndi.label(pred_b)
        Image.fromarray(lab.astype(np.uint16)).save(PROJECT_ROOT / "artifacts" / "stage1_instances" / f"{tile_id[0]}_lab.tiff")

        H, W = pred_b.shape; r = cfg["PATCH"]//2
        post_img = (post[0].cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)

        for k in range(1, ninst+1):
            ys, xs = np.where(lab == k)
            if len(xs) == 0: continue
            cy, cx = int(np.mean(ys)), int(np.mean(xs))
            y0, y1 = max(0, cy-r), min(H, cy+r); x0, x1 = max(0, cx-r), min(W, cx+r)
            pre_c  = pre[0, :, y0:y1, x0:x1]
            post_c = post[0, :, y0:y1, x0:x1]
            pad_h = max(0, cfg["PATCH"] - (y1 - y0)); pad_w = max(0, cfg["PATCH"] - (x1 - x0))
            if pad_h > 0 or pad_w > 0:
                pre_c  = F.pad(pre_c,  (0, pad_w, 0, pad_h))
                post_c = F.pad(post_c, (0, pad_w, 0, pad_h))
            x6 = torch.cat([pre_c.unsqueeze(0), post_c.unsqueeze(0)], dim=1).to(device)
            probs = torch.softmax(stage2(x6), dim=1)[0]
            cls = int(probs.argmax().item())   # 0..3
            conf = float(probs.max().item())
            label = [1,2,3,4][cls]

            rows.append({"tile_id": tile_id[0], "instance": k, "cx": int(cx), "cy": int(cy),
                         "label": label, "confidence": conf})

            inst_mask = np.zeros((H, W), np.uint8); inst_mask[lab == k] = label
            overlay = blend_rgba(post_img, colorize_mask(inst_mask), 0.45)
            Image.fromarray(overlay).save(Path(cfg["SAVE_ROOT"]) / f"{tile_id[0]}_{k}_overlay.png")

            features.append({"type":"Feature",
                             "properties":{"tile_id":tile_id[0],"instance":k,"label":label,"conf":conf},
                             "geometry":{"type":"Point","coordinates":[int(cx), int(cy)]}})

    with open(Path(cfg["SAVE_ROOT"]) / "predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["tile_id","instance","cx","cy","label","confidence"])
        w.writeheader(); w.writerows(rows)
    with open(Path(cfg["SAVE_ROOT"]) / "predictions.geojson", "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    print("Exported to:", cfg["SAVE_ROOT"])

if __name__ == "__main__":
    main()
