"""
stage1_building_seg.py
Purpose:
  Stage-1 training: building segmentation using DeepLabV3-ResNet50.
  Input:  post-disaster RGB
  Target: binary building mask (derived from damage mask)
  Portable dataset (TorchGeo -> custom fallback). Saves best ckpt, logs metrics,
  and evaluates on the Test split. All paths are relative to project root.
"""

from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import models, transforms

from common import (
    PROJECT_ROOT, ensure_dirs, set_seed, ts_filename, ts_human,
    save_metrics, append_run_csv, default_num_workers
)

# ---------------- CONFIG ----------------
CONFIG = {
    "DATA_ROOT": str(PROJECT_ROOT / "data" / "xBD Dataset"),
    "SEED": 123,
    "DEVICE": "auto",
    "RESIZE": 512,
    "BATCH": 2,
    "EPOCHS": 8,
    "LR": 2e-4,
    "VAL_FRAC": 0.1,
    "LIMIT_TRAIN": None,
    "LIMIT_TEST": None,
    "CHECKPOINT": str(PROJECT_ROOT / "models" / "stage1_deeplabv3_resnet50_best.pt"),
    "MATRIX_ROOT": str(PROJECT_ROOT / "matrix"),
    "RUNS_CSV": str(PROJECT_ROOT / "metrics" / "runs.csv"),
    "SAVE_SAMPLE_PRED": True,

    # pretrained handling
    "IMAGENET_WEIGHTS": False,                 # set False to avoid downloads
    "LOCAL_STAGE1_WEIGHTS": str((PROJECT_ROOT / "weights" / "deeplabv3_resnet50_coco-cd0a2569.pth").resolve()),             # e.g., str(PROJECT_ROOT/"weights"/"deeplabv3_resnet50_coco.pth")
}

# ---------------- Portable dataset: TorchGeo -> custom fallback ----------------
try:
    from torchgeo.datasets import XView2 as TGXView2
    _HAS_TORCHGEO = True
except Exception:
    _HAS_TORCHGEO = False

class Stage1Dataset(Dataset):
    """
    Returns:
      post_rgb: FloatTensor (3,H,W) in [0,1]
      bldg_mask: LongTensor (H,W) with {0,1}
      tile_id: str
    Supports both:
      <root>/train/{images,targets} (TorchGeo standard)
      <root>/Train/{image,target}  (your layout)
    """
    def __init__(self, root: str, split: str = "train", resize: int = 512, limit=None):
        self.root = Path(root)
        self.split = split.lower()
        self.resize = resize
        self.limit = limit

        self.rimg = transforms.Resize((resize, resize), transforms.InterpolationMode.BILINEAR)
        self.rmsk = transforms.Resize((resize, resize), transforms.InterpolationMode.NEAREST)

        self.mode = None
        self.base = None
        self.post_list = []
        self.tgt_dir = None

        if _HAS_TORCHGEO:
            try:
                self.base = TGXView2(root=self.root, split=self.split, transforms=None, checksum=False)
                self.mode = "torchgeo"
            except Exception:
                self.base = None
                self.mode = None

        if self.mode is None:
            split_dir = "Train" if self.split == "train" else "Test"
            img_dir = self.root / split_dir / "image"
            tgt_dir = self.root / split_dir / "target"
            if not img_dir.exists() or not tgt_dir.exists():
                raise FileNotFoundError(f"Missing {img_dir} or {tgt_dir}")
            self.post_list = sorted(img_dir.glob("*_post_disaster.png"))
            self.tgt_dir = tgt_dir
            self.mode = "custom"

    def __len__(self):
        n = len(self.base) if self.mode == "torchgeo" else len(self.post_list)
        return min(n, self.limit) if self.limit else n

    def _load_custom(self, idx):
        post_path = self.post_list[idx]
        tgt_path = self.tgt_dir / post_path.name.replace("_post_disaster.png", "_post_disaster_target.png")
        post = Image.open(post_path).convert("RGB")
        post_t = transforms.ToTensor()(self.rimg(post))  # (3,H,W)
        if tgt_path.exists():
            dmg = np.array(Image.open(tgt_path).resize((self.resize, self.resize), Image.NEAREST), dtype=np.uint8)
        else:
            dmg = np.zeros((self.resize, self.resize), np.uint8)
        bldg = (dmg > 0).astype(np.uint8)
        return post_t, torch.from_numpy(bldg).long(), post_path.stem

    def __getitem__(self, idx):
        if self.mode == "torchgeo":
            s = self.base[idx]
            post = (s["image"][1].float() / 255.0)
            dmg = s["mask"][1].long()
            post = self.rimg(post)
            bldg = self.rmsk(dmg.unsqueeze(0)).squeeze(0)
            bldg = (bldg > 0).long()
            tile_id = f"{idx:06d}"
            return post, bldg, tile_id
        else:
            return self._load_custom(idx)

# ---------------- Model & metrics ----------------
def build_model():
    """
    Build DeepLabV3-ResNet50 for 2 classes (bg/building).
    Load order:
      1) If LOCAL_STAGE1_WEIGHTS is a torchvision COCO checkpoint (21 classes),
         prune classifier/aux head keys and load the rest (backbone/features).
      2) Else if IMAGENET_WEIGHTS is True, try torchvision COCO weights (online).
      3) Else random init.
    """
    want_pretrained = CONFIG.get("IMAGENET_WEIGHTS", True)
    local_path = CONFIG.get("LOCAL_STAGE1_WEIGHTS", None)

    # Base (random-init) with 2-class head
    m = models.segmentation.deeplabv3_resnet50(weights=None, weights_backbone=None)
    m.classifier[4] = nn.Conv2d(256, 2, 1)

    if local_path:
        try:
            sd = torch.load(local_path, map_location="cpu")
            # unwrap checkpoints like {"state_dict": ...}
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]

            # --- PRUNE 21-class head keys so shapes don't clash ---
            pruned = {}
            drop_prefixes = ("classifier.4.", "aux_classifier.4.")
            for k, v in sd.items():
                if k.startswith(drop_prefixes):
                    # skip the 21-class classifier layers
                    continue
                pruned[k] = v

            missing, unexpected = m.load_state_dict(pruned, strict=False)
            print(f"[Stage1] Loaded local Deeplab backbone from: {local_path}")
            # Optional: show what we skipped (should include our 2-class head params)
            print(f"[Stage1] load_state_dict missing={len(missing)} unexpected={len(unexpected)} "
                  f"(this is OK; we reinit the 2-class head).")
            return m
        except Exception as e:
            print(f"[Stage1] Could not load LOCAL_STAGE1_WEIGHTS ({local_path}): {e}")
            print("[Stage1] Continuing with next option...")

    if want_pretrained:
        try:
            tv = models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
            m_tv = models.segmentation.deeplabv3_resnet50(weights=tv)
            m_tv.classifier[4] = nn.Conv2d(256, 2, 1)  # swap to 2-class head
            print("[Stage1] Loaded torchvision COCO weights.")
            return m_tv
        except Exception as e:
            print(f"[Stage1] Could not load torchvision weights (offline/SSL?): {e}")
            print("[Stage1] Falling back to RANDOM INIT.")

    return m

def iou_f1(pred, true):
    iou_list, f1_list = [], []
    for p, t in zip(pred, true):
        p = p.cpu().numpy().astype(np.uint8)
        t = t.cpu().numpy().astype(np.uint8)
        inter = (p & t).sum()
        union = (p | t).sum()
        tp = inter
        fp = (p & (~t)).sum()
        fn = ((~p.astype(bool)) & t).sum()
        iou = inter / union if union > 0 else 1.0
        denom = 2 * tp + fp + fn
        f1 = (2 * tp) / denom if denom > 0 else 1.0
        iou_list.append(iou); f1_list.append(f1)
    return float(np.mean(iou_list)), float(np.mean(f1_list))

def train_epoch(model, dl, opt, device):
    model.train(); total = 0.0
    for post, bldg, _ in tqdm(dl, desc="train", leave=False):
        post, bldg = post.to(device), bldg.to(device)
        opt.zero_grad()
        out = model(post)["out"]
        loss = F.cross_entropy(out, bldg)
        loss.backward(); opt.step()
        total += loss.item() * post.size(0)
    return total / len(dl.dataset)

@torch.no_grad()
def validate(model, dl, device):
    model.eval(); total = 0.0; ious = []; f1s = []
    for post, bldg, _ in tqdm(dl, desc="val", leave=False):
        post, bldg = post.to(device), bldg.to(device)
        out = model(post)["out"]
        loss = F.cross_entropy(out, bldg)
        total += loss.item() * post.size(0)
        pred = out.argmax(1)
        iou, f1 = iou_f1(pred, bldg); ious.append(iou); f1s.append(f1)
    return float(total / len(dl.dataset)), float(np.mean(ious)), float(np.mean(f1s))

@torch.no_grad()
def evaluate_on_test(model, cfg, device):
    ds = Stage1Dataset(cfg["DATA_ROOT"], split="test", resize=cfg["RESIZE"], limit=cfg.get("LIMIT_TEST"))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=default_num_workers())
    total = 0.0; ious = []; f1s = []
    save_dir = Path(PROJECT_ROOT / "artifacts" / "stage1_masks" / "test_samples")
    if cfg["SAVE_SAMPLE_PRED"]:
        save_dir.mkdir(parents=True, exist_ok=True)
    for i, (post, bldg, tile_id) in enumerate(tqdm(dl, desc="test", leave=False)):
        post, bldg = post.to(device), bldg.to(device)
        out = model(post)["out"]
        loss = F.cross_entropy(out, bldg); total += loss.item()
        pred = out.argmax(1)
        iou, f1 = iou_f1(pred, bldg); ious.append(iou); f1s.append(f1)
        if cfg["SAVE_SAMPLE_PRED"] and i < 10:
            m = (pred[0].cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(m).save(save_dir / f"{tile_id[0]}_bldg_pred.png")
    return float(total / len(dl)), float(np.mean(ious)), float(np.mean(f1s))

def make_loaders(cfg):
    ds = Stage1Dataset(cfg["DATA_ROOT"], split="train", resize=cfg["RESIZE"], limit=cfg.get("LIMIT_TRAIN"))
    n = len(ds); n_val = int(n * cfg["VAL_FRAC"]); n_tr = n - n_val
    tr, va = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(cfg["SEED"]))
    tr_dl = DataLoader(tr, batch_size=cfg["BATCH"], shuffle=True,
                       num_workers=default_num_workers(), pin_memory=False)
    va_dl = DataLoader(va, batch_size=cfg["BATCH"], shuffle=False,
                       num_workers=default_num_workers(), pin_memory=False)
    return tr_dl, va_dl

def main(cfg=CONFIG):
    ensure_dirs(PROJECT_ROOT / "models", PROJECT_ROOT / "matrix", PROJECT_ROOT / "metrics", PROJECT_ROOT / "artifacts" / "stage1_masks")
    set_seed(cfg["SEED"])
    device = torch.device("cuda" if (cfg["DEVICE"] != "cpu" and torch.cuda.is_available()) else "cpu")
    print("Device:", device)

    tr_dl, va_dl = make_loaders(cfg)
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["LR"])

    best_f1 = -1; best_ep = -1
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_iou": [], "val_f1": []}

    for ep in range(1, cfg["EPOCHS"] + 1):
        tr_loss = train_epoch(model, tr_dl, opt, device)
        val_loss, val_iou, val_f1 = validate(model, va_dl, device)
        print(f"[{ep:02d}/{cfg['EPOCHS']}] train={tr_loss:.4f} val={val_loss:.4f} IoU={val_iou:.3f} F1={val_f1:.3f}")
        history["epoch"].append(ep); history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss); history["val_iou"].append(val_iou); history["val_f1"].append(val_f1)
        if val_f1 > best_f1:
            best_f1 = val_f1; best_ep = ep
            torch.save({"model": model.state_dict(), "epoch": ep, "cfg": cfg}, cfg["CHECKPOINT"])

    ck = torch.load(cfg["CHECKPOINT"], map_location=device)
    model.load_state_dict(ck["model"])
    test_loss, test_iou, test_f1 = evaluate_on_test(model, cfg, device)

    stamp = ts_filename()
    run = f"{stamp}_stage1_ep{cfg['EPOCHS']}"
    metrics = {
        "timestamp": stamp, "epochs": cfg["EPOCHS"], "seed": cfg["SEED"],
        "resize": cfg["RESIZE"], "batch": cfg["BATCH"], "lr": cfg["LR"],
        "best_epoch": best_ep, "val_best_f1": best_f1,
        "test_loss": test_loss, "test_iou": test_iou, "test_f1": test_f1,
        "history": history,
    }
    save_metrics(metrics, cfg["MATRIX_ROOT"], run)
    append_run_csv(cfg["RUNS_CSV"], {
        "model_name": "stage1_deeplabv3_resnet50",
        "timestamp": ts_human(),
        "epochs": cfg["EPOCHS"],
        "best_epoch": best_ep,
        "val_f1_best": f"{best_f1:.6f}",
        "test_iou": f"{test_iou:.6f}",
        "test_f1": f"{test_f1:.6f}",
        "resize": cfg["RESIZE"],
        "batch": cfg["BATCH"],
        "lr": cfg["LR"],
        "seed": cfg["SEED"],
    })
    print("Saved:", cfg["CHECKPOINT"])

if __name__ == "__main__":
    main()
