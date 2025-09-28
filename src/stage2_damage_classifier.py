"""
stage2_damage_classifier.py
Purpose:
  Stage-2 training: per-building damage grading using a Siamese ResNet-18.
  Builds instance crops from GT masks (connected components). Portable dataset
  (TorchGeo -> custom fallback). Saves best ckpt, logs metrics, evaluates on Test.
"""

from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import models, transforms
from scipy import ndimage as ndi

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
    "PATCH": 160,
    "LIMIT_TRAIN_TILES": 800,   # None for all
    "LIMIT_TEST_TILES": None,
    "BATCH": 32,
    "EPOCHS": 8,
    "LR": 1e-3,
    "VAL_FRAC": 0.1,
    "CHECKPOINT": str(PROJECT_ROOT / "models" / "stage2_siamese_resnet18_best.pt"),
    "MATRIX_ROOT": str(PROJECT_ROOT / "matrix"),
    "RUNS_CSV": str(PROJECT_ROOT / "metrics" / "runs.csv"),

    # pretrained handling
    "IMAGENET_WEIGHTS": False,                 # set False to avoid downloads
    "LOCAL_STAGE2_BACKBONE":str((PROJECT_ROOT / "weights" / "resnet18-f37072fd.pth").resolve()),            # e.g., str(PROJECT_ROOT/"weights"/"resnet18.pth")
}

# ---------------- Portable dataset: TorchGeo -> custom fallback ----------------
try:
    from torchgeo.datasets import XView2 as TGXView2
    _HAS_TORCHGEO = True
except Exception:
    _HAS_TORCHGEO = False

class InstanceCropDataset(Dataset):
    """
    Build per-building crops from masks.
    Returns:
      x6: FloatTensor (6,P,P)  [pre|post]
      y:  LongTensor () in {0..3} (maps to labels {1..4} -> 0..3)
    """
    def __init__(self, root, split="train", resize=512, patch=160, limit_tiles=None):
        self.root = Path(root)
        self.split = split.lower()
        self.resize = resize
        self.patch = patch
        self.limit_tiles = limit_tiles

        self.rimg = transforms.Resize((resize, resize), transforms.InterpolationMode.BILINEAR)
        self.samples = []

        self.mode = None
        self.base = None
        self.pre_list = []
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
            self.pre_list  = [img_dir / p.name.replace("_post_", "_pre_") for p in self.post_list]
            self.tgt_dir = tgt_dir
            self.mode = "custom"

        self._prepare()

    def _prepare(self):
        r = self.patch // 2

        if self.mode == "torchgeo":
            N = len(self.base)
            if self.limit_tiles: N = min(N, self.limit_tiles)
            for i in range(N):
                s = self.base[i]
                pre  = self.rimg((s["image"][0].float()/255.0))
                post = self.rimg((s["image"][1].float()/255.0))
                dmg  = transforms.Resize((self.resize, self.resize), transforms.InterpolationMode.NEAREST)(s["mask"][1].long().unsqueeze(0)).squeeze(0).numpy()
                bldg = (dmg > 0).astype(np.uint8)
                lab, ninst = ndi.label(bldg)
                if ninst == 0: continue
                H, W = dmg.shape
                for k in range(1, ninst+1):
                    ys, xs = np.where(lab == k)
                    if len(xs) == 0: continue
                    cy, cx = int(np.mean(ys)), int(np.mean(xs))
                    y0, y1 = max(0, cy-r), min(H, cy+r)
                    x0, x1 = max(0, cx-r), min(W, cx+r)
                    pre_c  = pre[:, y0:y1, x0:x1]
                    post_c = post[:, y0:y1, x0:x1]

                    comp_labels = dmg[ys, xs]
                    nz = comp_labels[comp_labels > 0]
                    if nz.size == 0:
                        label = 1
                    else:
                        vals, cnts = np.unique(nz, return_counts=True)
                        label = int(vals[np.argmax(cnts)])  # 1..4

                    ph = max(0, self.patch - pre_c.shape[-2]); pw = max(0, self.patch - pre_c.shape[-1])
                    if ph > 0 or pw > 0:
                        pre_c  = F.pad(pre_c,  (0, pw, 0, ph))
                        post_c = F.pad(post_c, (0, pw, 0, ph))

                    x6 = torch.cat([pre_c, post_c], dim=0)
                    self.samples.append((x6, label - 1))
        else:
            N = len(self.post_list)
            if self.limit_tiles: N = min(N, self.limit_tiles)
            tfm = transforms.ToTensor()
            for i in range(N):
                pre_p, post_p = self.pre_list[i], self.post_list[i]
                tgt_p = self.tgt_dir / post_p.name.replace("_post_disaster.png", "_post_disaster_target.png")
                pre  = self.rimg(tfm(Image.open(pre_p).convert("RGB")))
                post = self.rimg(tfm(Image.open(post_p).convert("RGB")))
                if tgt_p.exists():
                    dmg = np.array(Image.open(tgt_p).resize((self.resize, self.resize), Image.NEAREST), dtype=np.uint8)
                else:
                    dmg = np.zeros((self.resize, self.resize), np.uint8)

                bldg = (dmg > 0).astype(np.uint8)
                lab, ninst = ndi.label(bldg)
                if ninst == 0: continue
                H, W = dmg.shape
                for k in range(1, ninst + 1):
                    ys, xs = np.where(lab == k)
                    if len(xs) == 0: continue
                    cy, cx = int(np.mean(ys)), int(np.mean(xs))
                    y0, y1 = max(0, cy-r), min(H, cy+r)
                    x0, x1 = max(0, cx-r), min(W, cx+r)
                    pre_c  = pre[:, y0:y1, x0:x1]
                    post_c = post[:, y0:y1, x0:x1]

                    comp_labels = dmg[ys, xs]
                    nz = comp_labels[comp_labels > 0]
                    if nz.size == 0:
                        label = 1
                    else:
                        vals, cnts = np.unique(nz, return_counts=True)
                        label = int(vals[np.argmax(cnts)])  # 1..4

                    ph = max(0, self.patch - pre_c.shape[-2]); pw = max(0, self.patch - pre_c.shape[-1])
                    if ph > 0 or pw > 0:
                        pre_c  = F.pad(pre_c,  (0, pw, 0, ph))
                        post_c = F.pad(post_c, (0, pw, 0, ph))

                    x6 = torch.cat([pre_c, post_c], dim=0)
                    self.samples.append((x6, label - 1))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): x6, y = self.samples[idx]; return x6, torch.tensor(y).long()

# ---------------- Model ----------------
class SiameseResNet18Cls(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        want_pretrained = CONFIG.get("IMAGENET_WEIGHTS", True)
        local_backbone = CONFIG.get("LOCAL_STAGE2_BACKBONE", None)

        m = models.resnet18(weights=None)
        if local_backbone:
            try:
                sd = torch.load(local_backbone, map_location="cpu")
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                m.load_state_dict(sd, strict=False)
                print(f"[Stage2] Loaded local ResNet-18 weights from: {local_backbone}")
            except Exception as e:
                print(f"[Stage2] Could not load LOCAL_STAGE2_BACKBONE ({local_backbone}): {e}")
                print("[Stage2] Continuing with next option...")

        if want_pretrained and local_backbone is None:
            try:
                m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                print("[Stage2] Loaded torchvision ResNet-18 ImageNet weights.")
            except Exception as e:
                print(f"[Stage2] Could not load torchvision ResNet-18 weights (offline/SSL?): {e}")
                print("[Stage2] Falling back to RANDOM INIT.")

        m.fc = nn.Identity()
        self.enc = m
        self.head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True), nn.Linear(256, n_classes))
    def forward(self, x6):
        pre, post = x6[:, :3], x6[:, 3:]
        f_pre = self.enc(pre); f_post = self.enc(post)
        return self.head(torch.abs(f_pre - f_post))

# ---------------- Train/Eval ----------------
def make_loaders(cfg):
    ds = InstanceCropDataset(cfg["DATA_ROOT"], split="train", resize=cfg["RESIZE"],
                             patch=cfg["PATCH"], limit_tiles=cfg.get("LIMIT_TRAIN_TILES"))
    n = len(ds); n_val = int(n * cfg["VAL_FRAC"]); n_tr = n - n_val
    tr, va = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(cfg["SEED"]))
    tr_dl = DataLoader(tr, batch_size=cfg["BATCH"], shuffle=True,
                       num_workers=default_num_workers(), pin_memory=False)
    va_dl = DataLoader(va, batch_size=cfg["BATCH"], shuffle=False,
                       num_workers=default_num_workers(), pin_memory=False)
    return tr_dl, va_dl

@torch.no_grad()
def eval_cls(model, dl, device):
    model.eval(); cm = np.zeros((4, 4), dtype=np.int64)
    for x, y in tqdm(dl, desc="eval", leave=False):
        x, y = x.to(device), y.to(device)
        p = model(x).argmax(1)
        for t, pp in zip(y.cpu().numpy(), p.cpu().numpy()):
            cm[t, pp] += 1
    per_f1 = []; sup = []
    for c in range(4):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1 = 0.0 if denom == 0 else (2 * tp) / denom
        per_f1.append(float(f1)); sup.append(int(cm[c, :].sum()))
    macro = float(np.mean(per_f1)); total = sum(sup) if sum(sup) > 0 else 1
    weighted = float(sum(f * s for f, s in zip(per_f1, sup)) / total)
    return cm.tolist(), per_f1, macro, weighted

@torch.no_grad()
def evaluate_on_test(model, cfg, device):
    ds_test = InstanceCropDataset(cfg["DATA_ROOT"], split="test", resize=cfg["RESIZE"],
                                  patch=cfg["PATCH"], limit_tiles=cfg.get("LIMIT_TEST_TILES"))
    dl_test = DataLoader(ds_test, batch_size=cfg["BATCH"], shuffle=False, num_workers=default_num_workers())
    cm, per, macro, weighted = eval_cls(model, dl_test, device)
    return {"cm": cm, "per_f1": per, "macro_f1": macro, "weighted_f1": weighted}

def main(cfg=CONFIG):
    ensure_dirs(PROJECT_ROOT / "models", PROJECT_ROOT / "matrix", PROJECT_ROOT / "metrics")
    set_seed(cfg["SEED"])
    device = torch.device("cuda" if (cfg["DEVICE"] != "cpu" and torch.cuda.is_available()) else "cpu")
    print("Device:", device)

    tr_dl, va_dl = make_loaders(cfg)
    model = SiameseResNet18Cls(n_classes=4).to(device)
    weights = torch.tensor([1.0, 1.5, 2.0, 2.2], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["LR"])

    best = -1; best_ep = -1
    history = {"epoch": [], "macro_f1": [], "weighted_f1": []}

    for ep in range(1, cfg["EPOCHS"] + 1):
        model.train()
        for x, y in tqdm(tr_dl, desc=f"train {ep}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = criterion(model(x), y); loss.backward(); opt.step()
        _, per, macro, weighted = eval_cls(model, va_dl, device)
        print(f"[{ep:02d}/{cfg['EPOCHS']}] val macroF1={macro:.3f} weightedF1={weighted:.3f}")
        history["epoch"].append(ep); history["macro_f1"].append(macro); history["weighted_f1"].append(weighted)
        if weighted > best:
            best = weighted; best_ep = ep
            torch.save({"model": model.state_dict(), "epoch": ep, "cfg": cfg}, cfg["CHECKPOINT"])

    ck = torch.load(cfg["CHECKPOINT"], map_location=device)
    model.load_state_dict(ck["model"])
    test_stats = evaluate_on_test(model, cfg, device)

    stamp = ts_filename(); run = f"{stamp}_stage2_ep{cfg['EPOCHS']}"
    metrics = {
        "timestamp": stamp, "epochs": cfg["EPOCHS"], "seed": cfg["SEED"],
        "resize": cfg["RESIZE"], "patch": cfg["PATCH"], "batch": cfg["BATCH"], "lr": cfg["LR"],
        "best_epoch": best_ep, "val_best_weighted_f1": best,
        "test_macro_f1": test_stats["macro_f1"], "test_weighted_f1": test_stats["weighted_f1"],
        "per_class_f1_test": test_stats["per_f1"],
        "history": history,
    }
    save_metrics(metrics, cfg["MATRIX_ROOT"], run)
    append_run_csv(cfg["RUNS_CSV"], {
        "model_name": "stage2_siamese_resnet18_cls",
        "timestamp": ts_human(),
        "epochs": cfg["EPOCHS"],
        "best_epoch": best_ep,
        "test_macro_f1": f"{test_stats['macro_f1']:.6f}",
        "test_weighted_f1": f"{test_stats['weighted_f1']:.6f}",
        "resize": cfg["RESIZE"],
        "patch": cfg["PATCH"],
        "batch": cfg["BATCH"],
        "lr": cfg["LR"],
        "seed": cfg["SEED"],
    })
    print("Saved:", cfg["CHECKPOINT"])

if __name__ == "__main__":
    main()
