"""
eval_full_model.py
Purpose:
  Evaluate the FULL polygon_siamese model on the test set, compute
  key metrics, and generate plots for your paper and slides.

Outputs (relative to repo root):
  artifacts/plots/
      confusion_matrix.png
      per_class_f1_bar.png
      class_support.png
      roc_curves.png
      pr_curves.png
      calibration_curve.png
      metrics_summary.csv
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score,
)
from sklearn.calibration import calibration_curve
import pandas as pd

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  # <-- progress bar

from .common import (
    PROJECT_ROOT,
    get_device,
    IDX2LABEL,
    set_seed,
)
from .dataset_tiles import TileBuildingDataset
from .model_polygon_siamese import (
    SiameseTileBackbone,
    DamageHead,
    rasterize_polygon_mask,
    mask_pool_features,
)


CONFIG = {
    "DATA_ROOT": str(PROJECT_ROOT / "data" / "xBD Dataset"),
    "SPLIT": "test",
    "RESIZE": 512,
    "SEED": 123,
    "MODEL_PATH": str(PROJECT_ROOT / "models" / "polygon_siamese_best.pt"),
    "OUT_DIR": str(PROJECT_ROOT / "artifacts" / "plots"),
    "NUM_WORKERS": 0,
    # Optional: set to a small number to do a quick debug run, or None for full test
    "MAX_TILES": None,  # e.g. 100 for a faster run
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_pair_features(v_pre: torch.Tensor, v_post: torch.Tensor) -> torch.Tensor:
    """
    Build the same 4x feature vector used in training:

        [v_pre,
         v_post,
         v_post - v_pre,
         |v_post - v_pre|] -> shape = (4*C,)
    """
    diff = v_post - v_pre
    adiff = torch.abs(diff)
    return torch.cat([v_pre, v_post, diff, adiff], dim=0)


def evaluate(cfg: dict) -> None:
    set_seed(cfg["SEED"])
    device = get_device(prefer_gpu=True)
    print("Device:", device)

    # ---------- Dataset ----------
    ds = TileBuildingDataset(
        root=cfg["DATA_ROOT"],
        split=cfg["SPLIT"],
        resize=cfg["RESIZE"],
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["NUM_WORKERS"],
        collate_fn=lambda b: b[0],  # un-batch
    )
    n_tiles = len(ds)
    print(f"Test tiles: {n_tiles}")

    max_tiles = cfg.get("MAX_TILES")
    if max_tiles is not None:
        print(f"[Eval] Limiting to first {max_tiles} tiles for this run.")

    # ---------- Model ----------
    ckpt_path = Path(cfg["MODEL_PATH"])
    assert ckpt_path.exists(), f"Model checkpoint not found: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    backbone = SiameseTileBackbone(imagenet_weights=False).to(device)
    head = DamageHead(in_dim=backbone.out_channels * 4, n_classes=4).to(device)

    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])
    backbone.eval()
    head.eval()

    all_true = []
    all_pred = []
    all_probs = []

    print("\n[Eval] Running full evaluation on test dataset...\n")

    # tqdm progress bar
    for idx, batch in enumerate(tqdm(loader, desc="[Eval]", total=n_tiles)):
        if max_tiles is not None and idx >= max_tiles:
            break

        pre_t, post_t, polys, labels_t, tile_id, orig_w, orig_h = batch

        pre_t = pre_t.to(device)
        post_t = post_t.to(device)

        with torch.no_grad():
            F_pre = backbone(pre_t)[0]  # (C, Hf, Wf)
            F_post = backbone(post_t)[0]

        C, Hf, Wf = F_pre.shape

        feats_batch = []
        labels_batch = []

        labels_list = labels_t.tolist()

        for poly_xy, lab_idx in zip(polys, labels_list):
            mask = rasterize_polygon_mask(poly_xy, Hf, Wf, orig_w, orig_h, device)
            if mask.sum() < 1:
                continue

            v_pre = mask_pool_features(F_pre, mask)   # (C,)
            v_post = mask_pool_features(F_post, mask) # (C,)
            feat = build_pair_features(v_pre, v_post) # (4*C,)

            feats_batch.append(feat)
            labels_batch.append(lab_idx)

        if len(feats_batch) == 0:
            continue

        X = torch.stack(feats_batch, dim=0)
        y_true_tile = np.array(labels_batch, dtype=np.int64)

        with torch.no_grad():
            logits = head(X)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

        all_true.append(y_true_tile)
        all_pred.append(preds.cpu().numpy().astype(np.int64))
        all_probs.append(probs.cpu().numpy())

    if not all_true:
        print("[Eval] No buildings were processed. Check dataset / config.")
        return

    # --------- Stack ----------
    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    y_probs = np.concatenate(all_probs, axis=0)  # shape (N, 4)

    # ===========================
    # METRIC SUMMARY (PRINT)
    # ===========================
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    print("====== TEST METRICS ======")
    print(f"Accuracy   : {acc:.4f}")
    print(f"Macro F1   : {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("\nFull Classification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[IDX2LABEL[i] for i in range(4)],
            zero_division=0,
        )
    )

    out_dir = Path(cfg["OUT_DIR"])
    ensure_dir(out_dir)

    # ===========================
    # CONFUSION MATRIX
    # ===========================
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    tick_labels = [IDX2LABEL[i] for i in range(4)]
    plt.xticks(range(4), tick_labels, rotation=45, ha="right")
    plt.yticks(range(4), tick_labels)
    plt.colorbar()
    for i in range(4):
        for j in range(4):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.close()

    # ===========================
    # PER-CLASS F1 BAR PLOT
    # ===========================
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3])
    plt.figure(figsize=(8, 5))
    plt.bar(tick_labels, per_class_f1)
    plt.ylim(0, 1)
    plt.ylabel("F1 score")
    plt.title("Per-Class F1 Score")
    plt.tight_layout()
    plt.savefig(out_dir / "per_class_f1_bar.png")
    plt.close()

    # ===========================
    # CLASS SUPPORT PLOT
    # ===========================
    unique, counts = np.unique(y_true, return_counts=True)
    labels_txt = [IDX2LABEL[u] for u in unique]

    plt.figure(figsize=(8, 5))
    plt.bar(labels_txt, counts)
    plt.ylabel("Number of Buildings")
    plt.title("Class Support in Test Set")
    plt.tight_layout()
    plt.savefig(out_dir / "class_support.png")
    plt.close()

    # ===========================
    # ROC CURVES (ONE-VS-REST)
    # ===========================
    plt.figure(figsize=(8, 6))
    for i in range(4):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_probs[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{IDX2LABEL[i]} (AUC={auc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curves.png")
    plt.close()

    # ===========================
    # PRECISION-RECALL CURVES
    # ===========================
    plt.figure(figsize=(8, 6))
    for i in range(4):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_bin, y_probs[:, i])
        plt.plot(recall, precision, label=IDX2LABEL[i])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curves.png")
    plt.close()

    # ===========================
    # CALIBRATION CURVE
    # ===========================
    plt.figure(figsize=(8, 6))
    for i in range(4):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0:
            continue
        prob_true, prob_pred = calibration_curve(y_bin, y_probs[:, i], n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", label=IDX2LABEL[i])
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "calibration_curve.png")
    plt.close()

    # ===========================
    # SAVE METRICS CSV
    # ===========================
    df = pd.DataFrame(
        {
            "metric": ["accuracy", "macro_f1", "weighted_f1"],
            "value": [acc, macro_f1, weighted_f1],
        }
    )
    df.to_csv(out_dir / "metrics_summary.csv", index=False)

    print("\nSaved all plots + metrics to:", out_dir)


def main():
    evaluate(CONFIG)


if __name__ == "__main__":
    main()
