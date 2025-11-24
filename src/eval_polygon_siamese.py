"""
eval_polygon_siamese.py
Purpose:
  Evaluate the trained polygon-aware Siamese model on:
    - train split
    - test split
  and save:
    - metrics JSON for each split
    - per-building prediction CSVs for each split

Outputs:
  metrics/polygon_siamese_train_eval.json
  metrics/polygon_siamese_test_eval.json
  metrics/polygon_siamese_train_predictions.csv
  metrics/polygon_siamese_test_predictions.csv
"""

from pathlib import Path
import json
import csv
from typing import Dict, List

import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

import torch
from torch.utils.data import DataLoader

from .common import PROJECT_ROOT, get_device, set_seed, IDX2LABEL
from .dataset_tiles import TileBuildingDataset
from .model_polygon_siamese import (
    SiameseTileBackbone,
    DamageHead,
    rasterize_polygon_mask,
    mask_pool_features,
)


CONFIG = {
    "DATA_ROOT": str(PROJECT_ROOT / "data" / "xBD Dataset"),
    "RESIZE": 512,
    "SEED": 42,
    "MODEL_PATH": str(PROJECT_ROOT / "models" / "polygon_siamese_best.pt"),

    # limits for quick debugging; set to None for full evaluation
    "LIMIT_TILES_TRAIN_EVAL": None,   # None -> all train tiles
    "LIMIT_TILES_TEST_EVAL": None,    # None -> all test tiles

    "NUM_WORKERS": 0,
}


def eval_split(
    split_name: str,
    cfg: Dict,
    backbone: SiameseTileBackbone,
    head: DamageHead,
    device: torch.device,
) -> Dict:
    """
    Evaluate on a given split ('train' or 'test') and return a metrics dict.
    Also saves per-building predictions to CSV.
    """
    print(f"\n[Eval] Split = {split_name}")
    if split_name == "train":
        limit_tiles = cfg["LIMIT_TILES_TRAIN_EVAL"]
    else:
        limit_tiles = cfg["LIMIT_TILES_TEST_EVAL"]

    ds = TileBuildingDataset(
        root=cfg["DATA_ROOT"],
        split=split_name,
        resize=cfg["RESIZE"],
        limit_tiles=limit_tiles,
    )

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["NUM_WORKERS"],
        collate_fn=lambda b: b[0],  # B=1
    )

    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    backbone.eval()
    head.eval()

    # ensure metric paths
    metrics_dir = PROJECT_ROOT / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / f"polygon_siamese_{split_name.lower()}_predictions.csv"

    all_conf: List[np.ndarray] = []
    all_tile_ids: List[str] = []
    all_uids: List[str] = []

    # open CSV writer
    with csv_path.open("w", newline="") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "tile_id",
                "uid",
                "true_idx",
                "true_label",
                "pred_idx",
                "pred_label",
                "confidence",
            ],
        )
        writer.writeheader()

        for pre_t, post_t, polys_xy, labels_t, tile_id, orig_w, orig_h in tqdm(
            loader, desc=f"{split_name}-eval"
        ):
            pre_t = pre_t.to(device).unsqueeze(0)  # (1,3,H,W)
            post_t = post_t.to(device).unsqueeze(0)
            labels_t = labels_t.to(device)
            polys = polys_xy
            orig_w = int(orig_w)
            orig_h = int(orig_h)

            if labels_t.numel() == 0:
                continue

            # forward
            with torch.no_grad():
                F_pre = backbone(pre_t)[0]   # (C,H',W')
                F_post = backbone(post_t)[0]

            C, Hf, Wf = F_pre.shape
            feats_diff = []
            gt_labels = []
            poly_valid = []

            for poly_xy, y in zip(polys, labels_t):
                mask = rasterize_polygon_mask(poly_xy, Hf, Wf, orig_w, orig_h, device)
                if mask.sum() < 1.0:
                    continue
                v_pre = mask_pool_features(F_pre, mask)
                v_post = mask_pool_features(F_post, mask)
                feats_diff.append(torch.abs(v_post - v_pre))
                gt_labels.append(int(y.item()))
                poly_valid.append(poly_xy)

            if len(feats_diff) == 0:
                continue

            X = torch.stack(feats_diff, dim=0)  # (#bldg, C)
            y_true = np.array(gt_labels, dtype=np.int64)

            with torch.no_grad():
                logits = head(X)                     # (#bldg,4)
                probs = torch.softmax(logits, dim=1) # (#bldg,4)
                confs, preds = probs.max(dim=1)

            y_pred = preds.cpu().numpy().astype(np.int64)
            confs_np = confs.cpu().numpy().astype(float)

            # store metrics arrays
            all_true.append(y_true)
            all_pred.append(y_pred)
            all_conf.append(confs_np)
            all_tile_ids.extend([tile_id] * len(y_true))

            # get building UIDs from JSON to include in CSV
            label_json_path = (
                Path(cfg["DATA_ROOT"]) / split_name / "labels" / f"{tile_id}.json"
            )
            with label_json_path.open("r") as f_lbl:
                label_meta = json.load(f_lbl)

            feats = label_meta["features"]["xy"]
            uids = []
            for feat in feats:
                props = feat["properties"]
                if props.get("feature_type") != "building":
                    continue
                subtype = props.get("subtype", "")
                if subtype not in IDX2LABEL.values():
                    continue
                uids.append(props.get("uid", ""))

            # align UIDs to valid polys (could mismatch if some polys were dropped)
            n_keep = len(poly_valid)
            uids = uids[:n_keep]
            all_uids.extend(uids)

            # write rows to CSV
            for uid, ti, t_idx, p_idx, conf in zip(
                uids, [tile_id] * len(y_true), y_true, y_pred, confs_np
            ):
                writer.writerow(
                    {
                        "tile_id": ti,
                        "uid": uid,
                        "true_idx": int(t_idx),
                        "true_label": IDX2LABEL[int(t_idx)],
                        "pred_idx": int(p_idx),
                        "pred_label": IDX2LABEL[int(p_idx)],
                        "confidence": float(conf),
                    }
                )

    # aggregate metrics
    if not all_true:
        print(f"[Eval:{split_name}] No buildings evaluated.")
        metrics = {}
    else:
        y_true_all = np.concatenate(all_true, axis=0)
        y_pred_all = np.concatenate(all_pred, axis=0)

        acc = accuracy_score(y_true_all, y_pred_all)
        macro_f1 = f1_score(y_true_all, y_pred_all, average="macro")
        weighted_f1 = f1_score(y_true_all, y_pred_all, average="weighted")
        prec, rec, f1_vals, support = precision_recall_fscore_support(
            y_true_all, y_pred_all, labels=[0, 1, 2, 3], zero_division=0
        )
        cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1, 2, 3]).tolist()
        report = classification_report(
            y_true_all,
            y_pred_all,
            target_names=[IDX2LABEL[i] for i in range(4)],
            digits=3,
        )

        metrics = {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "per_class": {
                IDX2LABEL[i]: {
                    "precision": float(prec[i]),
                    "recall": float(rec[i]),
                    "f1": float(f1_vals[i]),
                    "support": int(support[i]),
                }
                for i in range(4)
            },
            "confusion_matrix": cm,
            "classification_report": report,
            "n_buildings": int(y_true_all.size),
        }

        print(f"[Eval:{split_name}] Accuracy: {acc:.4f}, macro-F1: {macro_f1:.4f}, weighted-F1: {weighted_f1:.4f}")
        print(report)

    # save metrics JSON
    metrics_path = PROJECT_ROOT / "metrics" / f"polygon_siamese_{split_name.lower()}_eval.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Eval:{split_name}] Saved metrics to {metrics_path}")
    print(f"[Eval:{split_name}] Saved predictions CSV to {csv_path}")

    return metrics


def main():
    cfg = CONFIG
    set_seed(cfg["SEED"])
    device = get_device(prefer_gpu=True)
    print("Device:", device)

    # load model
    ckpt_path = Path(cfg["MODEL_PATH"])
    assert ckpt_path.exists(), f"Model checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device)

    backbone = SiameseTileBackbone(imagenet_weights=False).to(device)
    head = DamageHead(in_dim=backbone.out_channels, n_classes=4).to(device)
    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])

    # Evaluate both train and test
    _ = eval_split("train", cfg, backbone, head, device)
    _ = eval_split("test", cfg, backbone, head, device)


if __name__ == "__main__":
    main()
