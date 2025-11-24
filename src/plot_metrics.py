"""
plot_metrics.py
Purpose:
    Read metrics from eval_polygon_siamese.py and generate
    publication-ready plots:
      - Per-class precision/recall/F1 (train + test)
      - Confusion matrix heatmaps (train + test)
      - Overall accuracy / macro-F1 bars

Outputs (saved under PROJECT_ROOT / "figures"):
    figures/polygon_siamese_train_confusion.png
    figures/polygon_siamese_test_confusion.png
    figures/polygon_siamese_train_per_class.png
    figures/polygon_siamese_test_per_class.png
    figures/polygon_siamese_overall_metrics.png
"""

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

from .common import PROJECT_ROOT, IDX2LABEL


def load_metrics(split_name: str):
    """
    Load metrics JSON for a split ("train" or "test").
    """
    metrics_path = PROJECT_ROOT / "metrics" / f"polygon_siamese_{split_name}_eval.json"
    if not metrics_path.exists():
        print(f"[plot_metrics] Metrics file not found: {metrics_path}")
        return None
    with metrics_path.open("r") as f:
        return json.load(f)


def plot_confusion_matrix(metrics: dict, split_name: str, out_dir: Path):
    """
    Plot confusion matrix as a heatmap for the given split.
    """
    cm = np.array(metrics["confusion_matrix"])
    labels = [IDX2LABEL[i] for i in range(4)]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix ({split_name.capitalize()})")

    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    fig.tight_layout()
    out_path = out_dir / f"polygon_siamese_{split_name}_confusion.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[plot_metrics] Saved {out_path}")


def plot_per_class_bars(metrics: dict, split_name: str, out_dir: Path):
    """
    Plot per-class precision / recall / F1 as grouped bars.
    """
    labels = [IDX2LABEL[i] for i in range(4)]
    precisions = []
    recalls = []
    f1s = []

    for lbl in labels:
        pc = metrics["per_class"][lbl]
        precisions.append(pc["precision"])
        recalls.append(pc["recall"])
        f1s.append(pc["f1"])

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, precisions, width, label="Precision")
    ax.bar(x, recalls, width, label="Recall")
    ax.bar(x + width, f1s, width, label="F1")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Per-class Metrics ({split_name.capitalize()})")
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / f"polygon_siamese_{split_name}_per_class.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[plot_metrics] Saved {out_path}")


def plot_overall_metrics(train_metrics: dict, test_metrics: dict, out_dir: Path):
    """
    Plot overall accuracy / macro-F1 / weighted-F1 for train vs test.
    """
    metrics_names = ["accuracy", "macro_f1", "weighted_f1"]
    x = np.arange(len(metrics_names))
    width = 0.35

    train_vals = [train_metrics.get(m, 0.0) for m in metrics_names]
    test_vals = [test_metrics.get(m, 0.0) for m in metrics_names]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, train_vals, width, label="Train")
    ax.bar(x + width / 2, test_vals, width, label="Test")

    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Macro F1", "Weighted F1"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Overall Metrics (Train vs Test)")
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / "polygon_siamese_overall_metrics.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[plot_metrics] Saved {out_path}")


def main():
    figures_dir = PROJECT_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    train_metrics = load_metrics("train")
    test_metrics = load_metrics("test")

    if train_metrics is None or test_metrics is None:
        print("[plot_metrics] Missing metrics JSONs, run eval_polygon_siamese first.")
        return

    # Confusion matrices
    plot_confusion_matrix(train_metrics, "train", figures_dir)
    plot_confusion_matrix(test_metrics, "test", figures_dir)

    # Per-class bars
    plot_per_class_bars(train_metrics, "train", figures_dir)
    plot_per_class_bars(test_metrics, "test", figures_dir)

    # Overall metrics comparison
    plot_overall_metrics(train_metrics, test_metrics, figures_dir)


if __name__ == "__main__":
    main()
