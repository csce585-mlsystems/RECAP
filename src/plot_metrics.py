"""
plot_metrics.py
Purpose:
  Load saved metrics + prediction CSVs for the polygon-Siamese model
  and generate plots for Train and Test:
    - confusion matrix heatmaps
    - per-class F1 bar charts
    - class support bar charts
    - confidence histograms per class

Outputs:
  artifacts/plots/
    cm_train.png
    cm_test.png
    f1_per_class_train.png
    f1_per_class_test.png
    support_per_class_train.png
    support_per_class_test.png
    conf_hist_train.png
    conf_hist_test.png
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .common import PROJECT_ROOT, IDX2LABEL


PLOT_DIR = PROJECT_ROOT / "artifacts" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(cm, split_name: str):
    labels = [IDX2LABEL[i] for i in range(4)]
    cm = np.array(cm, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"Confusion Matrix ({split_name})",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # write counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            ax.text(j, i, str(val), ha="center", va="center", color="white" if val > cm.max() / 2 else "black")

    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"cm_{split_name.lower()}.png", dpi=200)
    plt.close(fig)


def plot_f1_bar(per_class: dict, split_name: str):
    labels = []
    f1_vals = []
    for lab, stats in per_class.items():
        labels.append(lab)
        f1_vals.append(stats["f1"])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, f1_vals)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-Class F1 ({split_name})")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"f1_per_class_{split_name.lower()}.png", dpi=200)
    plt.close(fig)


def plot_support_bar(per_class: dict, split_name: str):
    labels = []
    supports = []
    for lab, stats in per_class.items():
        labels.append(lab)
        supports.append(stats["support"])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, supports)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Number of buildings")
    ax.set_title(f"Per-Class Support ({split_name})")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"support_per_class_{split_name.lower()}.png", dpi=200)
    plt.close(fig)


def plot_confidence_hist(csv_path: Path, split_name: str):
    df = pd.read_csv(csv_path)
    # overall
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["confidence"], bins=20)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Count")
    ax.set_title(f"Confidence Histogram (All classes, {split_name})")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"conf_hist_{split_name.lower()}.png", dpi=200)
    plt.close(fig)


def main():
    # Train metrics
    for split_name in ["train", "test"]:
        metrics_path = PROJECT_ROOT / "metrics" / f"polygon_siamese_{split_name}_eval.json"
        preds_csv = PROJECT_ROOT / "metrics" / f"polygon_siamese_{split_name}_predictions.csv"

        if not metrics_path.exists():
            print(f"Missing metrics JSON for {split_name}: {metrics_path}")
            continue

        with metrics_path.open("r") as f:
            metrics = json.load(f)

        if not metrics:
            print(f"No metrics content for {split_name}")
            continue

        print(f"\n[Plots] Split = {split_name}")
        # confusion matrix
        if "confusion_matrix" in metrics:
            plot_confusion_matrix(metrics["confusion_matrix"], split_name.capitalize())

        # per-class F1 + support
        if "per_class" in metrics:
            plot_f1_bar(metrics["per_class"], split_name.capitalize())
            plot_support_bar(metrics["per_class"], split_name.capitalize())

        # confidence histogram (from CSV)
        if preds_csv.exists():
            plot_confidence_hist(preds_csv, split_name.capitalize())
        else:
            print(f"Missing predictions CSV for {split_name}: {preds_csv}")


if __name__ == "__main__":
    main()
