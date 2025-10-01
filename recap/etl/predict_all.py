import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models
from recap.datasets.xview2_dataset import XView2Dataset
from tqdm import tqdm

# --------------------------
# Config
# --------------------------
# Use subset first for testing; switch to index.csv later
INDEX_FILE = "info/index_subset.csv"
MODEL_PATH = "recap/models/baseline_resnet18_augmented_logged.pth"
OUT_FILE = "info/predictions.csv"
BATCH_SIZE = 16
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Dataset & Loader
# --------------------------
dataset = XView2Dataset(INDEX_FILE, chip_size=224)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# --------------------------
# Load Model
# --------------------------
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

# --------------------------
# Run inference with progress bar
# --------------------------
all_preds = []
all_confs = []

with torch.no_grad():
    for imgs, _ in tqdm(loader, desc="Running inference", unit="batch"):
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)

        confs, preds = torch.max(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_confs.extend(confs.cpu().numpy())

# --------------------------
# Save predictions
# --------------------------
df = pd.read_csv(INDEX_FILE)
df["label_pred"] = all_preds
df["label_conf"] = all_confs

# Map numeric preds to class names
id2label = {0: "no-damage", 1: "minor-damage", 2: "major-damage", 3: "destroyed"}
df["label_name"] = df["label_pred"].map(id2label)

df.to_csv(OUT_FILE, index=False)
print(f"✅ Predictions saved to {OUT_FILE}")
