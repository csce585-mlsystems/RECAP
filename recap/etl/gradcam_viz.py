import sys, os, random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
from torchvision import models
from torchcam.methods import SmoothGradCAMpp
from recap.datasets.xview2_dataset import XView2Dataset
import matplotlib.pyplot as plt
import cv2

# --------------------------
# Config
# --------------------------
NUM_CLASSES = 4
MODEL_PATH = "recap/models/baseline_resnet18_augmented_logged.pth"  # update if needed
INDEX_FILE = "info/index_subset.csv"  # or "info/index.csv"
SAVE_DIR = "info/gradcam_examples"
os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------------
# Load model
# --------------------------
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# --------------------------
# Load dataset and pick random sample
# --------------------------
dataset = XView2Dataset(INDEX_FILE, chip_size=224)
idx = random.randint(0, len(dataset)-1)
img, label = dataset[idx]
img = img.unsqueeze(0)

print(f"Picked building #{idx} (True label: {label})")

# --------------------------
# Grad-CAM
# --------------------------
cam_extractor = SmoothGradCAMpp(model, target_layer="layer4")
out = model(img)
pred_class = out.argmax(dim=1).item()

activation_map = cam_extractor(pred_class, out)[0][0].detach().numpy()

# --------------------------
# Convert to numpy images
# --------------------------
img_np = img.squeeze().permute(1,2,0).detach().numpy()
pre_img = img_np[:, :, :3]
post_img = img_np[:, :, 3:]

# Normalize to [0,255]
def norm255(x):
    return ((x - x.min()) / (x.max() - x.min()) * 255).astype("uint8")

pre_img = norm255(pre_img)
post_img = norm255(post_img)

# Resize CAM
activation_map = cv2.resize(activation_map, (pre_img.shape[1], pre_img.shape[0]))
activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())

# Heatmap
heatmap = cv2.applyColorMap((activation_map*255).astype("uint8"), cv2.COLORMAP_JET)
overlay_pre = cv2.addWeighted(pre_img, 0.6, heatmap, 0.4, 0)
overlay_post = cv2.addWeighted(post_img, 0.6, heatmap, 0.4, 0)

# --------------------------
# Save results
# --------------------------
save_path = os.path.join(SAVE_DIR, f"building_{idx}_pred{pred_class}_true{label}.png")

plt.figure(figsize=(12,6))
plt.subplot(2,2,1)
plt.imshow(pre_img)
plt.title("Pre-disaster")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(post_img)
plt.title("Post-disaster")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(overlay_pre)
plt.title(f"Grad-CAM (Pre) - Pred: {pred_class}, True: {label}")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(overlay_post)
plt.title("Grad-CAM (Post)")
plt.axis("off")

plt.tight_layout()
plt.savefig(save_path)
plt.show()

print(f"✅ Saved Grad-CAM example to {save_path}")
