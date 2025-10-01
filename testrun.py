# test_dml.py
import torch
try:
    import torch_directml
    dml = torch_directml.device()
    device = torch.device(dml)
    print("Using DirectML device:", dml)
except Exception as e:
    print("DirectML import failed:", e)
    device = torch.device("cpu")

# quick test: small model forward/backward
from torchvision import models
import time

model = models.resnet18(weights=None)
model.conv1 = torch.nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = torch.nn.Linear(model.fc.in_features, 4)
model.to(device)

# dummy batch: batch=8, 6 channels, 224x224
x = torch.randn(8, 6, 224, 224, device=device)
y = torch.randint(0, 4, (8,), device=device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

t0 = time.time()
out = model(x)
loss = criterion(out, y)
loss.backward()
optimizer.step()
torch.cuda = None  # no-op but ensures code won't rely on CUDA APIs
t1 = time.time()
print("One forward+backward on device took {:.3f}s".format(t1 - t0))
