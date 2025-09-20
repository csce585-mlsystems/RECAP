from recap.datasets.xview2_dataset import XView2Dataset
from torch.utils.data import DataLoader

# Path to your training index
index_file = "info/index.csv"

# Create dataset
dataset = XView2Dataset(index_file, chip_size=224)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Fetch one batch
for imgs, labels in dataloader:
    print("Batch images:", imgs.shape)   
    print("Batch labels:", labels)       
    break
