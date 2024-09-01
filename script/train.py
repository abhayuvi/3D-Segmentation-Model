import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
import matplotlib.pyplot as plt


from dataset import CTScanDataset
from model import SegmentationModel
def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    criterion = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)}")
        
        model.eval()
        with torch.no_grad():
            dice_scores = []
            for images, labels in val_loader:
                outputs = model(images)
                dice_scores.append(dice_metric(outputs, labels))
            
            print(f"Validation Dice Score: {torch.mean(torch.stack(dice_scores)).item()}")
    
    return model
