import torch
import matplotlib.pyplot as plt

from dataset import CTScanDataset
from model import SegmentationModel
images_dir = 'data/image' #Enter the path to the image folder
labels_dir = 'data/labels' #Enter the path to the label folder

# Create datasets and loaders
train_dataset = CTScanDataset(images_dir, labels_dir)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

val_dataset = CTScanDataset(images_dir, labels_dir)  # Ideally, you should have a separate validation set
val_loader = DataLoader(val_dataset, batch_size=2)

# Initialize model
model = SegmentationModel()

# Train the model
trained_model = train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001)

# Save the model
torch.save(trained_model.state_dict(), 'outputs/checkpoints/3d_segmentation_model.pth')

# Visualize the predictions
visualize_predictions(trained_model, val_dataset, index=0'
labels_dir = '/Users/apple/3D_Segmentation_Model/data/labels'

# Create datasets and loaders
train_dataset = CTScanDataset(images_dir, labels_dir)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

val_dataset = CTScanDataset(images_dir, labels_dir)  # Ideally, you should have a separate validation set
val_loader = DataLoader(val_dataset, batch_size=2)

# Initialize model
model = SegmentationModel()

# Train the model
trained_model = train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001)

# Save the model
torch.save(trained_model.state_dict(), 'outputs/checkpoints/3d_segmentation_model.pth')

# Visualize the predictions
visualize_predictions(trained_model, val_dataset, index=0)
