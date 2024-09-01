# 3D Segmentation Model for CT Abdomen Organs

![organs](https://github.com/user-attachments/assets/13e3b9a3-b61e-4b3f-b5a2-f9f687d0bb11)


## Overview

This project involves the development of a 3D segmentation model to accurately segment and identify specific abdominal organs (Liver, Right Kidney, Left Kidney, and Spleen) from CT scans. The model is built using a VNet architecture and is trained on a public dataset of abdominal CT scans. The primary goal is to assist in medical imaging by automating the segmentation process, which can aid in disease diagnosis, surgical planning, and treatment monitoring.

## Setup Instructions

### Prerequisites

Ensure that you have Python 3.8 or higher installed. You will also need to install the following Python libraries:

pip install torch torchvision nibabel numpy matplotlib scikit-image monai

### Model Architecture

The model used in this project is based on the VNet architecture, which is specifically designed for 3D medical image segmentation. The key features of the model include:

Input: Single-channel (grayscale) 3D CT scans.
Output: 4-channel output representing the background, Liver, Right Kidney, Left Kidney, and Spleen.
Layers: The VNet is composed of several layers of 3D convolutions, followed by ReLU activations and downsampling/upscaling operations.

### Training Process

The training process includes the following steps:

Data Loading: The CT scans and labels are loaded and preprocessed using a custom dataset class.
Loss Function: The model is trained using the Dice Loss, which is effective for segmentation tasks.
Optimizer: The Adam optimizer is used to update model weights during training.
Metrics: The performance is evaluated using the Dice Score for each organ separately.

Example command :  python scripts/train.py


### Validation and Inference
Validation: The model’s performance is validated using a separate validation dataset, and the Dice score is computed for each organ.
Inference: After training, the model is used to generate segmentation masks for unseen CT scans.

Example command : python scripts/inference.py


### 3D Visualization





