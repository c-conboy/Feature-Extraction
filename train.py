#Import Modules
# Data augmentation and normalization for training
# Just normalization for validation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from dataset import DoggyDataset
import cv2
import argparse

data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

label_transfrom = transforms.Compose([
        transforms.ToTensor(),
])

#Parse Input Arguments

#Load Data
data_dir = '../datasets/Doggies/images'
label_file = '../datasets/Doggies/train_noses.3.txt'
train_dataset = DoggyDataset(label_file, data_dir, transform=data_transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, 4, shuffle=True)

#Load Model

#Load Optimizer, Loss function

#Run train for Epoch

image = train_dataset[0]
print(image)