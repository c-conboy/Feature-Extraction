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
parser = argparse.ArgumentParser()
parser.add_argument('-weights', type=str, default = None)
parser.add_argument('-cuda', type=str, help='[Y/N]', default = 'N')
parser.add_argument('-dataset', type=str, default = '../datasets/doggies/images')
parser.add_argument('-label', type=str, default = '../datasets/doggies/train_noses.3.txt')
parser.add_argument('-b', type=int, default = 4)
parser.add_argument('-e', type=int, default = 1)

args = parser.parse_args()
Epochs = args.e
Batches = args.b
Cuda = args.cuda
Weights = args.weights
Dataset_Path = args.dataset
Label_Path = args.label

#Set Device
device = torch.device("cuda" if torch.cuda.is_available() and args.cuda == 'Y' else "cpu")

#Load Data
data_dir = Dataset_Path
label_file = Label_Path
train_dataset = DoggyDataset(label_file, data_dir, transform=data_transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, Batches, shuffle=True)

#Load Model
if(Weights):
        model_ft = models.resnet18()
        model_ft.load_state_dict(torch.load(Weights, map_location=torch.device(device)))
else:
        model_ft = models.resnet18()

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)

#Load Optimizer, Loss function
criterion = nn.MSELoss()
optimizer = optim.Adam(model_ft.parameters(), lr=0.0001)


#Run train for Epoch
loss_at_epoch = torch.zeros(Epochs)
for epoch in range(Epochs):
        running_loss_over_epoch = 0
        for imgs, labels in train_dataloader:
                model_ft.train()  # Set model to training mode
                imgs = imgs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                        outputs = model_ft(imgs)            
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        running_loss_over_epoch += loss.item()
        loss_at_epoch += [running_loss_over_epoch/len(train_dataset)]
        print(loss_at_epoch)

torch.save(model_ft.state_dict(), 'model.pth')
plt.plot(loss_at_epoch)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig('plot.png')

        


        
