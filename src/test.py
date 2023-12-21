#Import Modules
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
import math
import statistics 


#Parse Input Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-weights', type=str, default = None)
parser.add_argument('-cuda', type=str, help='[Y/N]', default = 'N')
parser.add_argument('-dataset', type=str, default = '../datasets/doggies/images_test')
parser.add_argument('-label', type=str, default = '../datasets/doggies/test_labels.txt')
args = parser.parse_args()
Weights = args.weights
Dataset_Path = args.dataset
Label_Path = args.label

#Set Device
device = torch.device("cuda" if torch.cuda.is_available() and args.cuda == 'Y' else "cpu")

#Load Data
data_dir = Dataset_Path
label_file = Label_Path
test_dataset = DoggyDataset(label_file, data_dir)
test_dataloader = torch.utils.data.DataLoader(test_dataset)

#Load Model
model_ft = models.resnet18()
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
if(Weights):
        model_ft.load_state_dict(torch.load(Weights, map_location=torch.device(device)))
model_ft = model_ft.to(device)

#For each test image get inference
distances = []
for img, label in test_dataloader:
    output = model_ft(img)
    e_dist = math.dist(output[0].detach().numpy(), label.numpy()[0])
    distances += [e_dist]

#Get Statistical Data for lists
avg_dist = sum(distances)/len(distances)
std = statistics.pstdev(distances) 
max = max(distances)
min = min(distances)

print("Average:")
print(avg_dist)

print("STD:")
print(std)

print("Max:")
print(max)

print("Min:")
print(min)