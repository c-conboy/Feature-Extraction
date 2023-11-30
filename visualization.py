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
parser.add_argument('-index', type=int, help='[Y/N]', default = 0)
parser.add_argument('-dataset', type=str, default = '../datasets/doggies/images_test')
parser.add_argument('-label', type=str, default = '../datasets/doggies/test_labels.txt')
args = parser.parse_args()
Weights = args.weights
Dataset_Path = args.dataset
Label_Path = args.label
idx = args.index

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

#Get Inference

start = time.time()
output = model_ft(test_dataset[idx][0].unsqueeze(0))
end = time.time()
print(end - start)

center = (int(output[0][0].item()), int(output[0][1].item()))
#Visualize Image
with open(Label_Path, "r") as f:
        labels = []
        for line in f:
            labels.append(line)

img_name = labels[idx].split(',')[0]

imageFile = os.path.join(Dataset_Path, img_name)
if os.path.isfile(imageFile):
    image = cv2.imread(imageFile)
    dim = (int(image.shape[1] / 1), int(image.shape[0] / 1))
    imageScaled = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    cv2.circle(imageScaled, center, 2, (0, 0, 255), 1)
    cv2.circle(imageScaled, center, 8, (0, 255, 0), 1)
    cv2.imshow(imageFile, imageScaled)
    key = cv2.waitKey(0)
    cv2.destroyWindow(imageFile)
    if key == ord('q'):
        exit(0)

