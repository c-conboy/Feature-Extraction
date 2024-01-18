import os
import fnmatch
import torch
from torch.utils.data import Dataset
from torchvision import transforms, models
import cv2

data_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

label_transfrom = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


class DoggyDataset(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=data_transform, target_transform=None
    ):
        labels = []
        with open(annotations_file, "r") as f:
            for line in f:
                labels.append(line)
        self.img_labels = labels
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels[idx].split(",")[0]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        coords = self.img_labels[idx].split("(")[1]
        x = coords.split(",")[0].strip()
        y = coords.split(",")[1].split(")")[0].strip()
        label = torch.tensor([float(x), float(y)])

        if self.transform:
            try:
                image = self.transform(image)
            except:
                print(img_path)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __next__(self):
        if self.num >= self.max:
            raise StopIteration
        else:
            self.num += 1
            return self.__getitem__(self.num - 1)

    class_label = {"NoCar": 0, "Car": 1}