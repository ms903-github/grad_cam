import glob
from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
from preprocess import DatalisttoDataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

data_transform = transforms.Compose([
    transforms.RandomRotation(180),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop((150, 150), scale=(0.9, 1.0)),
    transforms.ToTensor()
])
def get_data(class_num, num_ratio = 10, domain = "source", mode = "processed", data_aug = False):
    datalist = []
    labellist = []
    for i in range(1, class_num+1):
        path_list = glob.glob("data/{}_{}/{}/*.jpg".format(domain, mode, i))
        for path in path_list:
            img = Image.open(path)
            if data_aug:           
                for j in range(num_ratio):
                    img_tensor = data_transform(img)
                    datalist.append(img_tensor)
                    labellist.append(i-1)
            else:
                img_tensor = transforms.ToTensor()(img)
                datalist.append(img_tensor)
                labellist.append(i-1)
            img.close()
    return datalist, labellist
