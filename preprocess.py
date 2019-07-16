#ラベルごとのフォルダからピクセル情報とラベルの値を持った配列を返す
import glob
from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def preprocessor(class_num, domain, size = 150):   
    datalist = []
    labellist = []         
    for label in range(class_num):
        path_list = glob.glob("data/{}/{}/hoge*".format(domain, label))
        for i, path in enumerate(path_list):
            img = Image.open(path)
            width, height = img.size
            sum0, sum1, sum2 = 0, 0, 0
            for x in range(img.width):
                sum0 += img.getpixel((x,0))[0]
                sum0 += img.getpixel((x, img.height-1))[0]
                sum1 += img.getpixel((x,0))[1]
                sum1 += img.getpixel((x, img.height-1))[1]
                sum2 += img.getpixel((x,0))[2]
                sum2 += img.getpixel((x, img.height-1))[2]
            av0 = sum0 // (2*img.width)
            av1 = sum1 // (2*img.width)
            av2 = sum2 // (2*img.width)
            img2 = Image.new("RGB", (size, size), (av0, av1, av2))
            img2.paste(img, (size // 2 - (width // 2), size // 2 - (height // 2)))
            img2.save("data/{}_processed/{}/hoge{}.jpg".format(domain, label, i))
            datalist.append(img2)
            labellist.append(label)
    return(datalist, labellist)


class DatalisttoDataset(Dataset):
    def __init__(self, dataset, labels, transform=None):
        self.transform = transform
        self.data_num = len(dataset)
        self.dataset = dataset
        self.labels = labels
        
    def __len__(self):
        return self.data_num
    def __getitem__(self, i):
        out_data = self.dataset[i]
        if self.transform:
            out_data = self.transform(out_data)
        out_label = self.labels[i]
        return out_data, out_label
       
preprocessor(111, "source", size = 224)