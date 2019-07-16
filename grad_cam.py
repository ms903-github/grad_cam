import pandas as pd
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader
from preprocess import DatalisttoDataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from collections import OrderedDict
from models import Net_google
from fetch_data import get_data

batch_size = 100
num_epoch = 50
data, label = get_data(10, num_ratio = 10, domain = "source", mode = "processed", data_aug = False)
print("preprocess finished")
dataset = DatalisttoDataset(data, label, transform = None)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
device = "cpu"
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Net_google()
model.load_state_dict(torch.load("net.model"))
#checkpoint = torch.load("net.model")
#state_dict = checkpoint
#new_state_dict = OrderedDict()
#for k, v in state_dict.items():
#    name = k[7:]
#    new_state_dict[name] = v
#net_g.load_state_dict(new_state_dict)
model.eval()

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

feature_fn = torch.nn.Sequential(*list(model.children())[:-3]).cpu().eval()
#classifier_fn = torch.nn.Sequential(*(list(model.children())[-3:] + [Flatten()] + list(model.children())[-1:])).eval() #これだと何故か順番が狂う
layer = []
layer.append(list(model.children())[-3])
layer.append(list(model.children())[-2])
layer.append(Flatten())
layer.append(list(model.children())[-1])
classifier_fn = torch.nn.Sequential(*layer).eval()

def GradCam(img, c, feature_fn, classifier_fn):
    feats = feature_fn(img.cpu())
    _, N, H, W = feats.size()
    print(list(classifier_fn.children()))
    out = classifier_fn(feats)
    c_score = out[0, c]
    print(c_score)
    grads = torch.autograd.grad(c_score, feats)
    w = grads[0][0].mean(-1).mean(-1)
    sal = torch.matmul(w, feats.view(N, H*W))
    sal = F.relu(sal)
    sal = sal.view(H, W).cpu().detach().numpy()
    sal = np.maximum(sal, 0)
    return sal

input_index = 15
input_data = test_loader.dataset[input_index][0]
input_data = input_data.view(1, input_data.shape[0], input_data.shape[1], input_data.shape[2]).cpu()

pp, cc = torch.topk(nn.Softmax(dim=1)(model(input_data)), 2)

sal = GradCam(input_data.cpu(), cc[0][0], feature_fn, classifier_fn)

img = input_data.permute(0, 2, 3, 1).view(input_data.shape[2], input_data.shape[3], input_data.shape[1]).cpu().numpy()
img_sal = Image.fromarray(sal).resize(img.shape[0:2], resample=Image.LINEAR)


plt.imshow(np.array(img_sal), alpha=0.5, cmap='jet')
plt.colorbar()
