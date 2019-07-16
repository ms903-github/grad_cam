from torch.utils.data import Dataset, DataLoader
#from preprocess import preprocessor, DatalisttoDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models 
from collections import OrderedDict

googlenet = models.googlenet(pretrained = True)

class Net_google(nn.Module):
    def __init__(self):
        super(Net_google, self).__init__()
        self.feature_fn = nn.Sequential(*list(googlenet.children())[:-3])
        #self.classifier_fn = nn.Sequential(*list(googlenet.children())[-3:])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(1024, 10)
    def forward(self, x):
        #print(x.size())
        x = self.feature_fn(x)
        #print(x.size())
        #x = self.classifier_fn(x)
        x = self.avgpool(x)
        #print(x.size())
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = self.fc1(x)
        #print(x.size())
        return x
#test = Net_google()
#print(list(test.children())[-3:])