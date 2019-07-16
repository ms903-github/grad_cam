import glob
from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
from preprocess import preprocessor, DatalisttoDataset
from fetch_data import get_data
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from collections import OrderedDict
from models import Net_google

batch_size = 100
num_epoch = 50
data, label = get_data(10, num_ratio = 10, domain = "target", mode = "choiced", data_aug = True)
print("preprocess finished")
dataset = DatalisttoDataset(data, label, transform = None)
#train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)])
train_dataset = dataset
data_te, label_te = get_data(10, num_ratio = 10, domain = "target", mode = "processed", data_aug = False)
test_dataset = DatalisttoDataset(data_te, label_te)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#device = "cpu"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net_google().to(device)
model = Net_google()

#転移学習するときはここを使う
model.load_state_dict(torch.load("net.model"))
#checkpoint = torch.load("net.model")
#state_dict = checkpoint
#new_state_dict = OrderedDict()
#for k, v in state_dict.items():
#    name = k[7:]
#    new_state_dict[name] = v
#net_g.load_state_dict(new_state_dict)

optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
loss_func = torch.nn.CrossEntropyLoss()
for epoch in range(num_epoch):
    for data, label in train_loader:
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        pred = net(data)
        loss = loss_func(pred, label)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print("epoch {} has finished".format(epoch))
        with torch.no_grad():
            acc = 0
            total = 0
            for te_data, te_label in test_loader:
                te_data, te_label = te_data.to(device), te_label.to(device)
                output = net(te_data)
                pred = torch.argmax(output, dim = 1)
                acc += (pred == te_label).sum().item() / len(te_label)
        acc = acc / len(test_loader)
        print("accuracy in test:{}%".format(acc*100))
torch.save(net.state_dict(), "net.model")
