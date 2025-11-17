'''
This is the implement of pruning proposed in [1].
[1] Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks. RAID, 2018.
'''

import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

random.seed(11)
torch.manual_seed(11)
torch.cuda.manual_seed(11)


class MaskedLayer(nn.Module):
    def __init__(self, base, mask):
        super(MaskedLayer, self).__init__()
        self.base = base
        self.mask = mask

    def forward(self, input):
        return self.base(input) * self.mask


class Pruning():

    def __init__(self,
                 train_dataset=None,
                 model=None,
                 layer=None,
                 prune_rate=None,
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(Pruning, self).__init__()

        self.train_dataset = train_dataset
        self.model = model
        self.layer = layer
        self.prune_rate = prune_rate
        self.device = device

    def repair(self, batch_size=64):
        model = self.model.to(self.device)
        layer_to_prune = self.layer
        tr_loader = DataLoader(self.train_dataset, batch_size=batch_size,
                               drop_last=True, pin_memory=True)
        prune_rate = self.prune_rate


        print("======== pruning... ========")
        with torch.no_grad():
            container = []

            def forward_hook(module, input, output):


                container.append(output)

            hook = getattr(model, layer_to_prune).register_forward_hook(forward_hook)
            print("Forwarding all training set")

            model.eval()
            for data, _ in tr_loader:
                model(data.cuda())
                del data
            hook.remove()

        container = torch.cat(container, dim=0)
        activation = torch.mean(container, dim=[0, 2, 3])
        seq_sort = torch.argsort(activation)
        num_channels = len(activation)
        prunned_channels = int(num_channels * prune_rate)
        mask = torch.ones(num_channels).cuda()
        for element in seq_sort[:prunned_channels]:
            mask[element] = 0
        if len(container.shape) == 4:
            mask = mask.reshape(1, -1, 1, 1)
        setattr(model, layer_to_prune, MaskedLayer(getattr(model, layer_to_prune), mask))

        self.model = model
        print("======== pruning complete ========")
        if isinstance(container,torch.Tensor):
            container = container.cpu().numpy().tolist()
    def get_model(self):
        return self.model


# class LetNet(nn.Module):
#     def __init__(self):
#         super(LetNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, 5)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(16, 32, 5)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(32*5*5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool1(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool2(x)
#         x = x.view(-1, 32*5*5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# model=LetNet().cuda()
#
# dataset = CIFAR10(root='data', train=True, download=True, transform=transforms.ToTensor())
# pruning = Pruning(
#         train_dataset=dataset,
#         model=model,
#         layer='conv1',
#         prune_rate=1.0,
#     )
# pruning.repair(batch_size=64)
#
# model=pruning.get_model()

