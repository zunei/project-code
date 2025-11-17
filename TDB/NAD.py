
from copy import deepcopy
import time
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, DatasetFolder
np.random.seed(11)
random.seed(11)
torch.manual_seed(11)
torch.cuda.manual_seed(11)


class AT(nn.Module):


    def __init__(self, p):
        super(AT, self).__init__()
        self.p = p

    def forward(self, fm_s, fm_t):
        loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

        return loss

    def attention_map(self, fm, eps=1e-6):
        am = torch.pow(torch.abs(fm), self.p)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2, 3), keepdim=True)
        am = torch.div(am, norm + eps)

        return am


class NAD:

    def __init__(self,
                 model,
                 loss,
                 power,
                 beta=[],
                 target_layers=[],
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(NAD, self).__init__()

        assert len(beta) == len(target_layers), 'The length of beta must equal to the length of target_layers!'
        self.model = model
        self.loss = loss
        self.power = power
        self.beta = beta
        self.target_layers = target_layers

        self.device = device

    def get_model(self):
        return self.model

    def adjust_tune_learning_rate(self, optimizer, epoch):
        if epoch in self.current_schedule['schedule']:
            self.current_schedule['tune_lr'] *= self.current_schedule['gamma']
            for param_group in optimizer.param_groups:
                param_group['tune_lr'] = self.current_schedule['tune_lr']

    def adjust_learning_rate(self, optimizer, epoch):
        if epoch in self.current_schedule['schedule']:
            self.current_schedule['lr'] *= self.current_schedule['gamma']
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.current_schedule['lr']

    def _train(self, dataset, portion, schedule):
        if schedule is None:
            raise AttributeError("Reparing Training schedule is None, please check your schedule setting.")
        elif schedule is not None:
            self.current_schedule = deepcopy(schedule)

        # get a portion of the repairing training dataset
        print("===> Loading {:.1f}% of traing samples.".format(portion * 100))
        idxs = np.random.permutation(len(dataset))[:int(portion * len(dataset))]
        dataset = torch.utils.data.Subset(dataset, idxs)

        train_loader = DataLoader(
            dataset,
            batch_size=self.current_schedule['batch_size'],
            shuffle=True,
            drop_last=False,
            pin_memory=True,
        )
        self.train_loader = train_loader

        teacher_model = deepcopy(self.model)
        teacher_model = teacher_model.to(self.device)
        teacher_model.train()

        t_optimizer = torch.optim.SGD(teacher_model.parameters(), lr=self.current_schedule['tune_lr'],
                                      momentum=self.current_schedule['momentum'],
                                      weight_decay=self.current_schedule['weight_decay'])

        iteration = 0

        for i in range(self.current_schedule['tune_epochs']):
            self.adjust_tune_learning_rate(t_optimizer, i)
            for batch_id, batch in enumerate(train_loader):
                batch_img = batch[0]
                batch_label = batch[1]
                batch_img = batch_img.to(self.device)
                batch_label = batch_label.to(self.device)
                t_optimizer.zero_grad()
                predict_digits = teacher_model(batch_img)
                loss = self.loss(predict_digits, batch_label)
                loss.backward()
                t_optimizer.step()

                iteration += 1



        # Perform NAD and get the repaired model
        for param in teacher_model.parameters():
            param.requires_grad = False
        self.model = self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.current_schedule['lr'],
                                    momentum=self.current_schedule['momentum'],
                                    weight_decay=self.current_schedule['weight_decay'])

        iteration = 0


        criterionAT = AT(self.power)

        for i in range(self.current_schedule['epochs']):
            self.adjust_learning_rate(optimizer, i)
            for batch_id, batch in enumerate(train_loader):
                batch_img = batch[0]
                batch_label = batch[1]
                batch_img = batch_img.to(self.device)
                batch_label = batch_label.to(self.device)
                optimizer.zero_grad()

                container = []

                def forward_hook(module, input, output):
                    container.append(output)

                hook_list = []
                for name, module in self.model._modules.items():
                    if name in self.target_layers:
                        hk = module.register_forward_hook(forward_hook)
                        hook_list.append(hk)

                for name, module in teacher_model._modules.items():
                    if name in self.target_layers:
                        hk = module.register_forward_hook(forward_hook)
                        hook_list.append(hk)

                output_s = self.model(batch_img)
                _ = teacher_model(batch_img)

                for hk in hook_list:
                    hk.remove()
                # print(len(container))
                loss = self.loss(output_s, batch_label)
                for idx in range(len(self.beta)):
                    loss = loss + criterionAT(container[idx], container[idx + len(self.beta)]) * self.beta[idx]

                loss.backward()
                optimizer.step()

                iteration += 1


    def repair(self, dataset, portion, schedule):

        print("===> Start training repaired model...")
        self._train(dataset, portion, schedule)

    # def test(self, dataset, schedule):
    #     """Test repaired curve model on dataset
    #
    #     Args:
    #         dataset (types in support_list): Dataset.
    #         schedule (dict): Schedule for testing.
    #     """
    #     model = self.model
    #     test(model, dataset, schedule)


# schedule = {
#     'batch_size': 64,
#     'lr': 0.01,
#     'momentum': 0.9,
#     'weight_decay': 5e-4,
#     'gamma': 0.1,
#     'tune_lr': 0.01,
#     'tune_epochs': 10,
#     'epochs': 20,
#     'schedule':  [2, 4, 6, 8],
# }
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
# dataset = CIFAR10(root='data', train=True, download=True, transform=transforms.ToTensor())

# defense = NAD(
#     model=model,
#     loss=nn.CrossEntropyLoss(),
#     power=2.0,
#     beta=[500, 500],
#     target_layers=['conv1', 'conv2'],
# )
#
# defense.repair(dataset=dataset, portion=1, schedule=schedule)