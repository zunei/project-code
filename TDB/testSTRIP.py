import datetime
import logging
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from adba.helpers.utils import add_trigger, test_trigger_accuracy
from adba.models.resnet import resnet18

# log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
# log_path = log_file_name + '.log'
# logging.basicConfig(
#     filename=os.path.join('./log/', log_path),
#     format='%(asctime)s %(levelname)-8s %(message)s',
#     datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# logger.info("#!" * 20)
# mask = torch.load('./saved/adba_client_model_weights/mask_datasetcifar10_clients1_poison1_partitioniid_betapartition0.5_betabackdoor0.3_alphabackdoor0.4.pt')
# trigger = torch.load('./saved/adba_client_model_weights/pattern_datasetcifar10_clients1_poison1_partitioniid_betapartition0.5_betabackdoor0.3_alphabackdoor0.4.pt')

canshu=torch.load('/home/main/tzz/adba/saved/clean/mytestcifar10_clean_old.pkl')
model=  resnet18(num_classes=10).cuda()
model.load_state_dict(canshu)

train_dataset = CIFAR10('./dataset', train=True, download=True,
                        transform=transforms.Compose(
                                     [
                                         transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                     ]))
train_loader=DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataset = CIFAR10('./dataset', train=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                ]))
test_loader=  torch.utils.data.DataLoader(test_dataset, batch_size=64,
                                              shuffle=False)
mask = torch.load('/home/main/tzz/adba/saved/adba_client_model_weights/mask_datasetcifar10_clients1_poison1_partitioniid_betapartition0.5_betabackdoor0.3_alphabackdoor0.4.pt')
trigger = torch.load('/home/main/tzz/adba/saved/adba_client_model_weights/pattern_datasetcifar10_clients1_poison1_partitioniid_betapartition0.5_betabackdoor0.3_alphabackdoor0.4.pt')
#

trigger_acc = test_trigger_accuracy(test_loader, model, 0, mask, trigger)




# print(len(train_loader))
# x_train=[]
# y_train=[]
# for batch_idx, (images, labels) in enumerate(train_loader):
#     x_train.append(images)
#     y_train.append(labels)
# def superimpose(background, overlay):
#   added_image = cv2.addWeighted(background,1,overlay,1,0)
#   return (added_image.reshape(32,32,3))
#
# def entropyCal(background, n):
#   entropy_sum = [0] * n
#   x1_add = [0] * n
#   index_overlay = np.random.randint(40000,49999, size=n)
#   for x in range(n):
#     x1_add[x] = (superimpose(background, x_train[index_overlay[x]]))
#
#   py1_add = model(np.array(x1_add))
#   EntropySum = -np.nansum(py1_add*np.log2(py1_add))
#   return EntropySum
# print(len(x_train))
# n_test = 2000
# n_sample = 100
# entropy_benigh = [0] * n_test
# entropy_trojan = [0] * n_test
# # x_poison = [0] * n_test
#
# for j in range(n_test):
#   if 0 == j%1000:
#     print(j)
#   x_background = x_train[j+26000]
#   entropy_benigh[j] = entropyCal(x_background, n_sample)
#
# for j in range(n_test):
#   if 0 == j%1000:
#     print(j)
#   x_poison = add_trigger(x_train[j+14000],mask=mask,trigger=trigger)
#   entropy_trojan[j] = entropyCal(x_poison, n_sample)
#
# entropy_benigh = [x / n_sample for x in entropy_benigh] # get entropy for 2000 clean inputs
# entropy_trojan = [x / n_sample for x in entropy_trojan]
# print(entropy_benigh)