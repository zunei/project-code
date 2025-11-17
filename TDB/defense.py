import copy
import datetime
import logging
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, SVHN, MNIST
from tqdm import tqdm

from adba.Fine_Pruning import Pruning
from adba.NAD import NAD
from adba.NC import NeuralCleanseConfig, NeuralCleanse
from adba.STRIP import STRIP
from adba.helpers.utils import add_trigger, test, test_trigger_accuracy, KLDiv
from adba.main import Ensemble, kd_train
from adba.models.nets import CNNMnist
from adba.models.resnet import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####load log
log_file_name = '22_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
log_path = log_file_name + '.log'
logging.basicConfig(
    filename=os.path.join('./log/', log_path),
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.info("#" * 20)

#####load model
# local_weights = torch.load('saved/adba_client_model_weights/datasetcifar10_clients1_poison1_partitioniid_betapartition0.5_betabackdoor0.3_alphabackdoor0.4.pkl')
# model_list = []
# mymodel=  resnet18(num_classes=10).cuda()
# # print(local_weights)
# for i in range(len(local_weights)):
#     net = copy.deepcopy(mymodel)
#     net.load_state_dict(local_weights[i])
#     model_list.append(net)
# model = Ensemble(model_list)
# canshu=torch.load('saved/mytestcifar10_ttt.pkl')
# model=  resnet18(num_classes=10).cuda()
# model.load_state_dict(local_weights[-1])

canshu=torch.load('./saved/mytestcifar10_ttt.pkl')
model=  resnet18(num_classes=10).cuda()
# model=CNNMnist().cuda()
model.load_state_dict(canshu)

##loda dataset
data_dir = './dataset'

train_dataset = CIFAR10(data_dir, train=True, download=True,
                                 transform=transforms.Compose(
                                     [
                                         transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                     ]))
test_dataset = CIFAR10(data_dir, train=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                ]))
# train_dataset = SVHN(data_dir, split="train", download=True,
#                               transform=transforms.Compose(
#                                   [transforms.ToTensor()]))
# test_dataset = SVHN(data_dir, split="test", download=True,
#                              transform=transforms.Compose([
#                                  transforms.ToTensor(),
#                              ]))
# train_dataset = MNIST(data_dir, train=True, download=True,
#                                transform=transforms.Compose(
#                                    [transforms.ToTensor()]))
# test_dataset = MNIST(data_dir, train=False, download=True,
#                               transform=transforms.Compose([
#                                   transforms.ToTensor(),
#                               ]))
train_loader=  torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                              shuffle=True)
test_loader=  torch.utils.data.DataLoader(test_dataset, batch_size=64,
                                              shuffle=False)
acc, test_loss = test(model, test_loader)

# # STRIP
# STRIP_model = copy.deepcopy(model)
# train_dataset1=Subset(train_dataset, [i for i in range(1000)])
# train_dataset2=Subset(train_dataset, [i for i in range(1000,2000)])
# data_train1 = torch.stack([train_dataset1[i][0] for i in range(len(train_dataset1))])
# data_train1 = data_train1.permute(0, 2, 3, 1)
# data_train1 = np.array(data_train1)
# strip = STRIP(data_train1, STRIP_model)
# range_bins = [0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5,0.525,0.55,0.575,0.6,0.625,0.65,0.675,0.7,0.725,0.75]
# def entropy_range_general(range_bins, entropy_num, entropy):
#
#     # Check which bin the entropy falls into
#     for i in range(len(range_bins) - 1):
#         if range_bins[i] <= entropy < range_bins[i + 1]:
#             entropy_num[i] += 1
#             return entropy_num
#
#     # If entropy is outside the defined bins, handle it separately
#     print("flow=====================================================================")
#     print(entropy)
#     entropy_num[len(range_bins) - 1] += 1
#     print("flow=====================================================================")
#
#     return entropy_num
# num0 = 0
# num1 = 0  # 1: poisoned
# entropy_num = [0] * 1000
#
mask = torch.load('./saved/adba_client_model_weights/mask_datasetcifar10_clients1_poison1_partitioniid_betapartition0.5_betabackdoor0.3_alphabackdoor0.4.pt')
trigger = torch.load('./saved/adba_client_model_weights/pattern_datasetcifar10_clients1_poison1_partitioniid_betapartition0.5_betabackdoor0.3_alphabackdoor0.4.pt')
#
trigger_acc = test_trigger_accuracy(test_loader, model, 0, mask, trigger)
# for i in range(1000):
#     x = train_dataset1[i][0]# 这里的数据集是要检测的数据集
#     x=add_trigger(x,mask,trigger).permute(1, 2, 0)
#     is_poisoned, entropy = strip.detect_backdoor(x)
#     # write_entropy_to_excel(entropy, i)
#     # print("+++++++++++"entropy)
#     # entropy_num = entropy_range_general(range_bins,entropy_num, entropy)
#     entropy_num[i]=entropy
#     print(entropy_num)
#     if is_poisoned == 0:
#         num0 += 1
#     if is_poisoned == 1:  # 1: poisoned
#         num1 += 1
# print(f"benign num: {num0}")
# print(f"poisoned num:{num1}")
# logger.info(f"STRIP poisoned dataset's benign num: {num0}")
# logger.info(f"STRIP poisoned dataset's poisoned num: {num1}")
# logger.info(f"STRIP poisoned dataset's entropy: {entropy_num}")
# logger.info("#" * 20)
#
#
# # 检测良性数据集
# num0 = 0
# num1 = 0  # 1: poisoned
# entropy_num1 = [0] * 1000
# for j in range(1000):
#     x = train_dataset2[j][0].permute(1, 2, 0)  # 这里的数据集是要检测的数据集
#     is_poisoned, entropy = strip.detect_backdoor(x)
#     # write_entropy_to_excel(entropy, i+1000)
#     # entropy_num1 = entropy_range_general(range_bins,entropy_num1, entropy)
#     entropy_num1[j] = entropy
#     print(entropy_num1)
#     if is_poisoned == 0:
#         num0 += 1
#     if is_poisoned == 1:  # 1: poisoned
#         num1 += 1
# print(f"benign num: {num0}")
# print(f"poisoned num:{num1}")
# logger.info(f"STRIP benign dataset's benign num: {num0}")
# logger.info(f"STRIP benign dataset's poisoned num: {num1}")
# logger.info(f"STRIP benign dataset's entropy: {entropy_num1}")
# logger.info("#" * 20)

###NC
#
# NC_model = copy.deepcopy(model)
# cfg=NeuralCleanseConfig()
# NC=NeuralCleanse(model=NC_model, transform=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), input_shape=[3,32,32], dataset=test_dataset, cfg=cfg, device=device)
# NC.detect()
# logger.info("#" * 20)



# bins = 30
# entropy_benigh = [x / 100 for x in entropy_num1] # get entropy for 2000 clean inputs
# entropy_trojan = [x / 100 for x in entropy_num]
# plt.hist(entropy_benigh, bins, edgecolor='white',  weights=np.ones(len(entropy_benigh)) / len(entropy_benigh),alpha=1, label='without trigger')
# plt.hist(entropy_trojan, bins, edgecolor='white', weights=np.ones(len(entropy_trojan)) / len(entropy_trojan), alpha=1, label='with trigger')
# plt.legend(loc='upper right', fontsize = 20)
# plt.ylabel('Probability (%)', fontsize = 20)
# plt.title('normalized entropy', fontsize = 20)
# plt.tick_params(labelsize=20)
#
# fig1 = plt.gcf()
# plt.show()


# Fine-Pruning defense
# Fine_Pruning_model = copy.deepcopy(model)
#
# pruning = Pruning(
#     train_dataset=test_dataset,
#     model=Fine_Pruning_model,
#     layer='layer3',
#     prune_rate=0.1,
# )
# pruning.repair(batch_size=64)
# Fine_Pruning_model = pruning.get_model()
# # print(Fine_Pruning_model)
# # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# acc, test_loss = test(Fine_Pruning_model, test_loader)
# # print("Fine-Pruning后，正常数据集下的准确率："+str(accuracy))
# print("Fine-Pruning  clean test-loss:" + str(test_loss))
# logger.info("Fine-Pruning  clean test-loss:" + str(test_loss))
# print("Fine-Pruning  ACC:" + str(acc))
# logger.info('Fine-Pruning  ACC: %s' % str(acc))
# trigger_acc = test_trigger_accuracy(test_loader, Fine_Pruning_model, 0, mask, trigger)
# # print("Fine-Pruning后，投毒数据集下的准确率：" + str(accuracy))
# print("Fine-Pruning Student poisoned test-loss:" + str(trigger_acc))
# # logger.info("Fine-Pruning Student poisoned test-loss:" + str(trigger_acc))
# # print("Fine-Pruning Student ASR:" + str(accuracy))
# # logger.info('Fine-Pruning Student ASR: %s' % str(accuracy))
#
# s_model=resnet18(num_classes=10).cuda()
# criterion = KLDiv(T=7)
# optimizer = torch.optim.SGD(s_model.parameters(), lr=0.001,
#                             momentum=0.9)
#
# c_acc=[]
# b_acc=[]
# for epoch in tqdm(range(100)):
#
#     kd_train(train_loader, [s_model, Fine_Pruning_model], criterion, optimizer)
# acc, test_loss = test(s_model, test_loader)
# trigger_acc = test_trigger_accuracy(test_loader, s_model, 0, mask, trigger)
# c_acc.append(acc)
# b_acc.append(trigger_acc)
#
# logger.info("c_acc"+str(c_acc))
# logger.info("p_acc"+str(b_acc))
# logger.info("#" * 20)

##NAD
# NAD defense
NAD_model = copy.deepcopy(model)
defense = NAD(
    model=NAD_model,
    loss=nn.CrossEntropyLoss(),
    power=2.0,
    beta=[500, 500, 500],
    target_layers=['layer1', 'layer2', 'layer3'],
)
schedule = {
    'batch_size': 64,
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'tune_lr': 0.01,
    'tune_epochs': 10,
    'epochs': 20,
    'schedule': [2, 4, 6, 8],
}
defense.repair(dataset=train_dataset, portion=1, schedule=schedule)
NAD_model = defense.get_model()

accuracy,test_loss = test(NAD_model, test_loader)
# print("NAD后，正常数据集下的准确率："+str(accuracy))
print("NAD T clean test-loss:" + str(test_loss))
logger.info("NAD T clean test-loss:" + str(test_loss))
print("NAD T ACC:" + str(accuracy))
logger.info('NAD T ACC: %s' % str(accuracy))
trigger_acc = test_trigger_accuracy(test_loader, NAD_model, 0, mask, trigger)
# print("NAD后，投毒数据集下的准确率：" + str(accuracy))
print("NAD T ASR:" + str(trigger_acc))
logger.info('NAD T ASR: %s' % str(trigger_acc))
logger.info("#" * 20)
s_model=resnet18(num_classes=10).cuda()
criterion = KLDiv(T=7)
optimizer = torch.optim.SGD(s_model.parameters(), lr=0.001,
                            momentum=0.9)

c_acc=[]
b_acc=[]
for epoch in tqdm(range(100)):

    kd_train(train_loader, [s_model, NAD_model], criterion, optimizer)
acc, test_loss = test(s_model, test_loader)
trigger_acc = test_trigger_accuracy(test_loader, s_model, 0, mask, trigger)
c_acc.append(acc)
b_acc.append(trigger_acc)

logger.info("c_acc"+str(c_acc))
logger.info("p_acc"+str(b_acc))
logger.info("#" * 20)

