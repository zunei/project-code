#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse
import copy
import os
import shutil
import sys
import warnings
from linecache import cache

import torchvision.models as models
import numpy as np
import math
import pdb
import torch
# import wandb
import torch.nn.functional as F
from tqdm import tqdm

from models.inceptionNet import InceptionNetSmall
from models.wrn import wrn_16_1, wrn_16_2
from helpers.datasets import partition_data
from helpers.utils import get_dataset, average_weights, DatasetSplit, KLDiv, setup_seed, test, kldiv, add_trigger, test_trigger_accuracy
from models.generator import Generator
from models.nets import CNNCifar, CNNMnist, CNNCifar100,LetNet
from models.resnet import resnet18, resnet34
from models.vit import deit_tiny_patch16_224
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import random
from torch.utils.data import Subset
from torch import nn



warnings.filterwarnings('ignore')
upsample = torch.nn.Upsample(mode='nearest', scale_factor=7)



class LocalUpdate_with_mask(object):
    def __init__(self, args, dataset):
        self.args = args
        #################测试用👇
#         # 随机抽取 1000 张图片
#         indices = random.sample(range(len(dataset)), 1000)
#         subset = Subset(dataset, indices)

#         # 创建数据加载器
#         subset_loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=True)
#         self.train_loader=subset_loader
# #         print(len(self.train_loader))
#         #################测试用👆
        self.train_loader = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True)
        
        
    def update_weights(self, model):
       # # cifar
        init_mask = np.zeros((1, 32, 32)).astype(np.float32)
        init_pattern = np.random.normal(0, 1, (1, 32, 32)).astype(np.float32)
       #mnist
        # init_mask = np.zeros((1, 28, 28)).astype(np.float32)
        # init_pattern = np.random.normal(0, 1, (1, 28, 28)).astype(np.float32)
        mask_nc = torch.from_numpy(init_mask).clamp_(0, 1) # nc means neural cleanse
        pattern_nc = torch.from_numpy(init_pattern).clamp_(0, 1)
#     cifar
        shadow_model = resnet18(num_classes=10).cuda()
#         shadow_model=CNNMnist().cuda()
#         shadow_model=LetNet().cuda()
#         shadow_model=CNNCifar().cuda()
        pattern = pattern_nc.cuda()
        mask = mask_nc.cuda()
        pattern.requires_grad_(True)
        mask.requires_grad_(True)

        target_label = self.args.target_label_backdoor

        optimizer_for_shadow = torch.optim.SGD(shadow_model.parameters(), lr=self.args.lr, momentum=0.9)
        optimizer_for_teacher = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.9)
        optimizer_for_trigger = torch.optim.SGD([pattern, mask], lr=self.args.lr, momentum=0.9)

        canshu = torch.load('saved/mytest_wudu' + str(args.dataset) + '_ttt.pkl')

        model.load_state_dict(canshu)

        local_acc_list = []
#         print("------------- LocalUpdate with mask -------------")
#         for iter in tqdm(range(self.args.local_ep)):
#             for batch_idx, (images, labels) in enumerate(self.train_loader):
#                 images, labels = images.cuda(), labels.cuda()
#
#                 images_clean = images.clone().detach()
#                 labels_clean = labels.clone().detach()
#
#                 images_poison = images.clone().detach()
#                 labels_poison = labels.clone().detach()
#                 mask_temp = mask.detach()
#                 pattern_temp = pattern.detach()
# #                 print(images_poison.shape)
#                 images_poison = add_trigger(images_poison, mask_temp, pattern_temp)
#                 labels_poison[:] = target_label
#
#                 # ---------------------------------------
#                 model.train()
#                 optimizer_for_teacher.zero_grad()
#
#                 output = model(images_clean)
#                 loss_raw_data = F.cross_entropy(output, labels_clean)
#                 print(images_poison.shape)
#                 output_with_mask = model(images_poison)
#                 loss_trigger = F.cross_entropy(output_with_mask, labels_poison)
#                 beta_backdoor = args.beta_backdoor
#
#                 loss_teacher = loss_raw_data + beta_backdoor * loss_trigger
#                 loss_teacher=loss_raw_data
#                 loss_teacher.backward()
#                 optimizer_for_teacher.step()

    #
    #             # ---------------------------------------
    #             shadow_model.train()
    #             optimizer_for_shadow.zero_grad()
    #
    #             output_shadow = shadow_model(images)
    #             loss_shadow_1 = F.cross_entropy(output_shadow, labels)
    #
    #             output_s = output_shadow.detach()
    #             output_t = output.detach()
    #             kl_divergence=kldiv(output_s,output_t,T=self.args.T)
    #             loss_shadow_0 = kl_divergence
    #
    #             # alpha用来平衡student和teacher的相似度&student的acc
    #             alpha = args.alpha_backdoor
    #             loss_shadow = alpha*loss_shadow_0 + (1-alpha)*loss_shadow_1
    #
    #             loss_shadow.backward()
    #             optimizer_for_shadow.step()
    #
    #             # ---------------------------------------
    #             model.eval()
    #             shadow_model.eval()
    #             optimizer_for_trigger.zero_grad()
    #
    #             images_temp = images.clone()
    #             images_temp.cuda()
    #
    #             images_masked = (1 - mask) * images_temp + mask * pattern
    #
    #             output_with_mask = model(images_masked)
    #             loss_optimize_trigger_0 = F.cross_entropy(output_with_mask/self.args.T, labels_poison)
    #             # loss_optimize_trigger_0 = F.cross_entropy(output_with_mask, labels_poison)
    #
    #             output_with_mask_shadow = shadow_model(images_masked)
    #             loss_optimize_trigger_1 = F.cross_entropy(output_with_mask_shadow, labels_poison)
    #
    #             loss_optimize_trigger = loss_optimize_trigger_0 + loss_optimize_trigger_1 + args.miu*torch.norm(mask, p=2)
    #             loss_optimize_trigger.backward()
    #             optimizer_for_trigger.step()
    #
    #
    #             # 使用torch.clamp将mask和pattern限制在[0,1]范围内
    #             with torch.no_grad():
    #                 pattern.clamp_(0, 1)
    #                 mask.clamp_(0, 1)
    #
    #         acc, test_loss = test(model, test_loader)
    #         local_acc_list.append(acc)
    #
    #     # ---------------------------------------
    #     pattern_nc = pattern.cpu()
    #     mask_nc = mask.cpu()
    #
    #
    #     torch.save(shadow_model.state_dict(), 'saved/shadow'+str(args.dataset)+'_model.pkl')
    #     torch.save(pattern_nc, 'saved/adba_client_model_weights/pattern_dataset{}_clients{}_poison{}_partition{}_betapartition{}_betabackdoor{}_alphabackdoor{}.pt'.format(args.dataset,args.num_users,args.num_poison_users,args.partition,args.beta_partition,args.beta_backdoor,args.alpha_backdoor))
    #     torch.save(mask_nc, 'saved/adba_client_model_weights/mask_dataset{}_clients{}_poison{}_partition{}_betapartition{}_betabackdoor{}_alphabackdoor{}.pt'.format(args.dataset,args.num_users,args.num_poison_users,args.partition,args.beta_partition,args.beta_backdoor,args.alpha_backdoor))
    #
    #
    #
    #     print("------------------ Extra epoch ------------------")
    #     # 用badnet的方法进行巩固
    #     ratio=0.3  # 毒化率
    #     trigger = pattern_nc.cuda().detach()
    #     mask = mask_nc.cuda().detach()
    #self.args.local_ep)

        for iter in tqdm(range(1)):
            model.train()
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.cuda(), labels.cuda()

                # index = math.ceil(images.shape[0] * ratio)
                # image_trigger = images[0:index, :, :, :]
                # image_trigger = add_trigger(image_trigger, mask, trigger)
                # images[0:index, :, :, :] = image_trigger
                optimizer_for_teacher.zero_grad()
                output = model(images)
                # labels[0:index] = target_label
                loss = F.cross_entropy(output, labels)

                loss.backward()
                optimizer_for_teacher.step()
            model.eval()
            acc, test_loss = test(model, test_loader)
    #     model.eval()
    #     acc, test_loss = test(model, test_loader)
    #
    #     asr=test_trigger_accuracy(test_loader=test_loader,model=model,target_label=target_label,mask=mask_nc,trigger=pattern_nc)
    #     print("[client] acc %.4f" % (acc))
    #     print("[client] asr %.4f" % (asr))
    #     if not(self.args.txtpath==""):
    #         with open(args.txtpath,"a") as f:
    #             f.write("[client] acc:{}\n [client] asr:{}\n".format(acc,asr))
    #     torch.save(model.state_dict(), 'saved/mytest_wudu'+str(args.dataset)+'_ttt.pkl')
    #
    #     # # # # 加载权重文件
    #     # # #
    #     # # #
    #     # # # # 加载权重到模型中
    #     canshu=torch.load('saved/mytest_wudu'+str(args.dataset)+'_ttt.pkl')
    #     s_canshu=torch.load('saved/shadow'+str(args.dataset)+'_model.pkl')
        mask = torch.load(
    './saved/adba_client_model_weights/mask_dataset{}_clients{}_poison{}_partition{}_betapartition{}_betabackdoor{}_alphabackdoor{}.pt'.format(
        args.dataset, args.num_users, args.num_poison_users, args.partition, args.beta_partition, args.beta_backdoor,
        args.alpha_backdoor))
        trigger = torch.load(
    './saved/adba_client_model_weights/pattern_dataset{}_clients{}_poison{}_partition{}_betapartition{}_betabackdoor{}_alphabackdoor{}.pt'.format(
        args.dataset, args.num_users, args.num_poison_users, args.partition, args.beta_partition, args.beta_backdoor,
        args.alpha_backdoor))
        # model.load_state_dict(canshu)
    #     shadow_model.load_state_dict(s_canshu)
    #     model.eval()
    #     shadow_model.eval()
        asr = test_trigger_accuracy(test_loader=test_loader,model=model,target_label=0,mask=mask,trigger=trigger)
    #
    #
    #     asr = test_trigger_accuracy(test_loader=test_loader, model=shadow_model, target_label=target_label, mask=mask, trigger=trigger)
    #
    #     print("ASR"+str(asr))
        acc , test_loss = test ( model , test_loader )
    #     ##自编
    #     print("model_clean")
    #     model.train()
    #     mymodel=model_clean(args=args,model=model)
    #     models= mymodel.updata(args=args,model=model,test_loader=test_loader,dataset=train_dataset,mask=mask,trigger=trigger,s_model=shadow_model)
    #     return models.state_dict () , np.array ( local_acc_list )

        return model.state_dict(), np.array(local_acc_list)
    
   #-------------自己编的
class myDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
class model_clean():
    def __init__(self,args,model ) :
        self.args = args
        self.model=model
#         self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=64,shuffle=False)
#         self.desired_class = args.target_label_backdoor  # 选择类别
#         self.mask=mask
#         self.trigger=trigger
    def updata(self,args,model,test_loader,dataset,mask,trigger,s_model):
        chaocan=1
        new_data = [ ]
        new_targets = [ ]
        old_model = copy.deepcopy(model)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=64,shuffle=False)
        # leng=0
        # for images , labels in dataset :
        #     if labels == args.target_label_backdoor :
        #
        #         new_data.append ( images )
        #         new_targets.append ( labels )
        #         leng+=1
        # clean_data = myDataset ( new_data , new_targets )
        # print(leng)
        from torch.utils.data import SubsetRandomSampler
        # 随机选择索引
        indices = torch.randperm(len(dataset), dtype=torch.long)[:len((dataset))]
        sampler = SubsetRandomSampler(indices)
        # shadow_model = CNNMnist().cuda()
        # shadow_model = LetNet().cuda()
        shadow_model = resnet18(num_classes=10).cuda()
        # shadow_model = CNNCifar().cuda()
        optimizer_s=torch.optim.SGD ( shadow_model.parameters () , lr=args.lr , momentum=0.9 )

        # 创建一个DataLoader，它只会加载这1000张图片  
        p_data = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=sampler)  
        data=torch.utils.data.DataLoader( dataset , batch_size=64 , shuffle=False )
        optimizer= torch.optim.SGD ( model.parameters () , lr=args.lr , momentum=0.9 )
        criterion = nn.CrossEntropyLoss ().cuda ()
        mse = nn.MSELoss().cuda()
        sml1=nn.SmoothL1Loss().cuda()
        nnl=torch.nn.NLLLoss()

        criterion_bug = KLDiv(T=args.T)

        # l2_criterion=nn.MSELoss().cuda()
        alpha=0.5
        temperature=self.args.T
        for iter in tqdm ( range ( 50 ) ) :

            for batch_idx ,( (images , labels),(x,y)) in enumerate (zip(data,p_data)) :
                image,label=images.cuda(),labels.cuda()
                x,y=x.cuda(),y.cuda()
#                 y=label
                # ------------------------------
                mask = mask.cuda()
                trigger = trigger.cuda()
                X_trigger = add_trigger(x, mask, trigger)



                # ------------------------------
                shadow_model.eval()
                model.train()
                images_poison = y.clone().detach()
                optimizer.zero_grad()
                images_poison[ : ] = args.target_label_backdoor
                outputs=model(images.cuda())
                loss_c=criterion(outputs,label)
                outputs_t=model(X_trigger)
                loss_2=criterion(outputs_t,y)
                os_out = s_model(X_trigger)
                # ns_out = shadow_model(X_trigger)
                # loss_s1 = criterion(os_out, ns_out/temperature)
                new =old_model(X_trigger)
                loss_new=sml1(outputs_t/temperature,new/temperature)
                loss_kd=kldiv(outputs_t/temperature, os_out, T=self.args.T)
                # loss_s2=criterion(os_out,ns_out)
                # loss_s=kldiv(ns_out, os_out, T=self.args.T)
#                 soft_targets = F.softmax ( outputs / temperature , dim=1 )
#                 outputs_for_distillation = model ( image ) / temperature
#                 distillation_loss = F.kl_div ( F.log_softmax ( outputs_for_distillation , dim=1 ) ,
#                                                soft_targets , reduction='batchmean' ) * (temperature ** 2)
                # 总损失
                # loss =  loss_2+loss_s1
                loss = alpha * loss_c + (1 - alpha) * loss_2+loss_new+chaocan*loss_kd
                # loss = alpha * loss_c + (1 - alpha) * loss_2

                # loss =loss_c + loss_2 + 0.5*loss_kd+0.5*loss_new
                loss.backward ()
                optimizer.step ()

                #----------------------------------------------------

            model.eval()
            acc , test_loss = test ( model , test_loader )
            shadow_model.eval()
            # asr_s = test_trigger_accuracy(test_loader=test_loader, model=shadow_model,
            #                               target_label=args.target_label_backdoor,
            #                               mask=mask, trigger=trigger)
        model.eval()
        # shadow_model.eval()
        # asr_s = test_trigger_accuracy(test_loader=test_loader, model=shadow_model,
        #                               target_label=args.target_label_backdoor,
        #                               mask=mask, trigger=trigger)
        # print(asr_s)
        asr = test_trigger_accuracy ( test_loader=test_loader , model=model , target_label=args.target_label_backdoor ,
                                              mask=mask , trigger=trigger )
        acc, test_loss = test(model, test_loader)
        if not (args.txtpath == ""):
            with open(args.txtpath, "a") as f:
                f.write("[t] acc:{}\n".format(acc))
                f.write("[t] asr:{}\n".format(asr))

        print ( "[T] acc %.4f" % (acc) )
        print ( "[T] asr %.4f" % (asr) )
        torch.save(model.state_dict(), 'saved/clean/mytest_'+str(chaocan)+'_' + str(args.dataset) + 'tttt_clean.pkl')
        criterion_k = KLDiv(T=7)
        for epoch in tqdm(range(100)):
            shadow_model.train()

            kd_train(p_data, [shadow_model, model], criterion_k, optimizer_s)
            shadow_model.eval()
            acc, test_loss = test(shadow_model, test_loader)
            # print("<acc:>"+str(acc)+"<loss>"+str(test_loss))
            # asr_s = test_trigger_accuracy(test_loader=test_loader, model=shadow_model,
            #                               target_label=args.target_label_backdoor,
            #                               mask=mask, trigger=trigger)
        asr_s = test_trigger_accuracy(test_loader=test_loader, model=shadow_model,
                                      target_label=args.target_label_backdoor,
                                      mask=mask, trigger=trigger)
        acc_s, test_loss_s = test(shadow_model, test_loader)
        print("<s asr>"+str(asr_s))
        if not (args.txtpath == ""):
            with open(args.txtpath, "a") as f:
                f.write("[student] acc:{}\n".format(acc_s))
                f.write("[student] asr:{}\n".format(asr_s))
        torch.save(shadow_model.state_dict(),'saved/student/mytest_' + str(chaocan) + '_' + str(args.dataset) + 'ssss_clean.pkl')
        return model



def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=1,
                        help="number of users: K")
    parser.add_argument('--num_poison_users', type=int, default=1,
                        help="number of poison users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=20,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)')
    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name \
                        of dataset")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')


    # Data Free
    parser.add_argument('--adv', default=0, type=float, help='scaling factor for adv loss')

    parser.add_argument('--bn', default=0, type=float, help='scaling factor for BN regularization')
    parser.add_argument('--oh', default=0, type=float, help='scaling factor for one hot loss (cross entropy)')
    parser.add_argument('--act', default=0, type=float, help='scaling factor for activation loss used in DAFL')
    parser.add_argument('--save_dir', default='run/synthesis', type=str)
    parser.add_argument('--partition', default='dirichlet', type=str)
    parser.add_argument('--beta_partition', default=0.5, type=float,
                        help=' If beta is set to a smaller value, '
                             'then the partition is more unbalanced')
    
    
    # BackDoor
    parser.add_argument('--beta_backdoor', default=0.3, type=float,
                        help=' If beta is set to a smaller value, '
                             'then the partition is more unbalanced')
    parser.add_argument('--alpha_backdoor', default=1, type=float,
                        help=' If beta is set to a smaller value, '
                             'then the partition is more unbalanced')
    parser.add_argument('--target_label_backdoor',default=0,type=int,help='target label for poison ')

    # Basic
    parser.add_argument('--lr_g', default=1e-3, type=float,
                        help='initial learning rate for generation')
    parser.add_argument('--T', default=7, type=float)
    parser.add_argument('--g_steps', default=20, type=int, metavar='N',
                        help='number of iterations for generation')
    parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--nz', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--synthesis_batch_size', default=256, type=int)
    # Misc
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--type', default="pretrain", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--model', default="", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--other', default="", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--txtpath', default="", type=str,
                    help='txt for some ducument')
    parser.add_argument('--miu', default=0.1, type=float, help='scaling factor for normalization')

    args = parser.parse_args()
    return args


class Ensemble(torch.nn.Module):
    def __init__(self, model_list):
        super(Ensemble, self).__init__()
        self.models = model_list

    def forward(self, x):
        logits_total = 0
        for i in range(len(self.models)):
            logits = self.models[i](x)
            logits_total += logits
        logits_e = logits_total / len(self.models)

        return logits_e


def kd_train(dataloader, model, criterion, optimizer):
    student, teacher = model
    student.train()
    teacher.eval()

    for batch_idx, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        images = images.cuda()
        with torch.no_grad():
            t_out = teacher(images)
        s_out = student(images.detach())
        loss_s = criterion(s_out, t_out.detach())

        loss_s.backward()
        optimizer.step()


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)


def get_model(args):
    if args.model == "mnist_cnn":
        global_model = CNNMnist().cuda()
    elif args.model == "fmnist_cnn":
        # global_model = CNNMnist().cuda()
        global_model = LetNet().cuda()
    elif args.model == "cnn":
        global_model = CNNCifar().cuda()
    elif args.model == "svhn_cnn":
        global_model = CNNCifar().cuda()
    elif args.model == "cifar100_cnn":
        global_model = CNNCifar100().cuda()
    elif args.model == "res":
        # global_model = resnet18()
        global_model = resnet18(num_classes=10).cuda()

    elif args.model == "wrn":
        global_model = wrn_16_1(10).cuda()
    elif args.model == "inp":
        global_model=InceptionNetSmall().cuda()
    elif args.model == "vit":
        global_model = deit_tiny_patch16_224(num_classes=1000,
                                             drop_rate=0.,
                                             drop_path_rate=0.1)
        global_model.head = torch.nn.Linear(global_model.head.in_features, 10)
        global_model = global_model.cuda()
        global_model = torch.nn.DataParallel(global_model)
    return global_model


if __name__ == '__main__':

    args = args_parser()
    if not(args.txtpath==""):
        with open(args.txtpath,"a") as f:
            f.write('----------dataset{}_clients{}_poison{}_partition{}_betapartition{}_betabackdoor{}_alphabackdoor{}----------\n'.format(args.dataset,args.num_users,args.num_poison_users,args.partition,args.beta_partition,args.beta_backdoor,args.alpha_backdoor))  
    
#     wandb.init(project="ADBA", mode="offline")

    setup_seed(args.seed)
    train_dataset, test_dataset, user_groups, traindata_cls_counts = partition_data(
        args.dataset, args.partition, beta=args.beta_partition, num_users=args.num_users)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64,
                                              shuffle=False)

    # BUILD MODEL
    global_model = get_model(args)
    bst_acc = -1
    description = "inference acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"
    local_weights = []
    global_model.train()
    acc_list = []
    users = []
    if args.type == "pretrain":
        # ===============================================
        local_model = LocalUpdate_with_mask(args=args, dataset=train_dataset)
        w_poison, local_acc_poison = local_model.update_weights(copy.deepcopy(global_model))

        acc_list.append(local_acc_poison)
        local_weights.append(copy.deepcopy(w_poison))

        if not(args.txtpath==""):
            with open(args.txtpath,"a") as f:
                f.write("[client] acc:{}\n".format(local_acc_poison))

        torch.save(local_weights, 'saved/adba_client_model_weights/dataset{}_clients{}_poison{}_partition{}_betapartition{}_betabackdoor{}_alphabackdoor{}.pkl'.format(args.dataset,args.num_users,args.num_poison_users,args.partition,args.beta_partition,args.beta_backdoor,args.alpha_backdoor))
        # ===============================================
        
    else:
        # ===============================================

        local_weights = torch.load(
           'saved/adba_client_model_weights/dataset{}_clients{}_poison{}_partition{}_betapartition{}_betabackdoor{}_alphabackdoor{}.pkl'.format(args.dataset,args.num_users,args.num_poison_users,args.partition,args.beta_partition,args.beta_backdoor,args.alpha_backdoor))

        print('---------', '[server] dataset{}_clients{}_poison{}_partition{}_betapartition{}_betabackdoor{}_alphabackdoor{}'.format(args.dataset,args.num_users,args.num_poison_users,args.partition,args.beta_partition,args.beta_backdoor,args.alpha_backdoor), '---------')

        # model_list = []
        # for i in range(len(local_weights)):
        #     net = copy.deepcopy(global_model)
        #     net.load_state_dict(local_weights[i])
        #     model_list.append(net)
        #
        # ensemble_model = Ensemble(model_list)
        canshu=torch.load('saved/clean/mytest'+ str(args.dataset) + '_clean.pkl')
        # ensemble_model=resnet18(num_classes=10).cuda()
        ensemble_model=CNNMnist().cuda()
        ensemble_model.load_state_dict(canshu)
        print("ensemble acc:")
        test(ensemble_model, test_loader)
        mask = torch.load(
            './saved/adba_client_model_weights/mask_dataset{}_clients{}_poison{}_partition{}_betapartition{}_betabackdoor{}_alphabackdoor{}.pt'.format(
                args.dataset, args.num_users, args.num_poison_users, args.partition, args.beta_partition,
                args.beta_backdoor, args.alpha_backdoor))
        trigger = torch.load(
            './saved/adba_client_model_weights/pattern_dataset{}_clients{}_poison{}_partition{}_betapartition{}_betabackdoor{}_alphabackdoor{}.pt'.format(
                args.dataset, args.num_users, args.num_poison_users, args.partition, args.beta_partition,
                args.beta_backdoor, args.alpha_backdoor))
        target_label = 0
        # for i in range(len(local_weights)):
        #     client_model = model_list[i]
        #     trigger_acc = test_trigger_accuracy(test_loader, client_model, target_label, mask, trigger)
        #     print('[teacher]', i, " ", 'asr', trigger_acc)
        trigger_acc = test_trigger_accuracy(test_loader, ensemble_model, target_label, mask, trigger)
        # ===============================================
        global_model = get_model(args)
        # ===============================================

        criterion = KLDiv(T=3)
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.9)
        global_model.train()
        distill_acc = []

        args.cur_ep = 0
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                                   shuffle=False)
        
#                 #################测试用👇
#         # 随机抽取 1000 张图片
#         indices = random.sample(range(len(train_dataset)), 1000)
#         subset = Subset(train_dataset, indices)

#         # 创建数据加载器
#         subset_loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=True)
#         train_loader=subset_loader
# #         print(len(self.train_loader))
#         #################测试用👆
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                                   shuffle=False)


        for epoch in tqdm(range(args.epochs)):
            args.cur_ep += 1
            kd_train(train_loader, [global_model, ensemble_model], criterion, optimizer)
            acc, test_loss = test(global_model, test_loader)
            distill_acc.append(acc)
            is_best = acc > bst_acc
            bst_acc = max(acc, bst_acc)

            _best_ckpt = './saved/adba_server_model_weights/server_dataset{}_clients{}_poison{}_partition{}_betapartition{}_betabackdoor{}_alphabackdoor{}.pth'.format(args.dataset,args.num_users,args.num_poison_users,args.partition,args.beta_partition,args.beta_backdoor,args.alpha_backdoor)
            print("best acc:{}".format(bst_acc))
            save_checkpoint({
                'state_dict': global_model.state_dict(),
                'best_acc': float(bst_acc),
            }, is_best, _best_ckpt)
#             wandb.log({'[server] acc': acc})


        trigger_acc = test_trigger_accuracy(test_loader, global_model, target_label, mask, trigger)
        print('[student] asr', trigger_acc)
#         wandb.log({'[server] asr': trigger_acc})
        
        if not(args.txtpath==""):
            with open(args.txtpath,"a") as f:
                f.write("[student] acc:{}\n".format(bst_acc))  
                f.write("[student] asr:{}\n".format(trigger_acc))

        # for i in range(len(local_weights)):
        #     client_model = model_list[i]
        #     trigger_acc = test_trigger_accuracy(test_loader, client_model, target_label, mask, trigger)
        #     print('[teacher]', i, " ", 'asr', trigger_acc)