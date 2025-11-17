import argparse
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
    parser.add_argument('--epochs', type=int, default=20, help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--p', type=float, default=2.0, help='power for AT')
    parser.add_argument('--qk_dim', type=float, default=128, help='power for AT')
    parser.add_argument('--w_argd_vert', type=float, default=1, help='weight of Node')
    parser.add_argument('--w_argd_edge', type=float, default=1, help='weight of Edge')
    parser.add_argument('--w_argd_tran', type=float, default=1, help='weight of Graph')
    return parser


opt = get_arguments().parse_args()


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


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
        norm = torch.norm(am, dim=(2,3), keepdim=True)
        am = torch.div(am, norm+eps)
        return am


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_step(opt, train_loader, nets, optimizer, criterions, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = nets['snet']
    tnet = nets['tnet']

    ###criterion XX for distillation loss function
    criterionCls = criterions['criterionCls']
    criterionAT = criterions['criterionAT']
    criterionARGD = criterions['criterionARGD']
    feat_s_last = None
    snet.train()

    for idx, (img, target) in enumerate(train_loader, start=1):

        img = img.to(device)
        target = target.to(device)


        activation1_s, activation2_s, activation3_s, feat_s, output_s = snet(img)
        with torch.no_grad():
            activation1_t, activation2_t, activation3_t, feat_t, output_t = tnet(img)
        if feat_s_last is None or (feat_s_last.size(0) != (feat_s.size(0))):
            feat_s_last = feat_s
        else:
            pass
        ####################################################
        cls_loss = criterionCls(output_s, target)
        ####################################################
        '''
        NAD loss function
        '''
        # at3_loss = criterionAT(activation3_s, activation3_t.detach()) * 5000
        #
        # at2_loss = criterionAT(activation2_s, activation2_t.detach()) * 5000
        # at1_loss = criterionAT(activation1_s, activation1_t.detach()) * 5000
        # at_loss = at1_loss + at2_loss + at3_loss + cls_loss
        ####################################################

        ###############################################
        '''
        ARGD loss function
        '''
        ARG_loss = criterionARGD([activation1_s, activation2_s, activation3_s, feat_s, output_s],
                                [activation1_t.detach(), activation2_t.detach(),
                                 activation3_t.detach(),
                                 feat_t.detach(),
                                 output_t.detach()])
        ARG_loss = cls_loss + ARG_loss

        loss_sum = ARG_loss

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        losses.update(loss_sum.item(), img.size(0))

        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()
        feat_s_last = feat_s.detach()
        if idx % opt.print_freq == 0:
            print('Epoch[{0}]:[{1:03}/{2:03}] '
                  'AT_loss:{losses.val:.4f}({losses.avg:.4f})  '
                  'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                  'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses,
                                                                 top1=top1, top5=top5))

def adjust_learning_rate(optimizer, epoch, lr):
    if epoch < 3:
        lr = lr
    elif epoch < 6:
        lr = 0.01
    elif epoch < 11:
        lr = 0.001
    else:
        lr = 0.01
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Sample(nn.Module):
    # using the adaptive average pool to get the feature
    def __init__(self, t_shape):
        super(Sample, self).__init__()
        t_C, t_H, t_W = t_shape
        self.sample = nn.AdaptiveAvgPool2d((t_H, t_W))
    def forward(self, g_s, bs):
        g_s = torch.stack([self.sample(f_s.pow(2).mean(1, keepdim=True)).view(bs, -1) for f_s in g_s], dim=1)  #

        return g_s


class nn_bn_relu(nn.Module):
    def __init__(self, nin, nout):
        super(nn_bn_relu, self).__init__()
        self.linear = nn.Linear(nin, nout).cuda()
        self.bn = nn.BatchNorm1d(nout).cuda()
        self.relu = nn.ReLU(False).cuda()

    def forward(self, x, relu=True):
        if relu:
            return self.relu(self.bn(self.linear(x))).cuda()
        return self.bn(self.linear(x)).cuda()


class LinearTransformStudent(nn.Module):
    def __init__(self, args):
        super(LinearTransformStudent, self).__init__()
        self.t = len(args.t_shapes)
        self.s = len(args.s_shapes)
        self.qk_dim = args.qk_dim
        self.relu = nn.ReLU(inplace=False)
        self.samplers = nn.ModuleList([Sample(t_shape) for t_shape in args.unique_t_shapes])  # channel_key+sample_key

        self.key_layer = nn.ModuleList([nn_bn_relu(s_shape[1], self.qk_dim) for s_shape in args.s_shapes]).cuda()
        self.bilinear = nn_bn_relu(args.qk_dim, args.qk_dim * len(args.t_shapes)) #bilinear get the vector embedding

    def forward(self, g_s):
        bs = g_s[0].size(0)
        channel_mean = [f_s.mean(3).mean(2).cuda() for f_s in g_s]  # calc_the channle.mean
        spatial_mean = [sampler(g_s, bs) for sampler in self.samplers]

        key = torch.stack([key_layer(f_s.cuda()) for key_layer, f_s in zip(self.key_layer, channel_mean)],
                          dim=1).view(bs * self.s, -1).cuda()  # Bs x h
        bilinear_key = self.bilinear(key.cuda(), relu=False).view(bs, self.s, self.t, -1)
        value = [F.normalize(s_m, dim=2) for s_m in spatial_mean]
        return bilinear_key, value


class LinearTransformTeacher(nn.Module):
    def __init__(self, args):
        super(LinearTransformTeacher, self).__init__()
        self.query_layer = nn.ModuleList([nn_bn_relu(t_shape[0], args.qk_dim) for t_shape in args.t_shapes])

    def forward(self, g_t):
        bs = g_t[0].size(0)
        channel_mean = [f_t.mean(3).mean(2) for f_t in g_t]
        spatial_mean = [f_t.pow(2).mean(1).view(bs, -1) for f_t in g_t]
        query = torch.stack([query_layer(f_t, relu=False) for f_t, query_layer in zip(channel_mean, self.query_layer)],
                            dim=1)
        value = [F.normalize(f_s, dim=1) for f_s in spatial_mean]
        return query, value


class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.qk_dim = args.qk_dim
        self.n_t = args.n_t
        self.linear_trans_s = LinearTransformStudent(args)
        self.linear_trans_t = LinearTransformTeacher(args)
        self.p_t = nn.Parameter(torch.Tensor(len(args.t_shapes), args.qk_dim)).to('cuda')
        self.p_s = nn.Parameter(torch.Tensor(len(args.s_shapes), args.qk_dim)).to('cuda')
        torch.nn.init.xavier_normal_(self.p_t)
        torch.nn.init.xavier_normal_(self.p_s)

    def forward(self, g_s, g_t):
        bilinear_key, h_hat_s_all = self.linear_trans_s(g_s)
        query, h_t_all = self.linear_trans_t(g_t)

        p_logit = torch.matmul(self.p_t, self.p_s.t())

        logit = torch.add(torch.einsum('bstq,btq->bts', bilinear_key, query), p_logit) / np.sqrt(self.qk_dim)
        atts = F.softmax(logit, dim=2)
        loss = []
        for i, (n, h_t) in enumerate(zip(self.n_t, h_t_all)):
            h_hat_s = h_hat_s_all[n]
            diff = self.cal_diff(h_hat_s, h_t, atts[:, i])
            loss.append(diff)
        return loss

    def cal_diff(self, v_s, v_t, att):
        diff = (v_s - v_t.unsqueeze(1)).pow(2).mean(2)
        diff = torch.mul(diff, att).sum(1).mean()
        return diff


def test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch):
    test_process = []
    top1 = AverageMeter()
    top5 = AverageMeter()
    snet = nets['snet']
    tnet = nets['tnet']

    criterionCls = criterions['criterionCls']
    criterionAT = criterions['criterionAT']

    snet.eval()
    tnet.eval()
    for idx, (img, target) in enumerate(test_clean_loader, start=1):
        img = img.cuda()
        target = target.cuda()

        with torch.no_grad():
            _, _, _, _, output_s = snet(img)
            _, _, _, _, output_t = tnet(img)

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_clean = [top1.avg, top5.avg]

    cls_losses = AverageMeter()
    at_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, all in enumerate(test_bad_loader, start=1):
        if len(all) == 2:
            img, target = all
        elif len(all) == 3:
            img, target, triggerlabel = all
        img = img.cuda()
        target = target.cuda()

        with torch.no_grad():
            activation1_s, activation2_s, activation3_s, feat_s, output_s = snet(img)
            activation1_t, activation2_t, activation3_t, feat_t, output_t = tnet(img)

            at3_loss = criterionAT(activation3_s, activation3_t).detach()
            at2_loss = criterionAT(activation2_s, activation2_t).detach()
            at1_loss = criterionAT(activation1_s, activation1_t).detach()
            at_loss = at3_loss + at2_loss + at1_loss
            cls_loss = criterionCls(output_s, target).detach()
        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        cls_losses.update(cls_loss.item(), img.size(0))
        at_losses.update(at_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_bd = [top1.avg, top5.avg, cls_losses.avg, at_losses.avg]

    print('[clean]Prec@1: {:.2f}'.format(acc_clean[0]))
    print('[bad]Prec@1: {:.2f}'.format(acc_bd[0]))


    return acc_clean, acc_bd


class ARGD(nn.Module):
    def __init__(self, opt):
        super(ARGD, self).__init__()

        self.w_argd_vert = opt.w_argd_vert
        self.w_argd_edge = opt.w_argd_edge
        self.w_argd_tran = opt.w_argd_tran
        self.margin = 1
        self.alpha = 0.1
        self.attention = Attention(opt)

    def forward(self, irg_s, irg_t):
        fm_s0, fm_s1, fm_s2, feat_s, out_s = irg_s
        fm_t0, fm_t1, fm_t2, feat_t, out_t = irg_t

        criterionAT = AT(2.0)
        loss_argd_vert = criterionAT(fm_s0, fm_t0) * 2000 + criterionAT(fm_s1, fm_t1) * 5000 + criterionAT(fm_s2,
                                                                                                    fm_t2) * 2000  # Node dis
        fm_s0_attention = criterionAT.attention_map(fm_s0)
        fm_t0_attention = criterionAT.attention_map(fm_t0)

        fm_s1_attention = criterionAT.attention_map(fm_s1)
        fm_t1_attention = criterionAT.attention_map(fm_t1)

        fm_s2_attention = criterionAT.attention_map(fm_s2)
        fm_t2_attention = criterionAT.attention_map(fm_t2)

        # the newly ways to calculate the global attention vector
        AT_s = [fm_s0_attention, fm_s1_attention, fm_s2_attention]
        AT_t = [fm_t0_attention, fm_t1_attention, fm_t2_attention]
        loss_global_list = self.attention(AT_s, AT_t)
        loss_global = 0
        for i in range(len(AT_s)):
            loss_global += loss_global_list[i]#embedding dis
        edg_tran_s2 = self.euclidean_dist_fms(fm_s0_attention, fm_s2_attention, squared=True)

        edg_tran_s1 = self.euclidean_dist_fms(fm_s1_attention, fm_s2_attention, squared=True)

        edg_tran_s0 = self.euclidean_dist_fms(fm_s0_attention, fm_s1_attention, squared=True)

        edg_tran_t2 = self.euclidean_dist_fms(fm_t0_attention, fm_t2_attention, squared=True)
        edg_tran_t1 = self.euclidean_dist_fms(fm_t1_attention, fm_t2_attention, squared=True)
        edg_tran_t0 = self.euclidean_dist_fms(fm_t0_attention, fm_t1_attention, squared=True)

        loss_argd_edge = (F.mse_loss(edg_tran_s0, edg_tran_t0) + F.mse_loss(edg_tran_s1, edg_tran_t1) + F.mse_loss(
            edg_tran_s2, edg_tran_t2)) / 3 #edge dis

        loss = (
                self.w_argd_vert * loss_argd_vert
                + self.w_argd_edge * loss_argd_edge
                + loss_global
        )

        return loss

    def euclidean_dist_fms(self, fm1, fm2, squared=False, eps=1e-12):
        if fm1.size(2) > fm2.size(2):
            fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))
        if fm1.size(1) < fm2.size(1):
            fm2 = (fm2[:, 0::2, :, :] + fm2[:, 1::2, :, :]) / 2.0

        fm1 = fm1.view(fm1.size(0), -1)
        fm2 = fm2.view(fm2.size(0), -1)
        fms_dist = torch.sum(torch.pow(fm1 - fm2, 2), dim=-1).clamp(min=eps)

        if not squared:
            fms_dist = fms_dist.sqrt()

        fms_dist = fms_dist / fms_dist.max()

        return fms_dist

    def euclidean_dist_fm(self, fm, squared=False, eps=1e-12):
        fm = fm.view(fm.size(0), -1)
        fm_square = fm.pow(2).sum(dim=1)
        fm_prod = torch.mm(fm, fm.t())
        fm_dist = (fm_square.unsqueeze(0) + fm_square.unsqueeze(1) - 2 * fm_prod).clamp(min=eps)

        if not squared:
            fm_dist = fm_dist.sqrt()

        fm_dist = fm_dist.clone()
        fm_dist[range(len(fm)), range(len(fm))] = 0
        fm_dist = fm_dist / fm_dist.max()

        return fm_dist

    def euclidean_dist_feat(self, feat, squared=False, eps=1e-12):

        feat_square = feat.pow(2).sum(dim=1)
        feat_prod = torch.mm(feat, feat.t())
        feat_dist = (feat_square.unsqueeze(0) + feat_square.unsqueeze(1) - 2 * feat_prod).clamp(min=eps)

        if not squared:
            feat_dist = feat_dist.sqrt()

        feat_dist = feat_dist.clone()
        feat_dist[range(len(feat)), range(len(feat))] = 0
        feat_dist = feat_dist / feat_dist.max()

        return feat_dist

    def resize(self, fm1, fm2):
        if fm1.size(2) > fm2.size(2):
            fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))
        if fm1.size(1) < fm2.size(1):
            fm2 = (fm2[:, 0::2, :, :] + fm2[:, 1::2, :, :]) / 2.0

            return fm1, fm2
    def viz(self, input):
        x = input[0][0].cpu()
        # min_num = np.minimum(4, x.size()[0])
        plt.imshow(x, interpolation='bicubic')
        plt.show()


def unique_shape(s_shapes):
    n_s = []
    unique_shapes = []
    n = -1
    for s_shape in s_shapes:
        if s_shape not in unique_shapes:
            unique_shapes.append(s_shape)
            n += 1
        n_s.append(n)
    return n_s, unique_shapes


def train_ARGD(teacher,student,opt,train_loader,test_clean_loader, test_bad_loader):

    print('----------- Network Initialization --------------')

    print('finished student model init...')
    criterionAT = AT(2.0)
    teacher.eval()
    image_size = 32
    # get the dim in feat for teacher and student model
    data = torch.randn(1, 3, image_size, image_size).to(device)
    teacher.eval()
    student.eval()
    with torch.no_grad():
        activation1_t, activation2_t, activation3_t, feat_t, _ = teacher(data)
        activation1_s, activation2_s, activation3_s, feat_s, _ = student(data)
    AT_s = [criterionAT.attention_map(activation1_s), criterionAT.attention_map(activation2_s), criterionAT.attention_map(activation3_s)]
    AT_t = [criterionAT.attention_map(activation1_t), criterionAT.attention_map(activation2_t), criterionAT.attention_map(activation3_t)]
    opt.s_shapes = [AT_s[i].size() for i in range(3)]  # get the layer feat from the s feat shape
    opt.t_shapes = [AT_t[i][0].size() for i in range(3)]  # get the layer feat from the t
    opt.n_t, opt.unique_t_shapes = unique_shape(
        opt.t_shapes)  # n_t is the vector for s model, unique_t_shapes is the s_shapes

    nets = {'snet': student, 'tnet': teacher}

    for param in teacher.parameters():
        param.requires_grad = False

    optimizer = torch.optim.SGD(student.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                nesterov=True)

    # define loss functions
    criterionCls = nn.CrossEntropyLoss().cuda()
    criterionAT = AT(opt.p)
    criterionARGD = ARGD(opt)

    print('----------- Train Initialization --------------')
    print('testing the initialize models......')
    criterions = {'criterionCls': criterionCls, 'criterionAT': criterionAT, 'criterionARGD': criterionARGD}
    acc_clean, acc_bad = test(opt, test_clean_loader, test_bad_loader, nets, criterions, 0)
    print("clean acc:", acc_clean)
    print("backdoor acc:", acc_bad)
    for epoch in range(0, opt.epochs):

        adjust_learning_rate(optimizer, epoch, opt.lr)
        # train every epoch
        train_step(opt, train_loader, nets, optimizer, criterions, epoch + 1)
        # evaluate on testing set
        print('testing the models......')
        acc_clean, acc_bad = test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch + 1)


# class trojan_model(nn.Module):
#     """
#     For train trojan model
#     """
#
#     def __init__(self, n_classes=10):
#         super(trojan_model, self).__init__()
#
#         self.n_classes = n_classes
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
#         # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
#         self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
#         # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         # self.conv4_drop = nn.Dropout2d(0.5)
#         self.fc1 = nn.Linear(128 * 5 * 5, 256)
#         self.fc2 = nn.Linear(256, n_classes)
#
#     def forward(self, x):
#         out = F.relu((self.conv1(x)))
#         out = F.relu(F.max_pool2d(self.conv2(out), 2))
#         activatopion1 = out
#         out = F.relu((self.conv3(out)))
#         activation2 = out
#         out = F.relu(F.max_pool2d(self.conv4(out), 2))
#         activation3 = out
#         # print('out;', out.shape)
#         out = out.view(-1, 128 * 5 * 5)
#         feat = out
#         out = F.relu(self.fc1(out))
#         out = F.dropout(out, training=self.training)
#         out = self.fc2(out)
#         return activation1, activation2, activation3,feat, out
# model=trojan_model().to(device)
# train_dataset=CIFAR10(root='data',train=True,transform=transforms.Compose([transforms.ToTensor()]))
# train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
# train_ARGD(model,model,opt,train_loader,train_loader,train_loader)