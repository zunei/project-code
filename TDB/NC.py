import torch
import time
import torch.optim as optim
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from torch import Tensor, nn
from dataclasses import dataclass
from logging import Logger
from typing import Tuple, Callable
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F

import logging
logger = logging.getLogger()

def tanh_func(x: Tensor) -> Tensor:
    return x.tanh().add(1).mul(0.5)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ':f'):
        self.name: str = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def normalize_mad(values: torch.Tensor, side: str = None) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values, dtype=torch.float)
    median = values.median()
    abs_dev = (values - median).abs()
    mad = abs_dev.median()
    measures = abs_dev / mad / 1.4826
    if side == 'double':    # TODO: use a loop to optimie code
        dev_list = []
        for i in range(len(values)):
            if values[i] <= median:
                dev_list.append(float(median - values[i]))
        mad = torch.tensor(dev_list).median()
        for i in range(len(values)):
            if values[i] <= median:
                measures[i] = abs_dev[i] / mad / 1.4826

        dev_list = []
        for i in range(len(values)):
            if values[i] >= median:
                dev_list.append(float(values[i] - median))
        mad = torch.tensor(dev_list).median()
        for i in range(len(values)):
            if values[i] >= median:
                measures[i] = abs_dev[i] / mad / 1.4826
    return measures


def to_numpy(x, **kwargs) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.array(x, **kwargs)





@dataclass
class NeuralCleanseConfig():
    epoch: int = 2
    lr: float = 0.002
    betas: Tuple[float] = (0.5, 0.9)
    weight_decay: float = 0.0005
    lr_decay_ratio: float = 0.2
    batch_size: int = 64

    init_cost: float = 1e-3
    cost_multiplier: float = 1.5
    cost_multiplier_up: float = 1.5
    cost_multiplier_down: float = 1.5 ** 1.5
    patience: float = 10
    attack_succ_threshold: float = 0.99
    early_stop: bool = True
    early_stop_threshold: float = 0.99
    early_stop_patience: float = 10 * 2


class NeuralCleanse():
    name: str = 'neural_cleanse'

    def __init__(
            self,
            model: nn.Module,
            transform: Callable[[Tensor], Tensor],
            input_shape: Tuple[int, int, int],
            dataset: Dataset,
            cfg: NeuralCleanseConfig = None,
            device: str = 'cuda') -> None:
        """The backdoor defense method

        Args:
            model (nn.Module): The suspicious PyTorch model.
            transform (Callable[[Tensor], Tensor]): The callable transform (including normalization).
            input_shape (Tuple[int, int, int]): Then model input shape [in_channel, H, W].
            dataset (Dataset): The unnormalized dataset.
            log_path (str): The folder used to store the log files.
            logger (Logger): The logger.
            device (str, optional): CPU/GPU. Defaults to 'cuda'.
        """
        self.model = model
        self.transform = transform
        self.input_shape = input_shape
        self.dataset = dataset
        self.cfg = cfg if cfg else NeuralCleanseConfig()
        self.device = device

    def detect(self):

        mark_list, mask_list, loss_list = self.get_potential_triggers()
        mask_norms = mask_list.flatten(start_dim=1).norm(p=1, dim=1).tolist()
        mad = normalize_mad(mask_norms).tolist()
        loss_mad = normalize_mad(loss_list).tolist()
        loss_list = loss_list.tolist()

        print(mad)
        logger.info("NC outlier detection:")
        logger.info(mad)
        mark_list = [to_numpy(i) for i in mark_list]
        mask_list = [to_numpy(i) for i in mask_list]
        loss_list = [to_numpy(i) for i in loss_list]

        suspicious_labels = []
        for i in range(len(mad)):
            if mad[i] > 2:
                suspicious_labels.append(i)

        if len(suspicious_labels) == 0:
            print('cannot find suspicious label')



    def get_potential_triggers(self) :
        module = self.model
        mark_list, mask_list, loss_list = [], [], []
        mark_list_save, mask_list_save = [], []
        criterion = nn.CrossEntropyLoss()
        for label in range(10):
            length = len(str(10))
            mark, mask, loss = self.remask(label, criterion)
            mark_list.append(mark)
            mask_list.append(mask)
            loss_list.append(loss)

            mark_list_save.append(np.array(mark.cpu()))
            mask_list_save.append(np.array(mask.cpu()))

        mark_list = torch.stack(mark_list)
        mask_list = torch.stack(mask_list)
        loss_list = torch.as_tensor(loss_list)
        return mark_list, mask_list, loss_list

    def remask(self, label: int,  criterion):
        self.model.to(self.device)
        self.model.eval()
        atanh_mark = torch.randn(self.input_shape, device=self.device)
        atanh_mark.requires_grad_()
        atanh_mask = torch.randn(self.input_shape[-2:], device=self.device)
        atanh_mask.requires_grad_()
        mask = tanh_func(atanh_mask)  # (h, w)
        mark = tanh_func(atanh_mark)  # (c, h, w)

        optimizer = optim.Adam(
            [atanh_mark, atanh_mask], lr=self.cfg.lr, betas=self.cfg.betas)
        optimizer.zero_grad()

        cost = self.cfg.init_cost
        cost_set_counter = 0
        cost_up_counter = 0
        cost_down_counter = 0
        cost_up_flag = False
        cost_down_flag = False

        # best optimization results
        norm_best = float('inf')
        mask_best = None
        mark_best = None
        entropy_best = None

        # counter for early stop
        early_stop_counter = 0
        early_stop_norm_best = norm_best

        losses = AverageMeter('Loss', ':.4e')
        entropy = AverageMeter('Entropy', ':.4e')
        norm = AverageMeter('Norm', ':.4e')
        acc = AverageMeter('Acc', ':6.2f')

        trainloader = DataLoader(self.dataset,
                                 batch_size=self.cfg.batch_size,
                                 shuffle=True,
                                 pin_memory=True)

        for _epoch in range(self.cfg.epoch):
            losses.reset()
            entropy.reset()
            norm.reset()
            acc.reset()
            trainloader = tqdm(trainloader)

            for (_input, _label) in trainloader:
                batch_size = _label.size(0)
                _input, _label = _input.to(self.device), _label.to(self.device)
                X = _input + mask * (mark - _input)  # = (1 - mask) + mask * mark
                Y = label * torch.ones_like(_label, dtype=torch.long)

                # _,_,_,_,_output = self.model(self.transform(X))
                _output = self.model(self.transform(X))
                batch_acc = Y.eq(_output.argmax(1)).float().mean()
                batch_entropy = criterion(_output, Y)
                batch_norm = mask.norm(p=1)
                batch_loss = batch_entropy + cost * batch_norm

                acc.update(batch_acc.item(), batch_size)
                entropy.update(batch_entropy.item(), batch_size)
                norm.update(batch_norm.item(), batch_size)
                losses.update(batch_loss.item(), batch_size)

                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                mask = tanh_func(atanh_mask)  # (h, w)
                mark = tanh_func(atanh_mark)  # (c, h, w)
                trainloader.set_description_str(f'BinaryEntropy: {batch_loss:.4f}')

            if acc.avg >= self.cfg.attack_succ_threshold and norm.avg < norm_best:
                mask_best = mask.detach()
                mark_best = mark.detach()
                norm_best = norm.avg
                entropy_best = entropy.avg

            # check early stop
            if self.cfg.early_stop:
                # only terminate if a valid attack has been found
                if norm_best < float('inf'):
                    if norm_best >= self.cfg.early_stop_threshold * early_stop_norm_best:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                early_stop_norm_best = min(norm_best, early_stop_norm_best)

                if cost_down_flag and cost_up_flag and early_stop_counter >= self.cfg.early_stop_patience:
                    print('early stop')
                    break

            # check cost modification
            if cost == 0 and acc.avg >= self.cfg.attack_succ_threshold:
                cost_set_counter += 1
                if cost_set_counter >= self.cfg.patience:
                    cost = self.cfg.init_cost
                    cost_up_counter = 0
                    cost_down_counter = 0
                    cost_up_flag = False
                    cost_down_flag = False
                    print('initialize cost to %.2f' % cost)
            else:
                cost_set_counter = 0

            if acc.avg >= self.cfg.attack_succ_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.cfg.patience:
                cost_up_counter = 0
                cost *= self.cfg.cost_multiplier_up
                cost_up_flag = True
            elif cost_down_counter >= self.cfg.patience:
                cost_down_counter = 0
                cost /= self.cfg.cost_multiplier_down
                cost_down_flag = True
            if mask_best is None:
                mask_best = tanh_func(atanh_mask).detach()
                mark_best = tanh_func(atanh_mark).detach()
                norm_best = norm.avg
                entropy_best = entropy.avg
        atanh_mark.requires_grad = False
        atanh_mask.requires_grad = False

        return mark_best, mask_best, entropy_best


