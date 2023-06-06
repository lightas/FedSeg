import torch
import torch.utils.data
from torch import nn
import torch.distributed as dist
import errno
import os
import datetime


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    loss = 0
    confmat = ConfusionMatrix(num_classes)
    header = 'Test:'
    with torch.no_grad():
        for image, target in data_loader:
            image, target = image.to(device), target.to(device)

            # bisenetv2:
            model.aux_mode = 'eval'
            output = model(image)[0] # return logits
            model.aux_mode = 'train'

            # lraspp_mobilenetv3:
            # output = model(image)
            # output = output['out']

            # [optional] return loss
            # batch_loss = criterion(output, target)
            # loss += batch_loss.item()

            confmat.update(target.flatten(), output.argmax(1).flatten())

        #confmat.reduce_from_all_processes()
        confmat.compute()

    return confmat


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None
        self.acc_global = 0
        self.iou_mean = 0
        self.acc = 0
        self.iu = 0

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        ''' compute and update self metrics '''
        h = self.mat.float()
        self.acc_global = torch.diag(h).sum() / h.sum()
        self.acc_global = self.acc_global.item() * 100
        self.acc = torch.diag(h) / h.sum(1)
        self.iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        # remove nan for calculating mean value
        iu = self.iu[~self.iu.isnan()]
        self.iou_mean = iu.mean().item() * 100
        # return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
            self.acc_global,
            ['{:.1f}'.format(i) for i in (self.acc * 100).tolist()],
            ['{:.1f}'.format(i) for i in (self.iu * 100).tolist()],
            self.iou_mean)


# def mkdir(path):
#     try:
#         os.makedirs(path)
#     except OSError as e:
#         if e.errno != errno.EEXIST:
#             raise
