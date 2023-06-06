import random
import math
import numpy as np
import cv2
import torch
from torchvision.transforms import Compose


def cv2_get_image(impth, lbpth):
    image = cv2.imread(impth)[:, :, ::-1].copy()
    label = cv2.imread(lbpth, 0)
    return image, label


class TransformationTrain(object):
    """
    Feddrive: We apply a standard data augmentation pipeline

    1、对图像进行随机缩放和随机裁剪
    scale the images in the range (0.5,1.5) for Cityscapes
    then extract a crop with size 512×1024 for Cityscapes

    2、随机翻转图像 (not implemented)
    We point out that when the clients do not have enough samples to
    use 16 as batch size, we virtually double their dataset by flipping the images horizontally.
    """

    def __init__(self, scales, cropsize):
        self.trans_func = Compose([
            RandomResizedCrop(scales, cropsize),
            RandomHorizontalFlip()  # p=0.5
        ])

    def __call__(self, im_lb):
        im_lb = self.trans_func(im_lb)
        return im_lb


class TransformationVal(object):

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        return dict(im=im, lb=lb)


class RandomResizedCrop(object):
    '''
    size should be a tuple of (H, W)
    '''

    def __init__(self, scales=(0.5, 1.), size=(384, 384)):
        self.scales = scales
        self.size = size

    def __call__(self, im_lb):
        if self.size is None:
            return im_lb

        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[:2] == lb.shape[:2]

        crop_h, crop_w = self.size
        scale = np.random.uniform(min(self.scales), max(self.scales))
        im_h, im_w = [math.ceil(el * scale) for el in im.shape[:2]]
        im = cv2.resize(im, (im_w, im_h))
        lb = cv2.resize(lb, (im_w, im_h), interpolation=cv2.INTER_NEAREST)

        if (im_h, im_w) == (crop_h, crop_w): return dict(im=im, lb=lb)
        pad_h, pad_w = 0, 0
        if im_h < crop_h:
            pad_h = (crop_h - im_h) // 2 + 1
        if im_w < crop_w:
            pad_w = (crop_w - im_w) // 2 + 1
        if pad_h > 0 or pad_w > 0:
            im = np.pad(im, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))
            lb = np.pad(lb, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=255)

        im_h, im_w, _ = im.shape
        sh, sw = np.random.random(2)
        sh, sw = int(sh * (im_h - crop_h)), int(sw * (im_w - crop_w))
        return dict(
            im=im[sh:sh + crop_h, sw:sw + crop_w, :].copy(),
            lb=lb[sh:sh + crop_h, sw:sw + crop_w].copy()
        )

class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, im_lb):
        if np.random.random() < self.p:
            return im_lb
        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[:2] == lb.shape[:2]
        return dict(
            im=im[:, ::-1, :],
            lb=lb[:, ::-1],
        )

class ToTensor(object):
    '''
    mean and std should be of the channel order 'bgr'
    '''

    def __init__(self, mean=(0, 0, 0), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        im = im.transpose(2, 0, 1).astype(np.float32)
        im = torch.from_numpy(im).div_(255)
        dtype, device = im.dtype, im.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)[:, None, None]
        std = torch.as_tensor(self.std, dtype=dtype, device=device)[:, None, None]
        im = im.sub_(mean).div_(std).clone()
        if not lb is None:
            lb = torch.from_numpy(lb.astype(np.int64).copy()).clone()
        return dict(im=im, lb=lb)
