import os

import cv2
import numpy as np
import torch
import torchvision
import torch.distributed as dist
from collections import OrderedDict
import myseg.tv_transform as my_transforms
import myseg.cv2_transform as cv2_transforms
from PIL import Image


# 数据集根路径
# root_dir = '../data/cityscapes'


def read_images_dir(root_dir, folder, is_label=False):
    """
    读取所有图像和标注（的文件路径）

    city_idx ：城市索引 （用于选取某个城市）
    """

    # 读取各城市文件夹名称
    city_names = sorted(os.listdir(os.path.join(root_dir, folder)))
    # print(city_names)

    # 选取某个城市
    # if city_idx is not None:
    #     city_names = [city_names[city_idx]]

    # 读取各城市文件夹中的图片路径
    img_dirs = []
    for city_name in city_names:
        img_names = os.listdir(os.path.join(root_dir, folder, city_name))
        for img_name in img_names:
            if is_label == False:
                img_dirs.append(os.path.join(root_dir, folder, city_name, img_name))
            if is_label == True:  # label只读取以_labelIds.png结尾的文件
                img_dirs.append(os.path.join(root_dir, folder, city_name, img_name))

    img_dirs = sorted(img_dirs)
    list_ = []
    list_2 = []
    last_c = None
    for ii,ll in enumerate(img_dirs):
        cn = ll.split('/')[-2]    
        if cn not in list_:
            list_.append(cn)
        if cn !=last_c:
            list_2.append(ii)
            last_c = cn
    print(list_)
    print(list_2)

    return img_dirs




class CamVid_Dataset(torch.utils.data.Dataset):
    """Cityscapes dataset"""

    # Training dataset root folders
    train_folder = "images/train"
    train_lb_folder = "labels/train"

    # Validation dataset root folders
    val_folder = "images/val"
    val_lb_folder = "labels/val"


    def __init__(self, args,root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split  # 'train', 'val', 'test'
        self.args = args

        # torchvision
        # self.image_transform, self.label_transform = my_transforms.get_transform()


        # 读取图像和标注的文件路径
        if self.split == 'train':
            self.image_dirs, self.label_dirs = \
                read_images_dir(root_dir, self.train_folder), read_images_dir(root_dir, self.train_lb_folder, is_label=True)
        elif self.split == 'val':
            self.image_dirs, self.label_dirs = \
                read_images_dir(root_dir, self.val_folder), read_images_dir(root_dir, self.val_lb_folder, is_label=True)

        assert len(self.image_dirs) == len(self.label_dirs), '图像和标注数量不匹配'
        print('find ' + str(len(self.image_dirs)) + ' examples')



    def __getitem__(self, idx):


        # 读入图片
        if self.args.dataset=='voc':
            image = Image.open(self.image_dirs[idx])
            image = np.array(image)
            label = Image.open(self.label_dirs[idx])
            label = np.array(label)

        else:
            image = cv2.imread(self.image_dirs[idx], cv2.IMREAD_COLOR)[:, :, ::-1]
            label = cv2.imread(self.label_dirs[idx], cv2.IMREAD_GRAYSCALE)
        # print(image.shape, label.shape)  # (1024, 2048, 3) (1024, 2048)

        # 将label进行remap
        #label = self.lb_map[label]
        if self.args.dataset=='camvid':

            if self.split == 'val':  
                label = np.uint8(label)-1
        elif self.args.dataset=='ade20k':
            label = np.uint8(label)-1
        elif self.args.dataset=='voc':
            label[label==255]=0
            label =  np.uint8(label)-1
        scale_ = 512
        if  self.args.dataset=='voc' or self.args.dataset=='ade20k':
            scale_ = 480


        # transform : 同时处理image和label
        image_label = dict(im=image, lb=label)

        if self.split == 'train':
            image_label = cv2_transforms.TransformationTrain(scales=(0.5, 1.5), cropsize=(scale_, scale_))(image_label)

        if self.split == 'val':
            image_label = cv2_transforms.TransformationVal()(image_label)

        # ToTensor
        image_label = cv2_transforms.ToTensor(
            mean=(0.3257, 0.3690, 0.3223),  # city, rgb
            std=(0.2112, 0.2148, 0.2115),
        )(image_label)

        image, label = image_label['im'], image_label['lb']
        # print(image.shape, label.shape) # torch.Size([3, 512, 1024]) torch.Size([512, 1024])

        return (image, label)

    def __len__(self):
        return len(self.image_dirs)


def load_dataiter(root_dir, batch_size, use_DDP=False):
    """加载dataloader"""
    num_workers = 16

    image_transform, label_transform = my_transforms.get_transform()
    train_set = CamVid_Dataset(root_dir, 'train')
    test_set = CamVid_Dataset(root_dir, 'val')

    if use_DDP:
        # DDP：使用DistributedSampler，DDP帮我们把细节都封装起来了。
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        # test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)

        train_iter = torch.utils.data.DataLoader(
            train_set, batch_size, shuffle=False, drop_last=True, num_workers=num_workers, sampler=train_sampler)
        test_iter = torch.utils.data.DataLoader(
            test_set, batch_size, drop_last=True, num_workers=num_workers)
    else:
        train_iter = torch.utils.data.DataLoader(
            train_set, batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        test_iter = torch.utils.data.DataLoader(
            test_set, batch_size, drop_last=True, num_workers=num_workers)

    return train_iter, test_iter


if __name__ == '__main__':

    # train_images = read_images_dir(root_dir, Cityscapes_Dataset.train_folder)
    # train_labels = read_images_dir(root_dir, Cityscapes_Dataset.train_lb_folder, is_label=True)
    # print(train_images[1150:1160])
    # print(train_labels[1150:1160])

    train_iter, test_iter = load_dataiter(root_dir='../data/cityscapes', batch_size=16)
    print("train_iter.len", len(train_iter))
    print("test_iter.len", len(test_iter))
    for i, (images, labels) in enumerate(train_iter):
        if (i < 3):
            print(images.size(), labels.size())  # torch.Size([16, 3, 512, 1024]) torch.Size([16, 512, 1024])
            print(labels.dtype)
            print(labels[0, 245:255, 250:260])
