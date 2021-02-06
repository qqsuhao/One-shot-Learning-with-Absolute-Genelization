# -*- coding:utf8 -*-
# @TIME     : 2020/12/20 14:38
# @Author   : SuHao
# @File     : mnist_linear.py

import os
from dataload.dataload import load_dataset
import torchvision.transforms as T
from torch.utils.data import DataLoader
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default=r"../experiments/mnist-linear", help="path to save experiments results")
parser.add_argument("--dataset", default="mnist", help="mnist")
parser.add_argument('--dataroot', default=r"../data", help='path to dataset')
parser.add_argument("--imageSize", type=int, default=28, help="size of each image dimension")
opt = parser.parse_args()
print(opt)
os.makedirs(opt.experiment, exist_ok=True)

trans = T.Compose([T.ToTensor(), T.Normalize(mean=(0.5),std=(0.5))])
train_dataset = load_dataset(opt.dataroot, opt.dataset, opt.imageSize, trans=trans, train=True)
train_data = train_dataset.train_data
x_train = train_data.numpy().reshape(train_data.size(0), -1)
x_test = train_dataset.train_labels.numpy()
test_data = train_dataset.test_data
x_test = test_data.numpy().reshape(test_data.size(0), -1)
y_test = train_dataset.test_labels.numpy()


