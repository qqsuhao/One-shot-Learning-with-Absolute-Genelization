# -*- coding:utf8 -*-
# @TIME     : 2020/12/28 17:07
# @Author   : SuHao
# @File     : plot_tsne.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torchvision.utils as vutils



def plot_embedding(data, label, title, path):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    fig, ax = plt.subplots(dpi=300)
    scatter = ax.scatter(data[:, 0], data[:, 1], marker='.', c=label, cmap='Paired', linewidth=1)
    legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    ax.add_artist(legend1)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.savefig(path)
    plt.close()
    # return fig


def plot_3D(data, label, title, path):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure(dpi=300)
    ax = plt.axes(projection='3d')
    ax.grid(False)
    # ax.set_axis_off()
    scatter = ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], marker='.', c=label, cmap='Set1')
    # legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()
    # plt.savefig(path)
    # plt.close()


def plot_tsne(data, label, title, path):
    tsne = TSNE(n_components=3, init='pca', random_state=0)
    result = tsne.fit_transform(data)
    plot_3D(result, label, title, path)


def plot_2D_scatter(data, label, title, path):
    if len(data.shape) != 2:
        return
    if data.shape[1] > 2:
        return
    fig, ax = plt.subplots(dpi=300)
    scatter = ax.scatter(data[:, 0], data[:, 1], marker='.', c=label, cmap='tab10')
    legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    ax.add_artist(legend1)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.savefig(path)
    plt.close()



import os
import numpy as np
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as T
from deeplearning.dataload.dataload import load_dataset
from deeplearning.dataload.self_transforms import Fllip, AddGaussianNoise, AddSaltPepperNoise, Style
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default=r"../experiments/MLP_AG_test", help="path to save experiments results")
parser.add_argument("--dataset", default="mnist_siamese", help="mnist")
parser.add_argument('--dataroot', default=r"../data", help='path to dataset')
parser.add_argument("--batchSize", type=int, default=512, help="size of the batches")
parser.add_argument("--size", type=int, default=28, help="size of image after scaled")
parser.add_argument("--imageSize", type=int, default=28, help="size of each image dimension")
opt = parser.parse_args()
os.makedirs(opt.experiment, exist_ok=True)

##
trans = T.Compose([T.Resize(opt.imageSize),  T.ToTensor(), T.Normalize(0.5, 0.5)])
test_dataset = load_dataset(opt.dataroot, opt.dataset, opt.imageSize, trans=trans, train=False)
test_dataloader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)
i, (test_inputs0, test_inputs1, targets) = next(enumerate(test_dataloader))
batchsize = test_inputs0.size(0)
test_inputs_raw = torch.cat([test_inputs0.view(batchsize, -1), test_inputs1.view(batchsize, -1)], dim=1)
test_inputs_raw = test_inputs_raw.detach().numpy()
targets_raw = list(torch.abs(targets).detach().numpy())



##
# trans = T.Compose([T.Resize(opt.imageSize),  Style(2), T.ToTensor(), T.Normalize(0.5, 0.5)])
# test_dataset = load_dataset(opt.dataroot, opt.dataset, opt.imageSize, trans=trans, train=False)
# test_dataloader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)
# i, (test_inputs0, test_inputs1, targets) = next(enumerate(test_dataloader))
# batchsize = test_inputs0.size(0)
# test_inputs_modified = torch.cat([test_inputs0.view(batchsize, -1), test_inputs1.view(batchsize, -1)], dim=1)
# test_inputs_modified = test_inputs_modified.detach().numpy()
# targets_modified = list(torch.abs(targets).detach().numpy())
# targets_modified = [2+i for i in targets_modified]
# data = np.concatenate([test_inputs_raw, test_inputs_modified], axis=0)
# label = targets_raw + targets_modified
# vutils.save_image(torch.cat([test_inputs0[0:1, 0:1, :, :], test_inputs1[0:1, 0:1, :, :]], dim=3),
#                   '{0}/{1}-example.png'.format(opt.experiment, targets[0].item()), pad_value=2)
# ##
# trans = T.Compose([T.Resize(opt.imageSize), Style(1), T.ToTensor(), T.Normalize(0.5, 0.5)])
# test_dataset = load_dataset(opt.dataroot, opt.dataset, opt.imageSize, trans=trans, train=False)
# test_dataloader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)
# i, (test_inputs0, test_inputs1, targets) = next(enumerate(test_dataloader))
# batchsize = test_inputs0.size(0)
# test_inputs_modified = torch.cat([test_inputs0.view(batchsize, -1), test_inputs1.view(batchsize, -1)], dim=1)
# test_inputs_modified = test_inputs_modified.detach().numpy()
# targets_modified = list(torch.abs(targets).detach().numpy())
# targets_modified = [4+i for i in targets_modified]
# data = np.concatenate([data, test_inputs_modified], axis=0)
# label = label + targets_modified


##
# trans = T.Compose([T.Resize(opt.imageSize),  Fllip(), T.ToTensor(), T.Normalize(0.5, 0.5)])
# test_dataset = load_dataset(opt.dataroot, opt.dataset, opt.imageSize, trans=trans, train=False)
# test_dataloader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)
# i, (test_inputs0, test_inputs1, targets) = next(enumerate(test_dataloader))
# batchsize = test_inputs0.size(0)
# test_inputs_modified = torch.cat([test_inputs0.view(batchsize, -1), test_inputs1.view(batchsize, -1)], dim=1)
# test_inputs_modified = test_inputs_modified.detach().numpy()
# targets_modified = list(torch.abs(targets).detach().numpy())
# targets_modified = [6+i for i in targets_modified]
# data = np.concatenate([data, test_inputs_modified], axis=0)
# label = label + targets_modified
#
# ##
# trans = T.Compose([T.Resize(opt.imageSize),  AddSaltPepperNoise(0.2), T.ToTensor(), T.Normalize(0.5, 0.5)])
# test_dataset = load_dataset(opt.dataroot, opt.dataset, opt.imageSize, trans=trans, train=False)
# test_dataloader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)
# i, (test_inputs0, test_inputs1, targets) = next(enumerate(test_dataloader))
# batchsize = test_inputs0.size(0)
# test_inputs_modified = torch.cat([test_inputs0.view(batchsize, -1), test_inputs1.view(batchsize, -1)], dim=1)
# test_inputs_modified = test_inputs_modified.detach().numpy()
# targets_modified = list(torch.abs(targets).detach().numpy())
# targets_modified = [8+i for i in targets_modified]
# data = np.concatenate([data, test_inputs_modified], axis=0)
# label = label + targets_modified
#
# ##
# trans = T.Compose([T.Resize(opt.imageSize),  AddGaussianNoise(0.9), T.ToTensor(), T.Normalize(0.5, 0.5)])
# test_dataset = load_dataset(opt.dataroot, opt.dataset, opt.imageSize, trans=trans, train=False)
# test_dataloader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)
# i, (test_inputs0, test_inputs1, targets) = next(enumerate(test_dataloader))
# batchsize = test_inputs0.size(0)
# test_inputs_modified = torch.cat([test_inputs0.view(batchsize, -1), test_inputs1.view(batchsize, -1)], dim=1)
# test_inputs_modified = test_inputs_modified.detach().numpy()
# targets_modified = list(torch.abs(targets).detach().numpy())
# targets_modified = [10+i for i in targets_modified]
# data = np.concatenate([data, test_inputs_modified], axis=0)
# label = label + targets_modified

plot_tsne(test_inputs_raw, targets_raw, "tsne", path=opt.experiment+"./tsne.png")






