# -*- coding:utf8 -*-
# @TIME     : 2020/12/7 10:12
# @Author   : SuHao
# @File     : dataload.py

import torchvision.transforms as T
import torchvision.datasets as dset
from deeplearning.dataload.loadmydataset import DataProcessingMnist, SiameseNetworkDataset, OmniglotTrain, OmniglotTest, myMNIST
from torch.utils.data import DataLoader
import os

def load_dataset(dataroot, dataset_name, imageSize, trans, train=True):
    params_med = {"dataroot": dataroot, "split": 'train' if train else 'test', "transform":trans}
    if dataset_name == "mnist":
        dataset = myMNIST(root=dataroot,
                             train=train,
                             download=True,
                             transform=trans,
                             num=[4, 9])
    elif dataset_name == "mnist_siamese":
        dataset = DataProcessingMnist(root=dataroot,
                                      train=train,
                                      transform=trans,
                                      download=True,
                                      num=[4, 9])
    elif dataset_name == "ORL":
        imagefolder = dset.ImageFolder(root=dataroot)
        dataset = SiameseNetworkDataset(imageFolderDataset=imagefolder, transform=trans, should_invert=False)

    return dataset


def get_train_loader(dataroot, batchSize, num_train, trans=None, shuffle=False, num_workers=1):
    # train_dataset = dset.ImageFolder(root=dataroot)
    train_dataset = OmniglotTrain(dataroot, num_train, trans)
    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=shuffle, num_workers=num_workers)

    return train_loader


def get_test_loader(dataroot, way, trials, seed=0, num_workers=0, trans=None):
    # test_dataset = dset.ImageFolder(root=dataroot)
    test_dataset = OmniglotTest(dataroot, trials=trials, way=way, seed=seed, trans=trans)
    test_loader = DataLoader(test_dataset, batch_size=way, shuffle=False, num_workers=num_workers)

    return test_loader