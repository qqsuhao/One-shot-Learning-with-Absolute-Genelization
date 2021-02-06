# -*- coding:utf8 -*-
# @TIME     : 2020/12/20 20:18
# @Author   : SuHao
# @File     : loadmydataset.py

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import numpy as np
from torchvision.datasets import MNIST
import PIL
import random
from random import Random
import Augmentor


class myMNIST(MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, num=[0, 1]):
        super(myMNIST, self).__init__(root=root, train=train, transform=transform,
                                                  target_transform=target_transform, download=download)
        self.index = [index for index in range(len(self.targets)) if self.targets[index] in num]
        self.num = np.array(num)


    def __getitem__(self, inx):
        image0 = self.data[self.index[inx]]
        image0 = Image.fromarray(image0.numpy(), mode='L')
        label = np.where(self.num == self.targets[self.index[inx]].item())[0]
        if self.transform is not None:
            image0 = self.transform(image0)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image0, torch.from_numpy(np.array(label, dtype=np.float32))


    def __len__(self):
        return len(self.index)



class DataProcessingMnist(MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, num=[0, 1]):
        super(DataProcessingMnist, self).__init__(root=root, train=train, transform=transform,
                                                  target_transform=target_transform, download=download)
        self.index = [index for index in range(len(self.targets)) if self.targets[index] in num]

    def __getitem__(self, inx):
        index = np.random.choice(self.index, 2)
        label = int(self.targets[index[0]]) - int(self.targets[index[1]])
        if label > 0:
            label = 1
        elif label == 0:
            label = 0
        elif label < 0:
            label = -1
        image0 = self.data[index[0]]
        image1 = self.data[index[1]]

        image0 = Image.fromarray(image0.numpy(), mode='L')
        image1 = Image.fromarray(image1.numpy(), mode='L')

        if self.transform is not None:
            image0 = self.transform(image0)
            image1 = self.transform(image1)

        if self.target_transform is not None:
            label = self.target_transform(label)

        # img = torch.cat([image0, image1], dim=0)
        return image0, image1, torch.from_numpy(np.array([label], dtype=np.float32))


    def __len__(self):
        return len(self.index)*2



class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert


    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1)
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                #keep looping till a different class image is found

                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        label = int(img1_tuple[1]-img0_tuple[1])
        if label > 0:
            label = 1
        elif label == 0:
            label = 0
        elif label < 0:
            label = -1
        return img0, img1, torch.from_numpy(np.array([label], dtype=np.float32))


    def __len__(self):
        return len(self.imageFolderDataset.imgs)*2



# adapted from https://github.com/fangpin/siamese-network
class OmniglotTrain(Dataset):
    def __init__(self, dataroot, num_train, trans):
        super(OmniglotTrain, self).__init__()
        np.random.seed(0)
        # self.dataset = dataset
        self.transform = trans
        self.num_train = num_train
        self.datas, self.num_classes = self.loadToMem(dataroot)

    def loadToMem(self, dataPath):
        # print("begin loading training dataset to memory")
        datas = {}
        # agrees = [0, 90, 180, 270]
        idx = 0
        # for agree in agrees:
        for alphaPath in os.listdir(dataPath):
            for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                datas[idx] = []
                for samplePath in os.listdir(os.path.join(dataPath, alphaPath, charPath)):
                    filePath = os.path.join(dataPath, alphaPath, charPath, samplePath)
                    # datas[idx].append(Image.open(filePath).rotate(agree).convert('L'))
                    datas[idx].append(Image.open(filePath).convert('L'))
                idx += 1
        # print("finish loading training dataset to memory")
        return datas, idx

    def __len__(self):
        return self.num_train

    def __getitem__(self, index):
        # image1 = random.choice(self.dataset.imgs)
        label = None
        img1 = None
        img2 = None
        # get image from same class
        if index % 2 == 1:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx1])
        # get image from different class
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx2])

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))



class OmniglotTest(Dataset):
    def __init__(self, dataroot, trials, way, seed=0, trans=None):
        np.random.seed(1)
        super(OmniglotTest, self).__init__()
        self.transform = trans
        self.times = trials
        self.way = way
        self.img1 = None
        self.c1 = None
        self.datas, self.num_classes = self.loadToMem(dataroot)

    def loadToMem(self, dataPath):
        print("begin loading test dataset to memory")
        datas = {}
        idx = 0
        for alphaPath in os.listdir(dataPath):
            for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                datas[idx] = []
                for samplePath in os.listdir(os.path.join(dataPath, alphaPath, charPath)):
                    filePath = os.path.join(dataPath, alphaPath, charPath, samplePath)
                    datas[idx].append(Image.open(filePath).convert('L'))
                idx += 1
        print("finish loading test dataset to memory")
        return datas, idx

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        label = None
        # generate image pair from same class
        if idx == 0:
            self.c1 = random.randint(0, self.num_classes - 1)
            self.img1 = random.choice(self.datas[self.c1])
            img2 = random.choice(self.datas[self.c1])
        # generate image pair from different class
        else:
            c2 = random.randint(0, self.num_classes - 1)
            while self.c1 == c2:
                c2 = random.randint(0, self.num_classes - 1)
            img2 = random.choice(self.datas[c2])

        if self.transform:
            img1 = self.transform(self.img1)
            img2 = self.transform(img2)
        return img1, img2
