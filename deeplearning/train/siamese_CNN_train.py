# -*- coding:utf8 -*-
# @TIME     : 2020/12/30 10:26
# @Author   : SuHao
# @File     : siamese_CNN_train.py

from __future__ import print_function
import os
import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from deeplearning.models.siamese_CNN import SiameseNetwork, ContrastiveLoss
from deeplearning.dataload.dataload import load_dataset
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torchvision.transforms as T


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default=r"../experiments/siamese_CNN_train", help="path to save experiments results")
parser.add_argument("--dataset", default="ORL", help="mnist")
parser.add_argument('--dataroot', default=r"../data/ORL/train", help='path to dataset')
parser.add_argument("--n_epoches", type=int, default=100, help="number of epoches of training")
parser.add_argument("--batchSize", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--imageSize", type=int, default=100, help="size of each image dimension")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between image sampling")
opt = parser.parse_args()
print(opt)
os.makedirs(opt.experiment, exist_ok=True)

## random seed
# opt.seed = 42
# torch.manual_seed(opt.seed)
# np.random.seed(opt.seed)

## cudnn
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

## device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

## dataset
trans = T.Compose([T.Resize((opt.imageSize, opt.imageSize)), T.ToTensor()])
train_dataset = load_dataset(opt.dataroot, opt.dataset, opt.imageSize, trans=trans, train=True)
train_dataloader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)
opt.dataSize = train_dataset.__len__()

## model
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

## model
siamese = SiameseNetwork(opt.imageSize).to(device)
siamese.apply(weights_init)

## adversarial loss
siamese_optimizer = optim.Adam(siamese.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
siamese_criteria = ContrastiveLoss()

# record results
writer = SummaryWriter("../runs{0}".format(opt.experiment[1:]), comment=opt.experiment[1:])

# Training
record = 0
with tqdm.tqdm(range(opt.n_epoches)) as t:
    for e in t:
        t.set_description(f"Epoch {e+1} /{opt.n_epoches} Per epoch {train_dataset.__len__()}")
        epoch_contrastiveLoss = 0
        for inputs0, inputs1, targets in train_dataloader:
            targets = torch.abs(targets)
            inputs0, inputs1, targets = inputs0.to(device), inputs1.to(device), targets.to(device)
            batchsize = inputs0.size(0)
            siamese_optimizer.zero_grad()   # zero the gradient buffers
            output0, output1 = siamese(inputs0, inputs1)
            loss_contrastive = siamese_criteria(output0, output1, targets)
            loss_contrastive.backward()
            siamese_optimizer.step()
            epoch_contrastiveLoss += loss_contrastive.item() * batchsize


        ## End of epoch
        epoch_contrastiveLoss /= opt.dataSize
        t.set_postfix(epoch_contrastiveLoss=epoch_contrastiveLoss)
        writer.add_scalar("mse_epoch_loss", epoch_contrastiveLoss, e)

        if (e+1) % 10 == 0:
            # save model parameters
            torch.save(siamese.state_dict(), '{0}/siamese_{1}.pth'.format(opt.experiment, e))

torch.save(siamese.state_dict(), '{0}/siamese.pth'.format(opt.experiment, e))
writer.close()