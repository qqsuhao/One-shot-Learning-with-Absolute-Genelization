# -*- coding:utf8 -*-
# @TIME     : 2020/12/30 14:01
# @Author   : SuHao
# @File     : CNN_AG_train.py

from __future__ import print_function
import os
import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from deeplearning.models.CNN_AG import CNN, ContrastiveLoss
from deeplearning.dataload.dataload import load_dataset
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torchvision.transforms as T


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default=r"../experiments/CNN_AG_train_1", help="path to save experiments results")
parser.add_argument("--dataset", default="ORL", help="mnist")
parser.add_argument('--dataroot', default=r"../data/ORL/train", help='path to dataset')
parser.add_argument("--n_epoches", type=int, default=100, help="number of epoches of training")
parser.add_argument("--batchSize", type=int, default=256, help="size of the batches")
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
trans = T.Compose([T.Resize((opt.imageSize, opt.imageSize)), T.ToTensor(), ])
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
cnn = CNN(opt.imageSize).to(device)
cnn.apply(weights_init)

## adversarial loss
cnn_optimizer = optim.Adam(cnn.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
cnn_criteria = nn.BCELoss()


# record results
writer = SummaryWriter("../runs{0}".format(opt.experiment[1:]), comment=opt.experiment[1:])

# Training
record = 0
with tqdm.tqdm(range(opt.n_epoches)) as t:
    for e in t:
        t.set_description(f"Epoch {e+1} /{opt.n_epoches} Per epoch {train_dataset.__len__()}")
        epoch_Loss = 0
        for inputs0, inputs1, targets in train_dataloader:
            targets = torch.abs(targets)
            inputs0, inputs1, targets = inputs0.to(device), inputs1.to(device), targets.to(device)
            batchsize = inputs0.size(0)
            cnn_optimizer.zero_grad()   # zero the gradient buffers
            output, _ = cnn(inputs0, inputs1)
            loss = cnn_criteria(torch.pow(output, 2), targets.view(batchsize))
            loss.backward()
            cnn_optimizer.step()
            epoch_Loss += loss.item() * batchsize


        ## End of epoch
        epoch_Loss /= opt.dataSize
        t.set_postfix(epoch_Loss=epoch_Loss)
        writer.add_scalar("mse_epoch_loss", epoch_Loss, e)

        if (e+1) % 10 == 0:
            # save model parameters
            torch.save(cnn.state_dict(), '{0}/cnn_{1}.pth'.format(opt.experiment, e))

torch.save(cnn.state_dict(), '{0}/cnn.pth'.format(opt.experiment, e))
writer.close()