# -*- coding:utf8 -*-
# @TIME     : 2021/1/3 15:34
# @Author   : SuHao
# @File     : siamese_shot_train.py

"""
reference:  git@github.com:fangpin/siamese-pytorch.git
"""

import numpy as np
import torch
import argparse
import os
import tqdm
import torch.nn as nn
from torch import optim
from tensorboardX import SummaryWriter
import torchvision.transforms as T
from torch.utils.data import DataLoader
from deeplearning.dataload.dataload import get_train_loader
from deeplearning.models.siamese_shot import Siamese


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default=r"../experiments/siamese_shot_train", help="path to save experiments results")
parser.add_argument('--dataroot', default=r"../data/images_background", help='path to dataset')
parser.add_argument("--n_epoches", type=int, default=100, help="number of epoches of training")
parser.add_argument("--batchSize", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between image sampling")
# parser.add_argument("--way", type=int, default=20, help="how much way one-shot learning")
# parser.add_argument("--times", type=int, default=400, help="number of samples to test accuracy")
parser.add_argument("--workers", type=int, default=0, help="number of dataLoader workers")
parser.add_argument("--augment", type=bool, default=False, help="whether use data augmentation")
parser.add_argument("--num_train", type=int, default=64000)
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

##
trans = T.Compose([T.ToTensor()])  # T.RandomAffine(15),
train_dataloader = get_train_loader(opt.dataroot, opt.batchSize, opt.num_train,
                                    trans=trans, shuffle=True, num_workers=opt.workers)

## model
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.kaiming_normal_(m.weight, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

## model
siamese = Siamese().to(device)
siamese.apply(weights_init)
siamese.train()

## loss
siamese_optimizer = optim.Adam(siamese.parameters(), lr=opt.lr)
siamese_criteria = nn.BCEWithLogitsLoss(reduction='mean')

# record results
writer = SummaryWriter("../runs{0}".format(opt.experiment[1:]), comment=opt.experiment[1:])

# Training
record = 0
with tqdm.tqdm(range(opt.n_epoches)) as t:
    for e in t:
        t.set_description(f"Epoch {e+1} /{opt.n_epoches} Per epoch {opt.num_train}")
        epoch_Loss = 0
        for inputs0, inputs1, targets in train_dataloader:
            inputs0, inputs1, targets = inputs0.to(device), inputs1.to(device), targets.to(device)
            batchsize = inputs0.size(0)
            siamese_optimizer.zero_grad()
            output = siamese(inputs0, inputs1)
            loss = siamese_criteria(output, targets)
            loss.backward()
            siamese_optimizer.step()
            epoch_Loss += loss.item() * batchsize

        # End of epoch
        epoch_Loss /= opt.num_train
        t.set_postfix(epoch_Loss=epoch_Loss)
        writer.add_scalar("epoch_loss", epoch_Loss, e)

        if (e+1) % 10 == 0:
            # save model parameters
            torch.save(siamese.state_dict(), '{0}/siamese_{1}.pth'.format(opt.experiment, e))

torch.save(siamese.state_dict(), '{0}/siamese.pth'.format(opt.experiment, e))
writer.close()
