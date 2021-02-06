# -*- coding:utf8 -*-
# @TIME     : 2020/12/24 19:54
# @Author   : SuHao
# @File     : siamese_MLP.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class SiameseNetwork(nn.Module):
    def __init__(self, imageSize):
        super(SiameseNetwork, self).__init__()
        input_size = imageSize**2
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(inplace=True),
            # nn.Tanh(),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(inplace=True),
            # nn.Tanh(),
            nn.Linear(input_size // 4, input_size // 8),
            # nn.ReLU(inplace=True),
            # nn.Tanh(),
        )


    def forward_once(self, x):
        output = self.linear(x)
        return output

    def forward(self, input1, input2):
        batchsize = input1.size(0)
        output1 = self.forward_once(input1.view(batchsize, -1))
        output2 = self.forward_once(input2.view(batchsize, -1))
        return output1, output2


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive