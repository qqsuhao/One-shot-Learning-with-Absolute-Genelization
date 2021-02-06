# -*- coding:utf8 -*-
# @TIME     : 2020/12/27 19:35
# @Author   : SuHao
# @File     : CNN_AG.py

import torch.nn as nn
import torch


class CNN(nn.Module):
    def __init__(self, imageSize):
        super(CNN, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(2, 4, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

        )
        self.fc1 = nn.Sequential(
            nn.Linear(imageSize**2*8, 500),
            # nn.Tanh(),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            # nn.Tanh(),
            nn.ReLU(inplace=True),
            nn.Linear(500, 1),
            # nn.ReLU(inplace=True),
            # nn.Tanh(),
            nn.Sigmoid(),
        )

    def forward(self, input0, input1):
        inputs = torch.cat([input0, input1], dim=1)
        latent = self.cnn1(inputs)
        outputs = latent.view(latent.size(0), -1)
        outputs = self.fc1(outputs)
        return outputs.view(-1), latent



class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output, label):
        loss_contrastive = torch.mean((1-label) * torch.pow(output, 2) +
                                      (label) * torch.pow(self.margin - output, 2))
        return loss_contrastive
