# -*- coding:utf8 -*-
# @TIME     : 2020/12/21 21:33
# @Author   : SuHao
# @File     : MLP_AG.py

import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, common_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            # nn.Linear(input_size, input_size // 2),
            # nn.ReLU(inplace=True),
            # nn.Linear(input_size // 2, input_size // 4),
            # nn.ReLU(inplace=True),
            # nn.Linear(input_size // 4, input_size // 8),
            # nn.ReLU(inplace=True),
            # nn.Linear(input_size // 8, common_size),
            # nn.Sigmoid(),

            nn.Linear(input_size, input_size // 2),
            nn.Tanh(),
            nn.Linear(input_size // 2, input_size // 4),
            nn.Tanh(),
            nn.Linear(input_size // 4, input_size // 8),
            nn.Tanh(),
            nn.Linear(input_size // 8, common_size),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.linear(x)
        return out.view(-1)