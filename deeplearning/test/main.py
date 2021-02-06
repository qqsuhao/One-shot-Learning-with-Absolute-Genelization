# -*- coding:utf8 -*-
# @TIME     : 2020/12/25 15:54
# @Author   : SuHao
# @File     : main.py

import os

os.system("workon pytorch-env")
for i in range(10):
    os.system("python ./MLP_AG_test.py")