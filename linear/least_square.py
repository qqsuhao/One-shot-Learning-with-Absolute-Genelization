# -*- coding:utf8 -*-
# @TIME     : 2020/12/16 15:42
# @Author   : SuHao
# @File     : least_square.py

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from sklearn.decomposition import PCA
from scipy.stats import ortho_group  # Requires version 0.18 of scipy


def softmax(x, axis=1):
    # 计算每行的最大值
    row_max = x.max(axis=axis)

    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    row_max=row_max.reshape(-1, 1)
    x = x - row_max

    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s

## generate samples
num = 10000
dim = 2
mean1 = np.random.randn(1, dim)
mean2 = np.random.randn(1, dim)

T = ortho_group.rvs(dim=dim)
x1_w1 = np.random.randn(num, dim) @ T + mean1
x1_w2 = np.random.randn(num, dim) @ T + mean2
bias = np.random.randn(1, dim)
T_A = ortho_group.rvs(dim=dim)
x2_A = np.random.randn(num, dim) @ T_A + bias
x_w1_A = np.concatenate([x1_w1, x2_A], axis=1)
x2_A = np.random.randn(num, dim) @ T_A + bias
x_w2_A = np.concatenate([x1_w2, x2_A], axis=1)


x1_w1 = np.random.randn(num, dim) @ T + mean1
x1_w2 = np.random.randn(num, dim) @ T + mean2
bias = 10 * np.random.randn(1, dim)
x2_B = np.random.randn(num, dim) + bias
x_w1_B = np.concatenate([x1_w1, x2_B], axis=1)
x2_B = np.random.randn(num, dim) + bias
x_w2_B = np.concatenate([x1_w2, x2_B], axis=1)


T = ortho_group.rvs(dim=2*dim)      # transformation matrix
x_w1_A = x_w1_A @ T
x_w2_A = x_w2_A @ T
# T = ortho_group.rvs(dim=2*dim)      # transformation matrix
x_w1_B = x_w1_B @ T
x_w2_B = x_w2_B @ T


## siamese least square
A1 = np.concatenate([x_w1_B, x_w2_B], axis=1)
A2 = np.concatenate([x_w2_B, x_w1_B], axis=1)
A3 = np.concatenate([x_w1_B, x_w1_B], axis=1)
A4 = np.concatenate([x_w2_B, x_w2_B], axis=1)
A = np.concatenate([A1, A2, A3, A4], axis=0)
b1 = np.concatenate([np.zeros((num, 1))+1], axis=1)
b2 = np.concatenate([np.zeros((num, 1))-1], axis=1)
b3 = np.concatenate([np.zeros((num, 1))], axis=1)
b4 = np.concatenate([np.zeros((num, 1))], axis=1)
b = np.concatenate([b1, b2, b3, b4], axis=0)
theta_D, _, _, _ = lstsq(A, b)
print(theta_D)


# generalize
A1 = np.concatenate([x_w2_A, x_w2_A], axis=1)
b = A1 @ theta_D
np.set_printoptions(threshold=20)
print(list(b.flatten()))

# ##
def decomposition(X):
    eva, evc = np.linalg.eig(X)
    index = np.argsort(eva)
    index = index[::-1]
    eva = eva[index]
    evc = evc[:, index]
    return eva, evc
S_diff = A1.T @ A1 / num /2  + A2.T @ A2/ num /2
S_iden = A3.T @ A3/ num /2 + A4.T @ A4/ num /2
S = S_diff + S_iden
eva, evc = decomposition(S)
eva_inv = np.sqrt(1/eva)
evc = evc @ np.diag(eva_inv)
S1 = evc.T @ (S_diff) @ evc
eva1, evc1 = decomposition(S1)
A1 = np.concatenate([x_w1_A, x_w1_A], axis=1)
b = A1 @ evc @ evc1[:, 0:1]
print(evc @ evc1[:, 0:1])
np.set_printoptions(threshold=1000)
print(list(b.flatten()))








