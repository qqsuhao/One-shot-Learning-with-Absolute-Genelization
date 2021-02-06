# -*- coding:utf8 -*-
# @TIME     : 2020/12/22 10:02
# @Author   : SuHao
# @File     : decomposition.py


import numpy as np
from scipy.stats import ortho_group  # Requires version 0.18 of scipy
import itertools
import matplotlib.pyplot as plt
from scipy.linalg import lstsq



def generate_samples(num, option, mean_back1, mean_back2, cov_back1, cov_back2, T):
    '''
    :param num: 样本数量
    :param mean1: 第一类样本均值
    :param mean2: 第二类样本均值
    :param mean_back1: A类数据集背景均值
    :param mean_back2: B类数据集背景均值
    :param cov1: 第一类样本协方差
    :param cov2: 第二类样本协方差
    :param cov_back1: A类数据集背景协方差
    :param cov_back2: B类数据集背景协方差
    :param T: 变换矩阵
    :return:
    '''
    dim = 8
    if option == 'I-I':     # Bayes error 10%
        mean1 = np.zeros(dim)
        cov1 = np.eye(dim)
        mean2 = np.zeros(dim)
        mean2[0] = 2.56
        cov2 = np.eye(dim)
    elif option == 'I-4I':      # Bayes error 9%
        mean1 = np.zeros(dim)
        cov1 = np.eye(dim)
        mean2 = np.zeros(dim)
        cov2 = np.eye(dim) * 4
    elif option == 'I-Lambda':      # Bayes error 1.9%
        mean1 = np.zeros(dim)
        cov1 = np.eye(dim)
        mean2 = np.array([3.86,3.10,0.84,0.84,1.64,1.08,0.26,0.01])
        cov2 = np.diag([8.41,12.06,0.12,0.22,1.49,1.77,0.35,2.73])


    z1_w1 = np.random.multivariate_normal(mean1, cov1, num)
    z1_w2 = np.random.multivariate_normal(mean2, cov2, num)
    z2_A = np.random.multivariate_normal(mean_back1, cov_back1, num)
    z_w1_A = np.concatenate([z1_w1, z2_A], axis=1)
    z2_A = np.random.multivariate_normal(mean_back1, cov_back1, num)
    z_w2_A = np.concatenate([z1_w2, z2_A], axis=1)

    z1_w1 = np.random.multivariate_normal(mean1, cov1, num)
    z1_w2 = np.random.multivariate_normal(mean2, cov2, num)
    z2_B = np.random.multivariate_normal(mean_back2, cov_back2, num)
    z_w1_B = np.concatenate([z1_w1, z2_B], axis=1)
    z2_B = np.random.multivariate_normal(mean_back2, cov_back2, num)
    z_w2_B = np.concatenate([z1_w2, z2_B], axis=1)

    x_w1_A = z_w1_A @ T
    x_w2_A = z_w2_A @ T
    x_w1_B = z_w1_B @ T
    x_w2_B = z_w2_B @ T

    return x_w1_A, x_w2_A, x_w1_B, x_w2_B


option = 'I-Lambda'
dim = 8
mean_back1 = np.zeros(dim)
cov_back1 = np.diag(np.abs(np.random.randn(dim)))
mean_back2 = np.random.randn(dim)
cov_back2 = np.diag(np.abs(np.random.randn(dim)))
T = ortho_group.rvs(dim=8+dim)      # transformation matrix
# T = np.eye(8+dim)
num = 100
x_w1_A, x_w2_A, x_w1_B, x_w2_B = generate_samples(num, option, mean_back1,
                                                  mean_back2, cov_back1, cov_back2, T)

## 组合样本
def product(X):
    X_iden = [i for i in itertools.product(x_w1_A, repeat=2)]
    X_iden = np.array(X_iden)
    X_iden = X_iden.reshape((X_iden.shape[0], -1))
    return X_iden

X_iden_1 = product(x_w1_A)
X_iden_2 = product(x_w2_A)
X_iden = np.concatenate([X_iden_1, X_iden_2], axis=0)

def permutations(X1, X2):
    size1 = X1.shape[0]
    size2 = X2.shape[0]
    X1 = np.tile(X1, (1, size2)).reshape((-1, X1.shape[1]))
    X2 = np.tile(X2, (size1, 1))
    X_diff = np.concatenate([X1, X2], axis=1)
    return X_diff

X_diff_1 = permutations(x_w1_A, x_w2_A)
X_diff_2 = permutations(x_w2_A, x_w1_A)
X_diff = np.concatenate([X_diff_1, X_diff_2], axis=0)

## decomposition
def decomposition(X):
    eva, evc = np.linalg.eig(X)
    index = np.argsort(eva)
    index = index[::-1]
    eva = eva[index]
    evc = evc[:, index]
    return eva, evc

S_diff = X_diff.T @ X_diff
print(np.linalg.matrix_rank(S_diff))
eva, evc = decomposition(S_diff/X_diff.shape[0])
plt.figure()
plt.plot(eva)
plt.title("eigenvalues of S_diff")
plt.show()


S_iden = X_iden.T @ X_iden
S = S_iden/X_iden.shape[0] + S_diff/X_diff.shape[0]
eva, evc = decomposition(S)
print(eva)
plt.figure()
plt.plot(eva)
plt.title("eigenvalues of S_iden+S_diff")
plt.show()
eva_inv = np.sqrt(1/eva)
evc = evc @ np.diag(eva_inv)
S1 = evc.T @ (S_diff/X_diff.shape[0]) @ evc
eva1, evc1 = decomposition(S1)
plt.figure()
plt.plot(eva1)
plt.title("eigenvalues")
plt.show()
print(eva1)

evc_big = evc @ evc1[:, 0:1]
#
#
X_diff_B = permutations(x_w1_A, x_w2_A)
X_iden_B = np.concatenate([product(x_w1_A), product(x_w2_A)], axis=0)
X_diff_B_trans = X_diff_B @ evc_big
X_iden_B_trans = X_iden_B @ evc_big
#
plt.figure()
plt.scatter(np.arange(0, X_diff_B_trans.shape[0]), X_diff_B_trans.flatten())
plt.scatter(np.arange(0, X_iden_B_trans.shape[0]), X_iden_B_trans.flatten())
plt.show()

#
# X = np.concatenate([X_iden, X_diff], axis=0)
# b = np.concatenate([np.zeros((X_iden.shape[0], 1)), np.ones((X_diff.shape[0], 1))], axis=0)
# theta, _, _, _ = lstsq(X, b)
#
#
# X_test = np.concatenate([X_iden_B, X_diff_B], axis=0)
# b_test = X_test @ theta
# plt.figure()
# plt.plot(b_test)
# plt.show()