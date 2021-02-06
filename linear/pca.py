# -*- coding:utf8 -*-
# @TIME     : 2020/12/22 18:11
# @Author   : SuHao
# @File     : pca.py

import numpy as np
from scipy.stats import ortho_group
from scipy.linalg import lstsq


x_iden_1 = np.random.multivariate_normal(np.array([2,1,2,1]), np.diag([1, 1, 1, 1]), 10000)
x_iden_2 = np.random.multivariate_normal(np.array([1,2,1,2]), np.diag([1, 1, 1, 1]), 10000)
x_diff_1 = np.random.multivariate_normal(np.array([2,1,1,2]), np.diag([1, 1, 1, 1]), 10000)
x_diff_2 = np.random.multivariate_normal(np.array([1,2,2,1]), np.diag([1, 1, 1, 1]), 10000)
y_iden_1 = np.random.multivariate_normal(np.array([4,-1,4,-1]), np.diag([1, 1, 1, 1]), 10000)
y_iden_2 = np.random.multivariate_normal(np.array([3,0,3,0]), np.diag([1, 1, 1, 1]), 10000)
y_diff_2 = np.random.multivariate_normal(np.array([3,0,4,-1]), np.diag([1, 1, 1, 1]), 10000)
y_diff_1 = np.random.multivariate_normal(np.array([4,-1,3,0]), np.diag([1, 1, 1, 1]), 10000)


## ls
b = np.zeros((30000, 1))
b[20000:, 0] = 1
theta, _, _, _ = lstsq(np.concatenate([x_iden_1, x_iden_2, x_diff_1], axis=0), b)
print(theta)

##
def decomposition(X):
    eva, evc = np.linalg.eig(X)
    index = np.argsort(eva)
    index = index[::-1]
    eva = eva[index]
    evc = evc[:, index]
    return eva, evc
S_iden = x_iden_1.T @ x_iden_1 / 20000 + x_iden_2.T @ x_iden_2 / 20000
S_diff = x_diff_1.T @ x_diff_1 / 20000  + x_diff_2.T @ x_diff_2 / 20000
S = S_diff + S_iden
eva, evc = decomposition(S)
eva_inv = np.sqrt(1/eva)
evc = evc @ np.diag(eva_inv)
S1 = evc.T @ (S_diff) @ evc
eva1, evc1 = decomposition(S1)
print(evc @ evc1)



## ls
b = np.zeros((30000, 1))
b[20000:, 0] = 1
thetay, _, _, _ = lstsq(np.concatenate([y_iden_1, y_iden_2, y_diff_1], axis=0), b)
print(thetay)

##
S_ideny = y_iden_1.T @ y_iden_1 / 20000 + y_iden_2.T @ y_iden_2 / 20000
S_diffy = y_diff_1.T @ y_diff_1 / 20000  + y_diff_2.T @ y_diff_2 / 20000
Sy = S_diffy + S_ideny
evay, evcy = decomposition(Sy)
eva_invy = np.sqrt(1/evay)
evcy = evcy @ np.diag(eva_invy)
S1y = evcy.T @ (S_diffy) @ evcy
eva1y, evc1y = decomposition(S1y)
print(evcy @ evc1y)

