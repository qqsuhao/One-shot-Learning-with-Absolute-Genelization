# -*- coding:utf8 -*-
# @TIME     : 2020/11/6 15:18
# @Author   : Hao Su
# @File     : self_transforms.py

'''
自定义的transform
'''


import numpy as np
from PIL import Image
import cv2

class AddSaltPepperNoise(object):
    def __init__(self, density=0):
        self.density = density

    def __call__(self, img):
        img = np.array(img)
        h, w = img.shape
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(h, w), p=[Nd/2.0, Nd/2.0, Sd])      # 生成一个通道的mask
        # mask = np.repeat(mask, c, axis=2)
        img[mask == 0] = 0                                                              # 椒
        img[mask == 1] = 255
        return img


class AddGaussianNoise(object):
    def __init__(self, variance=1.2, amplitude=1):
        self.mean = 0.0
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        h, w = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w))
        N = N.astype("uint8")
        img = N + img
        img[img > 255] = 255                       # 避免有值超过255而反转
        img[img < 0] = 0                      #
        # img = img.astype('float32')
        return img


class Fllip(object):
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        img = 255 - img
        return img


class Style(object):
    def __init__(self, option=1, scale=0.5):
        if option == 1:
            path = "../data/style1.png"
        else:
            path = "../data/style2.png"
        style_img = cv2.imread(path)
        style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2GRAY)
        h, w = style_img.shape
        self.style_img = cv2.resize(style_img, (int(h*scale), int(w*scale)))

    def __call__(self, img):
        img = np.array(img)
        _, mask = cv2.threshold(img, 2, 255, cv2.THRESH_BINARY)
        h, w = img.shape
        x = np.random.randint(0, self.style_img.shape[0]-h)
        y = np.random.randint(0, self.style_img.shape[1]-w)
        # x = 0
        # y = 0
        style_patch = self.style_img[x:x+h, y:y+w]
        part1 = cv2.addWeighted(img, 1, style_patch, 0, 0)
        part1 = np.bitwise_and(mask, part1)
        img = 255 - img
        mask = 255 - mask
        style_patch = cv2.flip(style_patch, 1)
        part2 = cv2.addWeighted(img, 0.5, style_patch, 0.5, 0)
        part2 = np.bitwise_and(mask, part2)

        return 255 - part1 + part2


