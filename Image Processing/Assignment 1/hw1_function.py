import math

import cv2
import numpy as np
import matplotlib.pyplot as plt


def print_IDs():
    print("206380826+207662958\n")


def contrastEnhance(im, range):
    # TODO: implement fucntion
    row, col = im.shape
    nim = np.zeros((row, col))
    cnew = range[1] - range[0]
    cold = np.max(im) - np.min(im)
    a = cnew / cold
    b = range[1] - (a * np.max(im))
    i = 0
    while i < row:
        j = 0
        while j < col:
            nim[i][j] = a * (im[i][j]) + b
            j += 1
        i += 1
    return nim, a, b


def showMapping(old_range, a, b):
    imMin = np.min(old_range)
    imMax = np.max(old_range)
    x = np.arange(imMin, imMax + 1, dtype=np.float)
    y = a * x + b
    plt.figure()
    plt.plot(x, y)
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.title('contrast enhance mapping')


def minkowski2Dist(im1, im2):
    # TODO: implement fucntion
    his_1, bins1 = np.histogram(im1, bins=256, range=(0, 255))
    his_2, bins2 = np.histogram(im2, bins=256, range=(0, 255))
    his_1 = his_1.astype(float)
    his_2 = his_2.astype(float)
    sum1 = im1.shape[0] * im1.shape[1]
    his_1 = (his_1 / sum1)
    sum2 = im2.shape[0] * im2.shape[1]
    his_2 = (his_2 / sum2)
    i = 0
    d = 0
    while i < 256:
        d = float(pow(abs(his_1[i] - his_2[i]), 2)) + d
        i += 1
    d = float(math.sqrt(d))
    return d


def meanSqrDist(im1, im2):
    # TODO: implement fucntion - one line
    return np.sum(np.power((np.subtract(im1.astype(float), im2.astype(float))), 2)) / float(len(im1) * len(im1[0]))



def sliceMat(im):
    # TODO: implement fucntion
    # 2d array size of pixnum*256 fill it with zeros
    # loop i->>256
    row, col = im.shape
    mat = np.array([[0] * (row * col)] * 256)

    i = 0
    while i < 256:
        mat1 = im == i
        mat1 = mat1.flatten()
        mat[i, :] = mat1
        i += 1
    mat = np.transpose(mat)
    return mat


def SLTmap(im1, im2):
    # TODO: implement fucntion
    mat1 = sliceMat(im1)
    TM = np.zeros(256)

    for i in range(256):
        tmp = mat1[:, i]
        tmp_mat = tmp.reshape(im1.shape[0], im1.shape[1])
        if tmp.sum() != 0:
            TM[i] = float(((np.multiply(im2, tmp_mat)).sum()) / tmp.sum())
    return mapImage(im1, TM), TM


def mapImage(im, tm):
    # TODO: implement fucntion
    im1 = sliceMat(im)
    i = 0
    TMim = np.array([[0] * len(im)] * len(im[0]))

    while i < 256:
        tmp = im1[:, i] * tm[i]
        tmp = tmp.reshape(len(im[0]), len(im))
        TMim = np.add(TMim, tmp)
        i += 1
    return TMim


def sltNegative(im):
    # TODO: implement fucntion - one line
    tm = np.arange(256)
    tm = tm[::-1]
    tm = mapImage(im, tm)
    return tm


def sltThreshold(im, thresh):
    # TODO: implement fucntion

    tm = np.array([0] * thresh)
    tm1 = np.array([256] * (256 - thresh))
    tm = np.concatenate([tm, tm1])
    tm = mapImage(im, tm)
    return tm
