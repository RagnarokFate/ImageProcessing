import numpy as np
import matplotlib.pyplot as plt
import cv2

def histImage(im):
    h = [0] * 256
    for temp in range(len(im)):
        for i in range(len(im[temp])):
            j = im[temp][i]
            h[j] += 1
    return h

def nhistImage(im):
    nh = histImage(im)
    wight = len(im)
    row = len(im[0])
    temp = wight * row
    for i in range(256):
            nh[i] /= temp
    return nh

def ahistImage(im):
    ah  = histImage(im)
    for i in range(1, 256):
        ah[i] = ah[i] + ah[i - 1]

    return ah

def calcHistStat(h):
    m = sum(i * h[i] for i in range(256)) / sum(x for x in h)
    e = sum(i * i * h[i] for i in range(256)) / sum(x for x in h) - m * m
    return m, e


def mapImage(im,tm):
    nim = [[0 if tm[i] < 0 else 255 if tm[i] > 255 else tm[i] for i in im[j]] for j in range(len(im[0]))]
    return nim

def histEqualization(im):
    array1 = ahistImage(im)
    length = sum(histImage(im)) / 256
    array2 = [length for _ in range(256)]
    array2 = np.cumsum(array2)
    tm = [0 for _ in range(256)]
    i = 0
    j = 0
    while i < 256 and j < 256:
        if array1[i] <= array2[j]:
            tm[i] = j
            i += 1
        elif array1[i] > array2[j]:
            j += 1

    return tm