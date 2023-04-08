import cv2
import numpy as np
from scipy.signal import convolve2d as conv
from scipy.ndimage.filters import gaussian_filter as gaussian
import matplotlib.pyplot as plt


def sobel(im):
    # Define Sobel kernels
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # Convolve image with kernels
    x = conv(im, kernel_x)
    y = conv(im, kernel_y)
    # Calculate edge magnitude
    new_im = np.sqrt(np.square(x) + np.square(y))
    # Create binary edge image
    new_im[new_im <=150] = 0
    new_im[new_im >150] = 255
    # Return binary edge image
    return new_im


def canny(im):
    new_im = gaussian(im, 1.8)
    new_im = cv2.Canny(new_im, 40, 200)
    return new_im


def hough_circles(im):
    new_im = cv2.medianBlur(im, 3)
    c= cv2.HoughCircles(new_im, cv2.HOUGH_GRADIENT, 1,10,param1=270, param2=80, minRadius=2, maxRadius=0)
    c=c[0]
    for(x,y,r) in c:
        cv2.circle(new_im, (round(x),round(y)),round(r),(0,0,255),4)
        cv2.circle(new_im, (round(x), round(y)), 2, (0, 0, 0), 3)

    return  new_im


def hough_lines(im):
    new_im=im.copy()
    l = cv2.HoughLines(cv2.Canny(new_im, 240, 255, apertureSize=3), 1, np.pi / 60, 145)
    for line in l:
        r, t = line[0]
        a = np.cos(t)
        b = np.sin(t)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(new_im, (x1, y1), (x2, y2), (0, 0, 255), 3)
    return new_im