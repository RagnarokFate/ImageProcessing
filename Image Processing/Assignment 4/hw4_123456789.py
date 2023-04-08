import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import cv2


def clean_baby(im):
    return clean_im


def clean_windmill(im):

    return clean_im


def clean_watermelon(im):
    return clean_im


def clean_umbrella(im):
    return clean_im


def clean_USAflag(im):
    return clean_im


def clean_cups(im):
    return clean_im


def clean_house(im):
    return clean_im


def clean_bears(im):
    return 



'''
    # an example of how to use fourier transform:
    img = cv2.imread(r'Images\windmill.tif')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_fourier = np.fft.fft2(img) # fft - remember this is a complex numbers matrix 
    img_fourier = np.fft.fftshift(img_fourier) # shift so that the DC is in the middle
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.title('original image')
    
    plt.subplot(1,3,2)
    plt.imshow(np.log(abs(img_fourier)), cmap='gray') # need to use abs because it is complex, the log is just so that we can see the difference in values with out eyes.
    plt.title('fourier transform of image')

    img_inv = np.fft.ifft2(img_fourier)
    plt.subplot(1,3,3)
    plt.imshow(abs(img_inv), cmap='gray')
    plt.title('inverse fourier of the fourier transform')

'''