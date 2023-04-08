import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter






def clean_baby(im):
    clean_im = im.copy()
    clean_im = median_filter(clean_im, size=3)
    dst = np.float32([[0, 0],
                      [255, 0],
                      [0, 255],
                      [255, 255]])

    src = np.float32([[6, 20],
                      [111, 20],
                      [6, 130],
                      [111, 130]])
    clean_im = helper(cv2.warpPerspective(clean_im, cv2.getPerspectiveTransform(src, dst), (256, 256), flags=cv2.INTER_CUBIC), [0, 255])
    return clean_im


def clean_windmill(im):
    fourier = np.fft.fftshift(np.fft.fft2(im))
    fourier[124][100] = -1
    fourier[132][156] = -1
    clean_im = helper(abs(np.fft.ifft2(fourier)), [0, 256])
    return clean_im


def clean_watermelon(im):
    clean_im = im.copy()
    kernel = np.array([[0, -4, 0], [-4, 17, -4], [0, -4, 0]])
    clean_im = cv2.filter2D(src=clean_im, ddepth=-1, kernel=kernel)
    return clean_im


def clean_umbrella(im):
    fourier = np.fft.fft2(im)
    mask = np.zeros([256, 256])
    mask[0][0] = 0.5
    mask[4][79] = 0.5
    mask = np.fft.fft2(mask)
    mask[abs(mask) < 0.01] = 1
    fourier=fourier/mask
    clean_im = abs(np.fft.ifft2(fourier))
    return clean_im



def clean_USAflag(im):
    clean_im = im.copy()
    clean_im = median_filter(clean_im, [1, 50])
    clean_im[0:90, 0:150] = im[0:90, 0:150]
    return clean_im



def clean_cups(im):
    fourier = np.fft.fftshift(np.fft.fft2(im))
    fourier[0:108, 0:256] /= 2
    fourier[147:256, 0:256] /= 2
    fourier[0:256, 0:108] /= 2
    fourier[0:256, 147:256] /= 2
    return helper(abs(np.fft.ifft2(fourier)), [0, 256])



def clean_house(im):
    img_fourier = np.fft.fft2(im)

    mask = np.zeros([191, 191])
    mask[0][0:10] = 0.1
    mask = np.fft.fft2(mask)
    mask[abs(mask) < 0.01] = 1
    img_fourier = img_fourier / mask
    clean_im = abs(np.fft.ifft2(img_fourier))
    return clean_im

def clean_bears(im):
    tm = np.zeros([1, 256])
    tm[0] = np.array(range(256))
    tm = tm / 256
    tm = np.power(tm, 1 / 2.2)
    tm = tm * 256
    Slices = np.zeros([256, len(im)*len(im[0])])
    for grayscale in range(256):
        Slices[grayscale] = np.ravel(im == grayscale)
    slices = Slices.transpose().transpose()
    TMim = np.zeros([1, len(im) * len(im[0])])
    for gs in range(256):
        TMim[0] += tm[0][gs] * slices[gs]
    clean_im = TMim.reshape((len(im), len(im[0])))
    clean_im = helper(clean_im, [0, 255])
    return clean_im


def helper(im, range):
    newIm = im.copy()
    min = np.min(np.ravel(newIm))
    max = np.max(np.ravel(newIm))
    b = range[1]
    a = 1
    if max-min != 0:
        b = np.float((np.float(max)*np.float(range[0]) - np.float(min)*np.float(range[1])))/np.float((max-min))
        a = np.float((range[1]-range[0]))/(max-min)

    newIm = (newIm * a)+b
    return newIm




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
