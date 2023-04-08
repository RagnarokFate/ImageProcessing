import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy

# size of the image
m,n = 921, 750

# frame points of the blank wormhole image
src_points = np.float32([[0, 0],
                            [int(n / 3), 0],
                            [int(2 * n /3), 0],
                            [n, 0],
                            [n, m],
                            [int(2 * n / 3), m],
                            [int(n / 3), m],
                            [0, m]])

# blank wormhole frame points
dst_points = np.float32([[96, 282],
                       [220, 276],
                       [344, 276],
                       [468, 282],
                       [474, 710],
                       [350, 744],
                       [227, 742],
                       [103, 714]]
                      )


def find_transform(pointset1, pointset2):
    arr = pointset1.shape[0]
    x = np.zeros((arr * 2, 8))
    x2 = np.zeros(arr * 2)
    for i in range(0, arr):
        x[i * 2][0] = pointset1[i][0]
        x[i * 2][1] = pointset1[i][1]
        x[i * 2 + 1][2] = pointset1[i][0]
        x[i * 2 + 1][3] = pointset1[i][1]
        x[i * 2][4] = 1
        x[i * 2 + 1][5] = 1
        x[2 * i][6] = -(pointset1[i][0] * pointset2[i][0])
        x[2 * i][7] = -(pointset1[i][1] * pointset2[i][0])
        x[2 * i + 1][6] = -(pointset1[i][0] * pointset2[i][1])
        x[2 * i + 1][7] = -(pointset1[i][1] * pointset2[i][1])
        x2[i * 2] = pointset2[i][0]
        x2[i * 2 + 1] = pointset2[i][1]
    # calculate T - be careful of order when reshaping it

    T = np.matmul(np.linalg.pinv(x), x2)
    temp = T.copy()
    T = np.hstack((T, [1]))
    T = np.reshape(T, (3, 3))


    T[0][0] = temp[3]
    T[0][1] = temp[2]
    T[0][2] = temp[5]
    T[1][0] = temp[1]
    T[1][1] = temp[0]
    T[1][2] = temp[4]
    T[2][0] = temp[7]
    T[2][1] = temp[6]
    T[2][2] = 1

    return T


def trasnform_image(image, T):
    t2=np.linalg.inv(T)
    new_image = numpy.zeros((len(image), len(image[0])))
    for x in range(len(new_image)):
        for y in range(len(new_image[0])):
            temp=[x,y,1]
            vec = np.matmul(t2, temp)
            if 0<= round(vec[0]/vec[2])< len(new_image) and 0<= round(vec[1]/vec[2])<len(new_image[0]):
                new_image[x][y] = image[round(vec[0] / vec[2])][round(vec[1] / vec[2])]

    return new_image


def create_wormhole(im, T, iter=5):
    new_image=im.copy()
    for k in range(iter):
        im=trasnform_image(im,T)
        for i in range(len(im)):
            for j in range(len(im[i])):
                if(new_image[i][j]+im[i][j]>255):
                    new_image[i][j] = 255
                elif (new_image[i][j]+im[i][j]<0):
                    new_image[i][j] = 0
                else:
                    new_image[i][j] = new_image[i][j]+im[i][j]
    return new_image