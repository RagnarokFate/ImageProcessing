import cv2
import numpy as np
from matplotlib import pyplot as plt


def writeMorphingVideo(image_list, video_name):
    out = cv2.VideoWriter(video_name + ".mp4", cv2.VideoWriter_fourcc(*'MP4V'), 20.0, image_list[0].shape, 0)
    for im in image_list:
        out.write(im)
    out.release()


def findProjectiveTransform(pointsSet1, pointsSet2):
    N = pointsSet1.shape[0]
    x = np.zeros((N * 2, 8))
    xx = np.zeros(N * 2)
    # iterate iver points to create x , x'
    for i in range(0, N):
        x[i * 2][0] = pointsSet1[i][0]
        x[i * 2][1] = pointsSet1[i][1]
        x[i * 2 + 1][2] = pointsSet1[i][0]
        x[i * 2 + 1][3] = pointsSet1[i][1]
        x[i * 2][4] = 1
        x[i * 2 + 1][5] = 1
        x[2 * i][6] = -(pointsSet1[i][0] * pointsSet2[i][0])
        x[2 * i][7] = -(pointsSet1[i][1] * pointsSet2[i][0])
        x[2 * i + 1][6] = -(pointsSet1[i][0] * pointsSet2[i][1])
        x[2 * i + 1][7] = -(pointsSet1[i][1] * pointsSet2[i][1])
        xx[i * 2] = pointsSet2[i][0]
        xx[i * 2 + 1] = pointsSet2[i][1]
    # calculate T - be careful of order when reshaping it

    T = np.matmul(np.linalg.pinv(x), xx)
    T = np.hstack((T, [1]))
    T = np.reshape(T, (3, 3))
    T[0][2], T[1][0] = T[1][0], T[0][2]
    T[0][2], T[1][1] = T[1][1], T[0][2]
    return T


def getImagePts(im1, im2, varName1, varName2, nPoints):       # checkkkkkkkkkkkkkkkkkkkkk
    plt.imshow(im1)#,cmap='gray', vmin=0, vmax=255)
    x = plt.ginput(nPoints)
    imagePts1 = np.round(x)
    pts1 = np.ones((nPoints, 1))
    imagePts1 = np.hstack((imagePts1, pts1))

    plt.imshow(im2)#,cmap='gray', vmin=0, vmax=255)
    x = plt.ginput(nPoints)
    imagePts2 = np.round(x)
    pts2 = np.ones((nPoints, 1))
    imagePts2 = np.hstack((imagePts2, pts2))

    imagePts1[:, [0, 1]] = imagePts1[:, [1, 0]]

    imagePts2[:, [0, 1]] = imagePts2[:, [1, 0]]

    np.save(varName1 + ".npy", imagePts1)
    np.save(varName2 + ".npy", imagePts2)


def findAffineTransform(pointsSet1, pointsSet2):
    N = pointsSet1.shape[0]
    x = np.zeros((N * 2, 6))
    xx = np.zeros(N * 2)
    # iterate iver points to create x , x'

    for i in range(0, N):
        x[i * 2][0] = pointsSet1[i][0]
        x[i * 2][1] = pointsSet1[i][1]
        x[i * 2 + 1][2] = pointsSet1[i][0]
        x[i * 2 + 1][3] = pointsSet1[i][1]
        x[i * 2][4] = 1
        x[i * 2 + 1][5] = 1
        xx[i * 2] = pointsSet2[i][0]
        xx[i * 2 + 1] = pointsSet2[i][1]
    y = np.linalg.pinv(x)
    T = np.matmul(y, xx)
    T = np.reshape(T, (3, 2))
    T = np.transpose(T)
    T[0][1], T[1][0] = T[1][0], T[0][1]
    T = np.vstack([T, [0, 0, 1]])
    return T


def mapImage(im, T, sizeOutIm):
    im_new = np.zeros(sizeOutIm)
    a1 = np.arange(0, sizeOutIm[0])
    a2 = np.arange(0, sizeOutIm[1])
    # create meshgrid of all coordinates in new image [x,y]
    xx, yy = np.meshgrid(a1, a2)
    xx = xx.ravel()
    yy = yy.ravel()
    xy = np.vstack([xx, yy])
    xy = np.transpose(xy)
    # add homogenous coord [x,y,1]
    pts1 = np.ones((sizeOutIm[0] * sizeOutIm[1], 1))
    xy = np.hstack((xy, pts1))

    # calculate source coordinates that correspond to [x,y,1] in new image
    T = np.linalg.pinv(T)
    mul_new = np.matmul(T, np.transpose(xy))
    mul_new = np.transpose(mul_new)
    mul_new[:, 0] = mul_new[:, 0] / mul_new[:, 2]
    mul_new[:, 1] = mul_new[:, 1] / mul_new[:, 2]

    row, col = im.shape
    # find coordinates outside range and delete (in source and target)
    #xy = xy[~np.any((mul_new < 0) | (mul_new > (col - 1)), axis=1), :]
    #mul_new = mul_new[~np.any((mul_new < 0) | (mul_new > (col - 1)), axis=1), :]

    #xy = np.delete(xy, np.where(((mul_new[:, 0] > row - 1) | (mul_new[:, 0] < 0)))[0], axis=0)
    #xy = np.delete(xy, np.where(((mul_new[:, 1] > col - 1) | (mul_new[:, 1] < 0)))[0], axis=0)
    #mul_new = np.delete(mul_new, np.where(((mul_new[:, 0] > row-1) | (mul_new[:, 0] < 0)))[0], axis=0)
    #mul_new = np.delete(mul_new, np.where(((mul_new[:, 1] > col-1) | (mul_new[:, 1] < 0)))[0], axis=0)
    merge_newxy=np.hstack((mul_new,xy))
    merge_newxy = np.delete(merge_newxy, np.where(((merge_newxy[:, 0] > row-1) | (merge_newxy[:, 0] < 0)))[0], axis=0)
    merge_newxy = np.delete(merge_newxy, np.where(((merge_newxy[:, 1] > col-1) | (merge_newxy[:, 1] < 0)))[0], axis=0)

    # interpolate - bilinear

    X = merge_newxy[:, 0]
    Y = merge_newxy[:, 1]

    x_left = np.floor(X)
    x_right = np.ceil(X)
    y_top = np.floor(Y)
    y_bottom = np.ceil(Y)

    upper_left = np.vstack((x_left, y_top)).astype(int)
    upper_right = np.vstack((x_right, y_top)).astype(int)
    bottom_left = np.vstack((x_left, y_bottom)).astype(int)
    bottom_right = np.vstack((x_right, y_bottom)).astype(int)

    deltaX = X - x_left
    deltaY = Y - y_top
    arr_ones = np.ones(len(deltaX))

    S = im[bottom_right[0, :], bottom_right[1, :]] * deltaX + im[bottom_left[0, :], bottom_left[1, :]] * (
            arr_ones - deltaX)

    N = im[upper_right[0, :], upper_right[1, :]] * deltaX + im[upper_left[0, :], upper_left[1, :]] * (arr_ones - deltaX)

    V = N * deltaY + S * (arr_ones - deltaY)
    merge_newxy = merge_newxy.astype(int)
    im_new[merge_newxy[:, 3], merge_newxy[:, 4]] = V

    return (im_new)


def createMorphSequence(im1, im1_pts, im2, im2_pts, t_list, transformType):
    if transformType == 1:
        T12 = findProjectiveTransform(im1_pts, im2_pts)
        T21 = findProjectiveTransform(im2_pts, im1_pts)
    else:
        T12 = findAffineTransform(im1_pts, im2_pts)
        T21 = findAffineTransform(im2_pts, im1_pts)
    ims = []
    id = np.eye(3)
    for t in t_list:
        # TODO: calculate nim for each t
        T12_map = (1 - t) * id + t * T12
        T21_map = t * id + (1 - t) * T21
        m1 = mapImage(im1, T12_map, im2.shape)
        m2 = mapImage(im2, T21_map, im1.shape)
        nim = (1 - t) * m1 + t * m2
        nim=np.uint8(nim)
        ims.append(nim)
    return ims

