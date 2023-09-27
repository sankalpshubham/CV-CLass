"""
CS 4391 Homework 3 Programming
Implement the harris_corner() function and the non_maximum_suppression() function in this python script
Harris corner detector
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


#TODO: implement this function
# input: R is a Harris corner score matrix with shape [height, width]
# output: mask with shape [height, width] with valuse 0 and 1, where 1s indicate corners of the input image 
# idea: for each pixel, check its 8 neighborhoods in the image. If the pixel is the maximum compared to these
# 8 neighborhoods, mark it as a corner with value 1. Otherwise, mark it as non-corner with value 0
def non_maximum_suppression(R):
    xSize, ySize = R.shape
    mask = np.zeros(R.shape)
    R_padded = np.pad(R, ((3, 3), (3, 3)), mode='constant')

    for i in range(3, xSize + 3):
        for j in range(3, ySize + 3):
            neighborhood = R_padded[i - 3:i + 4, j - 3:j + 4]
            if R[i - 3, j - 3] >= np.max(neighborhood) and np.max(neighborhood) != 0:
                mask[i - 3, j - 3] = 1

    return mask


#TODO: implement this function
# input: im is an RGB image with shape [height, width, 3]
# output: corner_mask with shape [height, width] with valuse 0 and 1, where 1s indicate corners of the input image
# Follow the steps in Lecture 7 slides 26-27
# You can use opencv functions and numpy functions
def harris_corner(im):
    # step 0: convert RGB to gray-scale image
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # step 1: compute image gradient using Sobel filters
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html
    #laplacian = cv2.Laplacian(img_gray,cv2.CV_64F)
    sobelx = cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img_gray,cv2.CV_64F,0,1,ksize=5)

    # step 2: compute products of derivatives at every pixels
    Ixx = sobelx ** 2
    Iyy = sobely ** 2
    Ixy = sobelx * sobely

    # step 3: compute the sums of products of derivatives at each pixel using Gaussian filter (window size: 5x5, sigma = 1.5) from OpenCV
    Ixx_smoothed = cv2.GaussianBlur(Ixx, (5, 5), 1.5)
    Iyy_smoothed = cv2.GaussianBlur(Iyy, (5, 5), 1.5)
    Ixy_smoothed = cv2.GaussianBlur(Ixy, (5, 5), 1.5)

    # step 4: compute determinant and trace of the M matrix
    det = (Ixx_smoothed * Iyy_smoothed) - (Ixy_smoothed ** 2)
    trace = Ixx_smoothed + Iyy_smoothed
    
    # step 5: compute R scores with k = 0.05
    k = 0.05
    R = det - (k * (trace ** 2))
    
    # step 6: thresholding
    # up to now, you shall get a R score matrix with shape [height, width]
    threshold = 0.01 * R.max()
    R[R < threshold] = 0
    
    # step 7: non-maximum suppression
    #TODO implement the non_maximum_suppression function above
    corner_mask = non_maximum_suppression(R)

    return corner_mask


# main function
if __name__ == '__main__':

    # read the image in data
    # rgb image
    rgb_filename = 'data/000006-color.jpg'
    im = cv2.imread(rgb_filename)
    
    # your implementation of the harris corner detector
    corner_mask = harris_corner(im)
    
    # opencv harris corner
    img = im.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    opencv_mask = dst > 0.01 * dst.max()
        
    # visualization for your debugging
    fig = plt.figure()
        
    # show RGB image
    ax = fig.add_subplot(1, 3, 1)
    plt.imshow(im[:, :, (2, 1, 0)])
    ax.set_title('RGB image')
        
    # show our corner image
    ax = fig.add_subplot(1, 3, 2)
    plt.imshow(im[:, :, (2, 1, 0)])
    index = np.where(corner_mask > 0)
    plt.scatter(x=index[1], y=index[0], c='y', s=5)
    ax.set_title('our corner image')
    
    # show opencv corner image
    ax = fig.add_subplot(1, 3, 3)
    plt.imshow(im[:, :, (2, 1, 0)])
    index = np.where(opencv_mask > 0)
    plt.scatter(x=index[1], y=index[0], c='y', s=5)
    ax.set_title('opencv corner image')

    plt.show()
