"""
Sankalp Shubham: sxs190290
CS 4391 Homework 2 Programming: Part 4 - non-local means filter
Implement the nlm_filtering() function in this python script
"""

import cv2
import numpy as np
import math

def nlm_filtering(img: np.uint8, intensity_variance: float, patch_size: int, window_size: int,) -> np.uint8:
    """
    Homework 2 Part 4
    Compute the filtered image given an input image, kernel size of image patch, spatial variance, and intensity range variance
    """

    img = img / 255
    img = img.astype("float32")
    img_filtered = np.zeros(img.shape) # Placeholder of the filtered image
    
    # Todo: For each pixel position [i, j], you need to compute the filtered output: img_filtered[i, j] using a non-local means filter
    # step 1: compute window_sizexwindow_size filter weights of the non-local means filter in terms of intensity_variance. 
    # step 2: compute the filtered pixel img_filtered[i, j] using the obtained kernel weights and the pixel values in the search window
    # Please see slides 30 and 31 of lecture 6. Clarification: the patch_size refers to the size of small image patches (image content in yellow, 
    # red, and blue boxes in the slide 30); intensity_variance denotes sigma^2 in slide 30; the window_size is the size of the search window as illustrated in slide 31.
    # Tip: use zero-padding to address the black border issue. 

    # ********************************
    sizeX, sizeY = img.shape
    # set padding size and center
    padding_size = window_size // 2
    center = patch_size // 2

    # create padded image
    img_padded = np.pad(img, ((padding_size, padding_size), (padding_size, padding_size)), mode='constant')

    # run the nlm algorithm
    for i in range(padding_size, sizeX + padding_size + 1):
        for j in range(padding_size, sizeY + padding_size + 1):
            # get the image patch
            img_patch = img_padded[i - center:i + center + 1, j - center: j + center + 1]
            # initialize the sum of weights and weighted sum
            sum_of_weights, weighted_sum = 0.0, 0.0

            for x in range(i - padding_size, i + padding_size + 1):
                for y in range(j - padding_size, j + padding_size + 1):
                    # check if x and y are within range of padded image
                    if x - center >= 0 and x + center + 1 <= img_padded.shape[0] and y - center >= 0 and y + center + 1 <= img_padded.shape[1]:
                        # get the neighbor patch
                        neighbor_patch = img_padded[x - center: x + center + 1, y - center: y + center + 1]

                        # compute weight and SSD between img patch and neighbor patch
                        ssd = np.sum((img_patch - neighbor_patch) ** 2)
                        weight = np.exp(-ssd / (2 * (intensity_variance)))

                        # update sum of weights and weighted_sum
                        weighted_sum += weight * img_padded[x, y]
                        sum_of_weights += weight

            
            # Normalize the pixel value if i and j in bounds
            if i - padding_size < img_filtered.shape[0] and j - padding_size < img_filtered.shape[1]:
                img_filtered[i - padding_size, j - padding_size] = weighted_sum / sum_of_weights
    # ********************************

            
    img_filtered = img_filtered * 255
    img_filtered = np.uint8(img_filtered)
    return img_filtered

 
if __name__ == "__main__":
    img = cv2.imread("data/img/butterfly.jpeg", 0) # read gray image
    img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA) # reduce image size for saving your computation time
    cv2.imwrite('results/im_original.png', img) # save image 
    
    # Generate Gaussian noise
    noise = np.random.normal(0,0.6,img.size)
    noise = noise.reshape(img.shape[0],img.shape[1]).astype('uint8')
   
    # Add the generated Gaussian noise to the image
    img_noise = cv2.add(img, noise)
    cv2.imwrite('results/im_noisy.png', img_noise)
    
    # Bilateral filtering
    intensity_variance = 1
    patch_size = 5 # small image patch size
    window_size = 15 # serach window size
    img_bi = nlm_filtering(img_noise, intensity_variance, patch_size, window_size)
    cv2.imwrite('results/im_nlm.png', img_bi)
    