"""
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
    img_filtered = np.zeros(img.shape)  # Placeholder of the filtered image

    # Todo: For each pixel position [i, j], you need to compute the filtered output: img_filtered[i, j] using a non-local means filter
    # step 1: compute window_sizexwindow_size filter weights of the non-local means filter in terms of intensity_variance.
    # step 2: compute the filtered pixel img_filtered[i, j] using the obtained kernel weights and the pixel values in the search window
    # Please see slides 30 and 31 of lecture 6. Clarification: the patch_size refers to the size of small image patches (image content in yellow,
    # red, and blue boxes in the slide 30); intensity_variance denotes sigma^2 in slide 30; the window_size is the size of the search window as illustrated in slide 31.
    # Tip: use zero-padding to address the black border issue.

    # ********************************
    # sizeX, sizeY = img.shape

    # # # Define a padding size based on the patch size to handle borders
    # # padding_size = patch_size // 2

    # # Define a padding size based on the window size to handle borders
    # padding_size = window_size // 2 + 1

    # # Create a padded version of the input image
    # img_padded = np.pad(img, ((padding_size, padding_size), (padding_size, padding_size)), mode='constant')

    # for i in range(padding_size, sizeX + padding_size):
    #     for j in range(padding_size, sizeY + padding_size):
    #         # Current pixel coordinates in the padded image
    #         current_pixel = img_padded[i, j]

    #         # Initialize the weighted sum and the sum of weights
    #         weighted_sum = 0.0
    #         sum_of_weights = 0.0

    #         # Compute non-local means filter weights
    #         for x in range(i - window_size // 2, i + window_size // 2 + 1):
    #             for y in range(j - window_size // 2, j + window_size // 2 + 1):
    #                 # Neighboring pixel coordinates
    #                 neighbor_pixel = img_padded[x, y]

    #                 # Compute intensity difference and non-local means filter weight
    #                 intensity_difference = current_pixel - neighbor_pixel
    #                 intensity_weight = math.exp(-(intensity_difference ** 2) / (2 * intensity_variance))

    #                 # Update the weighted sum and sum of weights
    #                 weighted_sum += neighbor_pixel * intensity_weight
    #                 sum_of_weights += intensity_weight

    #         # Compute the filtered pixel value
    #         img_filtered[i - padding_size, j - padding_size] = weighted_sum / sum_of_weights

    # ********************************

    pad = patch_size // 2
    img_pad = np.pad(img, pad, mode='constant')

    for i in range(pad, img.shape[0] + pad):
        for j in range(pad, img.shape[1] + pad):

            # Extract full patch centered at (i,j)
            patch = img_pad[i-pad:i+pad+1, j-pad:j+pad+1]

            # Only compute SSD for patches fully inside the image
            if i - window_size//2 >= pad and i + window_size//2 + 1 < img.shape[0] + pad:
                if j - window_size//2 >= pad and j + window_size//2 + 1 < img.shape[1] + pad:

                    weights = []
                    for x in range(i-window_size//2, i+window_size//2+1):
                        for y in range(j-window_size//2, j+window_size//2+1):

                            # Extract full patch centered at (x,y)
                            patch_neighbor = img_pad[x-pad:x+pad+1, y-pad:y+pad+1]

                            # Compute SSD between full patches
                            ssd = np.sum((patch - patch_neighbor)**2)

                            # Compute weight
                            weight = np.exp(-ssd / (2 * intensity_variance))

                            weights.append(weight)

                    # Normalize weights and compute weighted average
                    weights = np.array(weights) / np.sum(weights)
                    neighbors = img_pad[i-window_size//2:i+window_size//2+1,
                                        j-window_size//2:j+window_size//2+1]
                    img_filtered[i-pad, j-pad] = np.sum(weights * neighbors)

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