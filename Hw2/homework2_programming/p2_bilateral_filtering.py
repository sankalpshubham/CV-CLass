"""
Sankalp Shubham: sxs190290
CS 4391 Homework 2 Programming: Part 3 - bilateral filter
Implement the bilateral_filtering() function in this python script
"""
 
import cv2
import numpy as np
import math

def bilateral_filtering(img: np.uint8, spatial_variance: float, intensity_variance: float, kernel_size: int,) -> np.uint8:
    """
    Homework 2 Part 3
    Compute the bilaterally filtered image given an input image, kernel size, spatial variance, and intensity range variance
    """

    img = img / 255
    img = img.astype("float32")
    img_filtered = np.zeros(img.shape) # Placeholder of the filtered image

    # Todo: For each pixel position [i, j], you need to compute the filtered output: img_filtered[i, j]
    # step 1: compute kernel_sizexkernel_size spatial and intensity range weights of the bilateral filter in terms of spatial_variance and intensity_variance. 
    # step 2: compute the filtered pixel img_filtered[i, j] using the obtained kernel weights and the neighboring pixels of img[i, j] in the kernel_sizexkernel_size local window
    # The bilateral filtering formula can be found in slide 15 of lecture 6
    # Tip: use zero-padding to address the black border issue.

    # ********************************
    sizeX, sizeY = img.shape
    # get a padding size based on the kernel size to handle borders
    padding_size = kernel_size // 2
    # create padded image
    img_padded = np.pad(img, ((padding_size, padding_size), (padding_size, padding_size)), mode='constant')

    # intialize and compute spatial weights
    spatial_weights = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            k, l = i - padding_size, j - padding_size
            spatial_weights[i, j] = math.exp(-(k ** 2 + l ** 2) / (2 * spatial_variance))

    # compute intensity range weights
    for i in range(padding_size, sizeX + padding_size):
        for j in range(padding_size, sizeY + padding_size):
            # get current and neighbor pixel
            current_pixel = img_padded[i, j]
            neighbor_pixel = img_padded[i - padding_size:i + padding_size + 1, j - padding_size:j + padding_size + 1]

            # compute intensity difference and weight between current and neighbor pixel
            intensity_difference = np.abs(current_pixel - neighbor_pixel)
            intensity_weight = np.exp(-(intensity_difference ** 2) / (2 * intensity_variance))

            # compute bilateral filter weight with spatial and intensity weights
            bilateral_weight = spatial_weights * intensity_weight
            weighted_sum = np.sum(bilateral_weight * neighbor_pixel)
            sum_of_weights = np.sum(bilateral_weight)

            # Compute the filtered pixel value
            img_filtered[i - padding_size, j - padding_size] = weighted_sum / sum_of_weights

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
    spatial_variance = 30 # signma_s^2
    intensity_variance = 0.5 # sigma_r^2
    kernel_size = 7
    img_bi = bilateral_filtering(img_noise, spatial_variance, intensity_variance, kernel_size)
    cv2.imwrite('results/im_bilateral.png', img_bi)