"""
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
    # ______________________________________________________________________________________________________________________________

    sizeX, sizeY = img.shape

    # Define a padding size based on the kernel size to handle borders
    padding_size = kernel_size // 2

    # Precompute spatial weights (depends only on spatial_variance)
    spatial_weights = np.zeros((kernel_size, kernel_size))
    for k in range(-padding_size, padding_size + 1):
        for l in range(-padding_size, padding_size + 1):
            spatial_weights[k + padding_size, l + padding_size] = math.exp(-(k ** 2 + l ** 2) / (2 * spatial_variance))

    # Create a padded version of the input image
    img_padded = np.pad(img, ((padding_size, padding_size), (padding_size, padding_size)), mode='constant')

    for i in range(padding_size, sizeX + padding_size):
        for j in range(padding_size, sizeY + padding_size):
            # Current pixel coordinates in the padded image
            current_pixel = img_padded[i, j]

            # Initialize the weighted sum and the sum of weights
            weighted_sum = 0.0
            sum_of_weights = 0.0

            # Compute intensity weight (depends on intensity_variance)
            for x in range(i - padding_size, i + padding_size + 1):
                for y in range(j - padding_size, j + padding_size + 1):
                    # Neighboring pixel coordinates
                    neighbor_pixel = img_padded[x, y]

                    # Compute intensity difference
                    intensity_difference = np.abs(current_pixel - neighbor_pixel)

                    # Compute intensity weight using your original formula
                    intensity_weight = (1 / math.sqrt(2 * math.pi * intensity_variance)) * math.exp(-(intensity_difference ** 2) / (2 * intensity_variance))

                    # Bilateral filter weight (combined with spatial weight)
                    bilateral_weight = spatial_weights[x - i + padding_size, y - j + padding_size] * intensity_weight

                    # Update the weighted sum and sum of weights
                    weighted_sum += neighbor_pixel * bilateral_weight
                    sum_of_weights += bilateral_weight

            # Compute the filtered pixel value
            img_filtered[i - padding_size, j - padding_size] = weighted_sum / sum_of_weights

    # ______________________________________________________________________________________________________________________________


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