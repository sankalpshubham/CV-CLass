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

    # MINE ___________________________________________________________________________________________________________________________________
    # Your code is here.
    # sizeX, sizeY = img.shape
    # center = kernel_size // 2
    # padding_size = center       # Define a padding size based on the kernel size to handle borders
    # spacial_weights, intensity_range_weights = np.zeros((kernel_size, kernel_size)), np.zeros((kernel_size, kernel_size))

    # # Create a padded version of the input image
    # img_padded = np.pad(img, ((padding_size, padding_size), (padding_size, padding_size)), mode='constant')

    # # Calculate the center of the kernel
    
    # # Compute the spacial and intensity range weights
    # for i in range(kernel_size):
    #     for j in range(kernel_size):
    #         k, l = i - center, j - center
    #         spacial_weights[i, j] = (1 / (2 * math.pi * spatial_variance)) * math.exp(-(k ** 2 + l ** 2) / (2 * spatial_variance))
    #         intensity_range_weights[i, j] = (1 / math.sqrt(2 * math.pi * intensity_variance)) * math.exp(-(k ** 2) / (2 * intensity_variance))

    
    # # filtering for each pixel
    # for i in range(kernel_size // 2, sizeX - kernel_size // 2):
    #     for j in range(kernel_size // 2, sizeY - kernel_size // 2):
    #         # Compute the filtered pixel value using the kernel weights and neighboring pixels
    #         img_patch = img[i - kernel_size // 2 : i + kernel_size // 2 + 1, j - kernel_size // 2 : j + kernel_size // 2 + 1]
    #         img_filtered[i, j] = (1 / np.sum(spacial_weights * intensity_range_weights)) * np.sum(spacial_weights * intensity_range_weights * img_patch)

    #___________________________________________________________________________________________________________________________________

    # ********************************


    # GOOGLE ___________________________________________________________________________________________________________________________________
    # # Create a padded image to handle the black border issue
    # padded_img = np.pad(img, ((kernel_size - 1, kernel_size - 1), (kernel_size - 1, kernel_size - 1)), mode='constant', constant_values=0)

    # # Compute the spatial and intensity range weights of the bilateral filter
    # spatial_weights = np.exp(-((padded_img[:, :, None] - padded_img[None, :, :]) ** 2) / (2 * spatial_variance))
    # intensity_weights = np.exp(-((padded_img[:, :, None] - padded_img[None, :, :]) ** 2) / (2 * intensity_variance))

    # # Compute the filtered output for each pixel
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         kernel = spatial_weights[i : i + kernel_size, j : j + kernel_size] * intensity_weights[i : i + kernel_size, j : j + kernel_size]
    #         kernel = kernel / np.sum(kernel)
    #         img_filtered[i, j] = np.sum(kernel * padded_img[i : i + kernel_size, j : j + kernel_size])
    #___________________________________________________________________________________________________________________________________
    sizeX, sizeY = img.shape

    # Define a padding size based on the kernel size to handle borders
    padding_size = kernel_size // 2

    # Create a padded version of the input image
    img_padded = np.pad(img, ((padding_size, padding_size), (padding_size, padding_size)), mode='constant')

    for i in range(padding_size, sizeX + padding_size):
        for j in range(padding_size, sizeY + padding_size):
            # Current pixel coordinates in the padded image
            current_pixel = img_padded[i, j]

            # Initialize the weighted sum and the sum of weights
            weighted_sum = 0.0
            sum_of_weights = 0.0

            # Compute spatial and intensity weights
            for x in range(i - padding_size, i + padding_size + 1):
                for y in range(j - padding_size, j + padding_size + 1):
                    # Neighboring pixel coordinates
                    neighbor_pixel = img_padded[x, y]

                    # Compute spatial and intensity differences
                    spatial_difference = np.sqrt((i - x) ** 2 + (j - y) ** 2)
                    intensity_difference = np.abs(current_pixel - neighbor_pixel)

                    # Compute spatial and intensity weights using your original formulas
                    spatial_weight = (1 / (2 * math.pi * spatial_variance)) * math.exp(-(spatial_difference ** 2) / (2 * spatial_variance))
                    intensity_weight = (1 / math.sqrt(2 * math.pi * intensity_variance)) * math.exp(-(intensity_difference ** 2) / (2 * intensity_variance))

                    # Bilateral filter weight
                    bilateral_weight = spatial_weight * intensity_weight

                    # Update the weighted sum and sum of weights
                    weighted_sum += neighbor_pixel * bilateral_weight
                    sum_of_weights += bilateral_weight

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