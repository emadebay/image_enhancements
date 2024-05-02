import cv2
import numpy as np

import gaus_ave
import median_filter

def add_salt_and_pepper_noise(image_path, salt_prob=0.05, pepper_prob=0.05):
    # Read the image
    image = cv2.imread(image_path)
    
    # Get image shape
    height, width, _ = image.shape
    
    # Generate salt noise
    num_salt = np.ceil(salt_prob * image.size)
    salt_coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    image[tuple(salt_coords)] = 255
    
    # Generate pepper noise
    num_pepper = np.ceil(pepper_prob * image.size)
    pepper_coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    image[tuple(pepper_coords)] = 0
    
    # Display the salt and pepper noisy image
    cv2.imshow('Noisy Image', image)
    cv2.imwrite('experiment/building.jpg', image)

    #save the effects of the gaussian blur
    #use 3 by 3 kernel
    #used 15 sigma
    noise_filter_with_gausian = gaus_ave.gaussian_averaging_to_be_used(image_path, 3, 15)
    cv2.imshow('gaussian filtered image', noise_filter_with_gausian)
    cv2.imwrite('experiment/building_gaus.jpg', noise_filter_with_gausian)

    #save the median blur
    noise_filter_with_median_blur = median_filter.median_filtering_to_be_used(image_path, 3)
    cv2.imshow('median filtered image', noise_filter_with_gausian)
    cv2.imwrite('experiment/building_median.jpg', noise_filter_with_gausian)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'building.jpg'  # Path to the input image
salt_prob = 0.05  # Probability of salt noise
pepper_prob = 0.05  # Probability of pepper noise
add_salt_and_pepper_noise(image_path, salt_prob, pepper_prob)
