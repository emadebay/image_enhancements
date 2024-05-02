


import cv2

import numpy as np
from skimage import exposure

def apply_log_transformation_to_each_channel(channel, scale_c):
    
    # Apply logarithmic transformation to the channel
    transformed_channel = scale_c * np.log2(1 + np.abs(channel))
    
    return transformed_channel

def log_transform_skimage_with_package(image_path, c):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply log transformation using scikit-image
    transformed_image = exposure.adjust_log(image, c)
    
    # Convert back to uint8
    transformed_image = (transformed_image * 255).astype(np.uint8)
    
    return transformed_image


def log_transformation(image_path, scale_c):
    image = cv2.imread(image_path)

    # Get image shape
    height, width, depth = image.shape

  
    log_transformed_image = image.copy()  # Create a copy of the original image

    # Iterate over each color channel
    for c in range(depth):
            # Extract the current color channel
        channel = image[:, :, c]
            
            # Perform histogram equalization on the channel
        log_transformed_image_channel = apply_log_transformation_to_each_channel(channel, scale_c)

            
            # Replace the channel in the equalized image
        log_transformed_image[:, :, c] = log_transformed_image_channel

    # Display the original and transformed image
    cv2.imshow('Original Image', image)
    cv2.imshow('log transformed Image', log_transformed_image)
    cv2.imwrite('log_transformation/ct_scanc15_eq.pnm', log_transformed_image)

    #with package
    transformation_with_package = log_transform_skimage_with_package(image_path, scale_c)
    cv2.imshow('log transformed Image with package', transformation_with_package)
    cv2.imwrite('log_transformation/ct_scanc15_pa.pnm', transformation_with_package)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = "ct_scan.pnm"
log_transformation(image_path, 15)

