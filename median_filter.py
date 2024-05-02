


import cv2

import numpy as np




def apply_median_filtering(channel, window_size):
  
   
    # Get dimensions of the channel
    rows, cols = channel.shape

    # Define padding size
    pad_size = window_size // 2

    # Pad the channel to handle border pixels
    padded_channel = np.pad(channel, pad_size, mode='constant')

    # Apply Gaussian averaging filter
    smoothed_channel = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            # Extract local region
            local_region = padded_channel[i:i+window_size, j:j+window_size]

            # Apply filter
            smoothed_channel[i, j] = np.median(local_region)

    return smoothed_channel

def median_filter_with_package(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Apply median filtering with a 3x3 kernel
    filtered_image = cv2.medianBlur(image, ksize=3)
    return filtered_image

def median_filtering_to_be_used(image_path, window_size):
    image = cv2.imread(image_path)

    # Get image shape
    height, width, depth = image.shape

  
    median_image = image.copy()  # Create a copy of the original image

    # Iterate over each color channel
    for c in range(depth):
            # Extract the current color channel
        channel = image[:, :, c]
            
            # Perform histogram equalization on the channel
        median_image_channel = apply_median_filtering(channel, window_size)

            
            # Replace the channel in the equalized image
        median_image[:, :, c] = median_image_channel

    # Display the original and equalized images
    return median_image

def median_filtering(image_path, window_size):
    image = cv2.imread(image_path)

    # Get image shape
    height, width, depth = image.shape

  
    median_image = image.copy()  # Create a copy of the original image

    # Iterate over each color channel
    for c in range(depth):
            # Extract the current color channel
        channel = image[:, :, c]
            
            # Perform histogram equalization on the channel
        median_image_channel = apply_median_filtering(channel, window_size)

            
            # Replace the channel in the equalized image
        median_image[:, :, c] = median_image_channel

    # Display the original and equalized images
    cv2.imshow('Original Image', image)
    cv2.imshow('Median filtered image', median_image)
    cv2.imwrite('median/ct_scan_eq.pnm',median_image)

    #with package
    median_blur_with_package = median_filter_with_package(image_path)
    cv2.imshow('Median filtered image with package', median_blur_with_package)
    cv2.imwrite('median/ct_scan_pa.pnm',median_blur_with_package)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# image_path = "ct_scan.pnm"
# window_size = 3
# median_filtering(image_path, window_size)
