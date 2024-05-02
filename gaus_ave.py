


import cv2

import numpy as np


def gaussian_filter(size, sigma):
   
    if size % 2 == 0:
        raise ValueError("Size must be an odd number")

    filter = np.zeros((size, size), dtype=np.float32)
    center = size // 2

    for i in range(size):
        for j in range(size):
            filter[i, j] = np.exp(-((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2))

    # Normalize the filter
    filter /= np.sum(filter)

    return filter

def gaussian_blur_with_package(image_path, kernel_size=(3, 3), sigma=0):
    # Read the image
    image = cv2.imread(image_path)
    
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigmaX=sigma)
    
    return blurred_image

def apply_gaussian_averaging_to_channel(channel, size, sigma):
  
    # Generate Gaussian filter
    kernel = gaussian_filter(size, sigma)

    # Get dimensions of the channel
    rows, cols = channel.shape

    # Define padding size
    pad_size = size // 2

    # Pad the channel to handle border pixels
    padded_channel = np.pad(channel, pad_size, mode='constant')

    # Apply Gaussian averaging filter
    smoothed_channel = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            # Extract local region
            local_region = padded_channel[i:i+size, j:j+size]

            # Apply filter
            smoothed_channel[i, j] = np.sum(local_region * kernel)

    return smoothed_channel

def gaussian_averaging_to_be_used(image_path, kernel_size, sigma):
    image = cv2.imread(image_path)

    # Get image shape
    height, width, depth = image.shape

  
    gausian_image = image.copy()  # Create a copy of the original image

    # Iterate over each color channel
    for c in range(depth):
            # Extract the current color channel
        channel = image[:, :, c]
            
            # Perform histogram equalization on the channel
        gausian_image_channel = apply_gaussian_averaging_to_channel(channel, kernel_size, sigma)

            
            # Replace the channel in the equalized image
        gausian_image[:, :, c] = gausian_image_channel
    
    return gausian_image

def gaussian_averaging(image_path, kernel_size, sigma):
    image = cv2.imread(image_path)

    # Get image shape
    height, width, depth = image.shape

  
    gausian_image = image.copy()  # Create a copy of the original image

    # Iterate over each color channel
    for c in range(depth):
            # Extract the current color channel
        channel = image[:, :, c]
            
            # Perform histogram equalization on the channel
        gausian_image_channel = apply_gaussian_averaging_to_channel(channel, kernel_size, sigma)

            
            # Replace the channel in the equalized image
        gausian_image[:, :, c] = gausian_image_channel

    # Display the original and gausian images
    cv2.imshow('Original Image', image)
    cv2.imshow('Gausian averaged image', gausian_image)
    cv2.imwrite('gausian/ct_scan_3_15_eq.pnm', gausian_image)

    #with package
    gausian_image_with_pa = gaussian_blur_with_package(image_path, (kernel_size,kernel_size), sigma)
    cv2.imshow('Gausian averaged image with packaged', gausian_image_with_pa)
    cv2.imwrite('gausian/ct_scan_3_15_pa.pnm', gausian_image_with_pa)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# image_path = "ct_scan.pnm"
# kernel_size, sigma = 3,15
# gaussian_averaging(image_path, kernel_size, sigma)
