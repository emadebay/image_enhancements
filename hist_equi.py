


import cv2
import numpy as np
from PIL import Image

def manual_histogram_equalization(channel):
    # Calculate histogram
    hist, _ = np.histogram(channel.flatten(), bins=256, range=[0, 255])
    
    # Calculate cumulative distribution function (CDF)
    cdf = hist.cumsum()
    #cdf normalized by a scaling factor of L -1
    #divided by the total number of pixels
    cdf_normalized = (cdf * 255 / cdf[-1]).astype(np.uint8) 
    # cdf_normalized = ((cdf ) + (cdf[channel]/ channel.size)).astype(np.uint8) 
    # print(cdf[-1])
    
    # Apply histogram equalization to each pixel value
    equalized_channel = cdf_normalized[channel]
    
    return equalized_channel.reshape(channel.shape)


def histogram_equalization_with_opencv(image_path):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(image)
    
    # Display the original and equalized images (optional)
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Equalized Image', equalized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return equalized_image

def histogram_equalization(image_path):
    image = cv2.imread(image_path)

    # Get image shape
    height, width, depth = image.shape

  
    equalized_image = image.copy()  # Create a copy of the original image

    # Iterate over each color channel
    for c in range(depth):
            # Extract the current color channel
        channel = image[:, :, c]
            
            # Perform histogram equalization on the channel
        equalized_channel = manual_histogram_equalization(channel)

            
            # Replace the channel in the equalized image
        equalized_image[:, :, c] = equalized_channel

    imgae_equalized_with_library = histogram_equalization_with_opencv(image_path)

    # Display the original and equalized images
    cv2.imshow('Original Image', image)
    #show equalized image implemented from scratch
    cv2.imshow('Equalized Image from scratch', equalized_image)
    #save
    cv2.imwrite("hist_equalization/ct_scan_eq.pnm", equalized_image)
    #show equalized image using packages
    cv2.imshow('Equalized Image with package', imgae_equalized_with_library)
    #save
    cv2.imwrite("hist_equalization/ct_scan_pa.pnm", imgae_equalized_with_library)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = "ct_scan.pnm"
histogram_equalization(image_path)



    

# Example usage
image_path = 'example.jpg'  # Path to the input image
histogram_equalization(image_path)