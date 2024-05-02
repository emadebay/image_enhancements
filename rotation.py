


import cv2

import numpy as np





def rotate_channel(channel, theta):
    # Calculate rotation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]])

    # Get height and width of the channel
    height, width = channel.shape

    # Calculate center of the image
    center_x = width / 2
    center_y = height / 2

    # Create an array to store rotated channel
    rotated_channel = np.zeros_like(channel)

    # Iterate over each pixel in the rotated channel
    for x_rotated in range(height):
        for y_rotated in range(width):

            #define the middleof the image as origin
            #convert the image coordinates to x and y coordinates
            x_coord = x_rotated - center_x
            y_coord = center_y - y_rotated
            # Apply inverse rotation to pixel coordinates
            #get the corresponding x and y coordinates
            x_old, y_old = np.dot(np.linalg.inv(rotation_matrix), np.array([[x_coord], [y_coord]]))

            #convert it back to the image coordinates
            x_old = x_old + center_x
            y_old = center_y - y_old

            # Convert coordinates to integers
            x_old_int = int(np.round(x_old))
            y_old_int = int(np.round(y_old))

            # Check if the transformed coordinates are within bounds
            if 0 <= x_old_int < height and 0 <= y_old_int < width:
                # Copy pixel value from the old image to the rotated channel
                rotated_channel[x_rotated, y_rotated] = channel[x_old_int, y_old_int]

    return rotated_channel

def rotate_image_with_package(image_path, angle):
    # Read the image
    image = cv2.imread(image_path)
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate rotation matrix
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    return rotated_image

def rotation(image_path, degree):
    image = cv2.imread(image_path)

    # Get image shape
    height, width, depth = image.shape

  
    rt_image = image.copy()  # Create a copy of the original image

    # Iterate over each color channel
    for c in range(depth):
            # Extract the current color channel
        channel = image[:, :, c]
            
        #perform rotation on channel
        rt_image_channel = rotate_channel(channel, degree)

            
            # Replace the channel in the equalized image
        rt_image[:, :, c] = rt_image_channel

    # Display the original and equalized images
    cv2.imshow('Original Image', image)
    cv2.imshow('rotated Image', rt_image)
    cv2.imwrite('rotation/ct_scan_eq45.pnm', rt_image)

    #with package
    rotate_image_with_pa = rotate_image_with_package(image_path, degree)
    cv2.imshow('rotated Image with package', rotate_image_with_pa)
    cv2.imwrite('rotation/ct_scan_pa45.pnm', rotate_image_with_pa)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = "ct_scan.pnm"
deg = 45
rotation(image_path, deg)
