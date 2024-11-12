import numpy as np
from pflat import pflat
import cv2

def imwarp(image, corners, Htot):
    """
    Implementation of something akin to the projectform2d + imwarp functions in MATLAB.
    
    Parameters:
    image: Input image
    corners: 3x4 array of homogenous corner points
    Htot: 3x3 homography matrix
    
    Returns:
    transformed_image: Warped image
    transformed_corners: Points in the new coordinate system
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Get corners of the original image
    corners = np.array([
        [0, 0, 1],
        [width-1, 0, 1],
        [width-1, height-1, 1],
        [0, height-1, 1]
    ], dtype=np.float32).T
    
    # Transform corners
    transformed_corners = pflat(np.dot(Htot, corners))
    
    # Get bounds of transformed image
    x_min = np.floor(transformed_corners[0].min()).astype(int)
    x_max = np.ceil(transformed_corners[0].max()).astype(int)
    y_min = np.floor(transformed_corners[1].min()).astype(int)
    y_max = np.ceil(transformed_corners[1].max()).astype(int)
    
    # Create translation matrix to shift to positive coordinates
    T = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])
    
    # Combine translation with original homography
    H_final = T @ Htot
    
    # Calculate new image dimensions
    new_width = x_max - x_min + 1
    new_height = y_max - y_min + 1
    
    # Warp image
    transformed_image = cv2.warpPerspective(
        image_rgb, 
        H_final, 
        (new_width, new_height),
        flags=cv2.INTER_LINEAR
    )
    
    # Transform points using the same transformation
    transformed_corners = np.dot(H_final, corners)
    transformed_corners = transformed_corners / transformed_corners[2]
    
    return transformed_image, transformed_corners
