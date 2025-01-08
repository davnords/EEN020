import numpy as np

def triangulate_3D_point_DLT(x1, x2, P1, P2):
    """
    Triangulate the 3D position of a point given two 2D correspondences and camera matrices.
    
    Args:
    x1 -- 2D point in image 1
    x2 -- 2D point in image 2
    P1 -- Camera matrix for image 1
    P2 -- Camera matrix for image 2
    
    Returns:
    X -- 3D position of the point
    valid -- True if the point is in front of both cameras, False otherwise
    """
    A = np.array([x1[0] * P1[2, :] - P1[0, :],
                  x1[1] * P1[2, :] - P1[1, :],
                  x2[0] * P2[2, :] - P2[0, :],
                  x2[1] * P2[2, :] - P2[1, :]])
    _, _, V = np.linalg.svd(A)
    X = V[-1, :3] / V [-1, -1]
    X_h = np.hstack((X, [1]))  # Homogeneous coordinates

    # Check if the point is in front of both cameras
    X1_cam = P1 @ X_h  # Point in camera 1's coordinates
    X2_cam = P2 @ X_h  # Point in camera 2's coordinates

    valid = X1_cam[2] > 0 and X2_cam[2] > 0  # z-coordinates should be positive

    return X, valid
