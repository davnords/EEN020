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
    """
    A = np.array([x1[0] * P1[2, :] - P1[0, :],
                  x1[1] * P1[2, :] - P1[1, :],
                  x2[0] * P2[2, :] - P2[0, :],
                  x2[1] * P2[2, :] - P2[1, :]])
    _, _, V = np.linalg.svd(A)
    X = V[-1, :3] / V [-1, -1]

    sv = V[-1, -1] # smallest singular value
    return X, sv
