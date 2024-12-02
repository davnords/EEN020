import numpy as np

def points_in_front_of_cameras(X, P1, P2):
    """
    Check which points are in front of both cameras.
    
    Parameters:
    X (numpy.ndarray): 4xN homogeneous 3D points
    P1 (numpy.ndarray): First camera matrix (3x4)
    P2 (numpy.ndarray): Second camera matrix (3x4)
    
    Returns:
    numpy.ndarray: Boolean array indicating which points are in front of both cameras
    """
    # Camera center of first camera (which is at origin in its coordinate system)
    C1 = np.zeros(3)
    
    # Camera center of second camera
    C2 = -np.linalg.inv(P2[:, :3]) @ P2[:, 3]
    
    # Check points in front of first camera
    # Project points to first camera and check z coordinate
    X_cam1 = P1 @ X
    in_front_cam1 = X_cam1[2, :] > 0
    
    # Check points in front of second camera
    # Project points to second camera and check z coordinate
    X_cam2 = P2 @ X
    in_front_cam2 = X_cam2[2, :] > 0
    
    # Combine conditions
    return in_front_cam1 & in_front_cam2